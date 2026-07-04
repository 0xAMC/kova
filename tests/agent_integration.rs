mod mock;

use std::sync::Arc;

use kova_sdk::agent::AgentBuilder;
use kova_sdk::error::KovaError;
use kova_sdk::models::*;

use mock::provider::{MockLlmProvider, make_text_response, run_text};

// ── Basic run loop ─────────────────────────────────────────────────

#[tokio::test]
async fn agent_run_returns_assistant_text() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        "Hello from the LLM!",
    )));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .expect("build should succeed");

    let reply = run_text(&agent, "Hi there").await.unwrap();
    assert_eq!(reply, "Hello from the LLM!");
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test]
async fn agent_run_with_system_prompt() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        "I am helpful.",
    )));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .system_prompt("You are a helpful assistant.")
        .build()
        .expect("build should succeed");

    let reply = run_text(&agent, "Who are you?").await.unwrap();
    assert_eq!(reply, "I am helpful.");
}

#[tokio::test]
async fn agent_run_multiple_turns() {
    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_text_response("First reply"),
        make_text_response("Second reply"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .unwrap();

    let r1 = run_text(&agent, "msg 1").await.unwrap();
    let r2 = run_text(&agent, "msg 2").await.unwrap();

    assert_eq!(r1, "First reply");
    assert_eq!(r2, "Second reply");
    assert_eq!(provider.call_count(), 2);
}

#[tokio::test]
async fn agent_run_empty_content_returns_empty_string() {
    // A response with no text content blocks should return "".
    let resp = ModelResponse {
        content: vec![],
        stop_reason: StopReason::EndTurn,
        usage: None,
        thinking: None,
    };

    let provider = Arc::new(MockLlmProvider::with_response(resp));

    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let reply = run_text(&agent, "hello").await.unwrap();
    assert_eq!(reply, "");
}

// ── AgentBuilder validation ────────────────────────────────────────

#[test]
fn agent_builder_fails_without_provider() {
    let result = AgentBuilder::new().build();

    match result {
        Err(KovaError::Build(msg)) => {
            assert!(
                msg.contains("LlmProvider"),
                "error should mention LlmProvider, got: {msg}"
            );
        }
        Err(other) => panic!("expected KovaError::Build, got: {other:?}"),
        Ok(_) => panic!("expected build to fail without a provider"),
    }
}

#[test]
fn agent_builder_succeeds_with_provider() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response("ok")));
    let result = AgentBuilder::new().provider(provider).build();
    assert!(result.is_ok());
}

// ── Compile-time Send + Sync assertions ────────────────────────────

#[test]
fn _assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<kova_sdk::agent::Agent>();
    assert_send_sync::<kova_sdk::provider::openai::OpenAiCompatibleProvider>();
}

// ── Stateless run() API ────────────────────────────────────────────

mod stateless_run {
    use std::sync::Arc;

    use kova_sdk::agent::AgentBuilder;
    use kova_sdk::models::*;

    use crate::mock::provider::{MockLlmProvider, make_text_response, make_tool_call_response};
    use crate::mock::tool::MockTool;

    fn user(text: &str) -> ConversationMessage {
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    #[tokio::test]
    async fn run_returns_new_messages() {
        let provider = Arc::new(MockLlmProvider::with_responses(vec![
            make_tool_call_response(vec![(
                "tc-1".to_string(),
                "greet".to_string(),
                serde_json::json!({}),
            )]),
            make_text_response("done"),
        ]));
        let agent = AgentBuilder::new()
            .provider(provider)
            .tool(Arc::new(MockTool::new("greet", "hi")))
            .build()
            .unwrap();

        let history = vec![user("go")];
        let response = agent.run(&history).await.unwrap();

        assert_eq!(response.text, "done");
        // assistant tool-use + tool result + final assistant message
        assert_eq!(response.new_messages.len(), 3);
        assert_eq!(response.new_messages[0].role, Role::Assistant);
        assert_eq!(response.new_messages[1].role, Role::Tool);
        assert_eq!(response.new_messages[2].role, Role::Assistant);
        assert_eq!(response.llm_calls, 2);
        // Usage aggregated across both provider calls (10/5 each).
        assert_eq!(response.usage.input_tokens, 20);
        assert_eq!(response.usage.output_tokens, 10);
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn run_output_feeds_next_run_as_history() {
        let provider = Arc::new(MockLlmProvider::with_responses(vec![make_text_response(
            "first",
        )]));
        let agent = AgentBuilder::new().provider(provider).build().unwrap();

        let mut history = vec![user("hello")];
        let response = agent.run(&history).await.unwrap();
        history.extend(response.new_messages);
        history.push(user("again"));

        // Second turn over caller-threaded history works without memory.
        let response = agent.run(&history).await.unwrap();
        assert_eq!(response.text, "first");
    }



}

// ── Retry layer ────────────────────────────────────────────────────

mod retries {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use kova_sdk::agent::AgentBuilder;
    use kova_sdk::error::KovaError;
    use kova_sdk::models::*;
    use kova_sdk::provider::{LlmProvider, RetryConfig};

    use crate::mock::provider::{make_text_response, run_text};

    /// Fails the first `failures` calls with a retryable 429, then succeeds.
    struct FlakyProvider {
        failures: usize,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmProvider for FlakyProvider {
        async fn chat_completion(
            &self,
            _messages: &[ConversationMessage],
            _tools: &[ToolDefinition],
            _config: &InferenceConfig,
        ) -> Result<ModelResponse, KovaError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.failures {
                Err(KovaError::provider_http(429, None, "rate limited"))
            } else {
                Ok(make_text_response("recovered"))
            }
        }

        async fn chat_completion_stream(
            &self,
            _messages: &[ConversationMessage],
            _tools: &[ToolDefinition],
            _config: &InferenceConfig,
        ) -> Result<
            std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, KovaError>> + Send>>,
            KovaError,
        > {
            Err(KovaError::provider_http(400, None, "unused"))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
            Ok(vec![])
        }
    }

    fn fast_retries(max: u32) -> RetryConfig {
        RetryConfig {
            max_retries: max,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(4),
        }
    }

    #[tokio::test]
    async fn transient_provider_failure_is_retried() {
        let provider = Arc::new(FlakyProvider {
            failures: 2,
            calls: AtomicUsize::new(0),
        });
        let agent = AgentBuilder::new()
            .provider(provider.clone())
            .retry_config(fast_retries(2))
            .build()
            .unwrap();

        let reply = run_text(&agent, "hi").await.unwrap();
        assert_eq!(reply, "recovered");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retries_exhausted_returns_error() {
        let provider = Arc::new(FlakyProvider {
            failures: 5,
            calls: AtomicUsize::new(0),
        });
        let agent = AgentBuilder::new()
            .provider(provider.clone())
            .retry_config(fast_retries(1))
            .build()
            .unwrap();

        let err = run_text(&agent, "hi").await.unwrap_err();
        assert!(err.is_retryable());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2, "1 try + 1 retry");
    }

    #[tokio::test]
    async fn non_retryable_error_fails_immediately() {
        struct BadRequestProvider {
            calls: AtomicUsize,
        }

        #[async_trait]
        impl LlmProvider for BadRequestProvider {
            async fn chat_completion(
                &self,
                _messages: &[ConversationMessage],
                _tools: &[ToolDefinition],
                _config: &InferenceConfig,
            ) -> Result<ModelResponse, KovaError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Err(KovaError::provider_http(400, None, "bad request"))
            }

            async fn chat_completion_stream(
                &self,
                _messages: &[ConversationMessage],
                _tools: &[ToolDefinition],
                _config: &InferenceConfig,
            ) -> Result<
                std::pin::Pin<
                    Box<dyn futures::Stream<Item = Result<StreamEvent, KovaError>> + Send>,
                >,
                KovaError,
            > {
                Err(KovaError::provider_invalid("unused"))
            }

            async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
                Ok(vec![])
            }
        }

        let provider = Arc::new(BadRequestProvider {
            calls: AtomicUsize::new(0),
        });
        let agent = AgentBuilder::new()
            .provider(provider.clone())
            .retry_config(fast_retries(3))
            .build()
            .unwrap();

        let err = run_text(&agent, "hi").await.unwrap_err();
        assert!(!err.is_retryable());
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            1,
            "no retries on 400"
        );
    }
}

// ── Cancellation ───────────────────────────────────────────────────

mod cancellation {
    use super::*;
    use async_trait::async_trait;
    use futures::Stream;
    use kova_sdk::provider::LlmProvider;
    use std::pin::Pin;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    /// Provider that never responds — cancellation must cut it off.
    struct HangingProvider;

    #[async_trait]
    impl LlmProvider for HangingProvider {
        async fn chat_completion(
            &self,
            _messages: &[ConversationMessage],
            _tools: &[ToolDefinition],
            _config: &InferenceConfig,
        ) -> Result<ModelResponse, KovaError> {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            unreachable!("cancelled before completion")
        }

        async fn chat_completion_stream(
            &self,
            _messages: &[ConversationMessage],
            _tools: &[ToolDefinition],
            _config: &InferenceConfig,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError>
        {
            Ok(Box::pin(futures::stream::pending()))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn run_cancellable_aborts_hung_provider_call() {
        let agent = AgentBuilder::new()
            .provider(Arc::new(HangingProvider))
            .build()
            .unwrap();

        let cancel = CancellationToken::new();
        let canceller = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            canceller.cancel();
        });

        let start = std::time::Instant::now();
        let err = agent
            .run_cancellable(&[], cancel)
            .await
            .expect_err("cancelled turn must error");
        assert!(matches!(err, KovaError::Cancelled), "got {err:?}");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "must abort promptly"
        );
    }

    #[tokio::test]
    async fn run_stream_cancellable_ends_with_cancelled_mid_stream() {
        use futures::StreamExt;

        let agent = AgentBuilder::new()
            .provider(Arc::new(HangingProvider))
            .build()
            .unwrap();

        let cancel = CancellationToken::new();
        let canceller = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            canceller.cancel();
        });

        let stream = agent.run_stream_cancellable(&[], cancel);
        futures::pin_mut!(stream);
        let start = std::time::Instant::now();
        let mut saw_cancelled = false;
        while let Some(event) = stream.next().await {
            if let Err(KovaError::Cancelled) = event {
                saw_cancelled = true;
            }
        }
        assert!(saw_cancelled, "stream must surface Cancelled");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "must abort promptly"
        );
    }

    #[tokio::test]
    async fn pre_cancelled_token_fails_immediately() {
        let agent = AgentBuilder::new()
            .provider(Arc::new(HangingProvider))
            .build()
            .unwrap();

        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = agent.run_cancellable(&[], cancel).await.unwrap_err();
        assert!(matches!(err, KovaError::Cancelled), "got {err:?}");
    }
}

// ── Structured output (run_structured) ─────────────────────────────

#[tokio::test]
async fn run_structured_parses_schema_constrained_text() {
    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Route {
        route: String,
        confidence: f64,
    }

    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        r#"{"route": "billing", "confidence": 0.9}"#,
    )));
    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let format = ResponseFormat::named(
        "route",
        serde_json::json!({
            "type": "object",
            "properties": {
                "route": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["route", "confidence"],
            "additionalProperties": false
        }),
    );
    let messages = vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "Which team?".into(),
        }],
    }];
    let (route, response) = agent
        .run_structured::<Route>(&messages, format)
        .await
        .unwrap();
    assert_eq!(route.route, "billing");
    assert!(route.confidence > 0.8);
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn run_structured_rejects_nonconforming_text() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        "sorry, plain prose",
    )));
    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let messages = vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text { text: "go".into() }],
    }];
    let err = agent
        .run_structured::<serde_json::Value>(
            &messages,
            ResponseFormat::new(serde_json::json!({"type": "object"})),
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("structured output"),
        "unexpected error: {err}"
    );
}

// ── Context budget ─────────────────────────────────────────────────

#[tokio::test]
async fn context_budget_fails_before_provider_call() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response("hi")));
    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .context_budget(5) // ~20 chars — the message below exceeds it
        .build()
        .unwrap();

    let messages = vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "x".repeat(500),
        }],
    }];
    let err = agent.run(&messages).await.unwrap_err();
    assert!(
        matches!(err, KovaError::ContextBudgetExceeded { measured, budget: 5 } if measured > 5),
        "unexpected: {err}"
    );
    // The doomed provider call was never made.
    assert_eq!(provider.call_count(), 0);
}

#[tokio::test]
async fn context_budget_allows_small_prompts() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response("ok")));
    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .context_budget(10_000)
        .build()
        .unwrap();

    let messages = vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text { text: "hi".into() }],
    }];
    let response = agent.run(&messages).await.unwrap();
    assert_eq!(response.text, "ok");
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test]
async fn default_count_tokens_uses_heuristic() {
    use kova_sdk::provider::{LlmProvider, heuristic_count_tokens};

    let provider = MockLlmProvider::with_response(make_text_response("unused"));
    let messages = vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "a".repeat(400),
        }],
    }];
    let counted = provider.count_tokens(&messages, &[]).await.unwrap();
    assert_eq!(counted, heuristic_count_tokens(&messages, &[]));
    // ~400 chars / 4 ≈ 100 tokens (+ overhead)
    assert!((100..=110).contains(&counted), "got {counted}");
}
