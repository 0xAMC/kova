mod mock;

use std::sync::Arc;

use kova_sdk::agent::AgentBuilder;
use kova_sdk::error::KovaError;
use kova_sdk::models::*;

use mock::provider::{MockLlmProvider, make_text_response};

// ── Basic chat loop ────────────────────────────────────────────────

#[tokio::test]
async fn agent_chat_returns_assistant_text() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        "Hello from the LLM!",
    )));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .expect("build should succeed");

    let reply = agent.chat("conv-1", "Hi there").await.unwrap();
    assert_eq!(reply, "Hello from the LLM!");
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test]
async fn agent_chat_with_system_prompt() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        "I am helpful.",
    )));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .system_prompt("You are a helpful assistant.")
        .build()
        .expect("build should succeed");

    let reply = agent.chat("conv-1", "Who are you?").await.unwrap();
    assert_eq!(reply, "I am helpful.");
}

#[tokio::test]
async fn agent_chat_multiple_turns() {
    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_text_response("First reply"),
        make_text_response("Second reply"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .unwrap();

    let r1 = agent.chat("conv-1", "msg 1").await.unwrap();
    let r2 = agent.chat("conv-1", "msg 2").await.unwrap();

    assert_eq!(r1, "First reply");
    assert_eq!(r2, "Second reply");
    assert_eq!(provider.call_count(), 2);
}

#[tokio::test]
async fn agent_chat_empty_content_returns_empty_string() {
    // A response with no text content blocks should return "".
    let resp = ModelResponse {
        content: vec![],
        stop_reason: StopReason::EndTurn,
        usage: None,
        thinking: None,
    };

    let provider = Arc::new(MockLlmProvider::with_response(resp));

    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let reply = agent.chat("conv-1", "hello").await.unwrap();
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

    use async_trait::async_trait;
    use kova_sdk::agent::AgentBuilder;
    use kova_sdk::error::KovaError;
    use kova_sdk::memory::MemoryStore;
    use kova_sdk::memory::in_memory::InMemoryStore;
    use kova_sdk::models::*;
    use kova_sdk::provider::LlmProvider;

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
    async fn run_returns_new_messages_without_touching_memory() {
        let provider = Arc::new(MockLlmProvider::with_responses(vec![
            make_tool_call_response(vec![(
                "tc-1".to_string(),
                "greet".to_string(),
                serde_json::json!({}),
            )]),
            make_text_response("done"),
        ]));
        let memory = Arc::new(InMemoryStore::new());
        let agent = AgentBuilder::new()
            .provider(provider)
            .tool(Arc::new(MockTool::new("greet", "hi")))
            .memory(memory.clone())
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

        // run() is stateless: memory untouched.
        let stored = memory.get_history("any").await.unwrap();
        assert!(stored.is_empty());
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

    struct FailingProvider;

    #[async_trait]
    impl LlmProvider for FailingProvider {
        async fn chat_completion(
            &self,
            _messages: &[ConversationMessage],
            _tools: &[ToolDefinition],
            _config: &InferenceConfig,
        ) -> Result<ModelResponse, KovaError> {
            Err(KovaError::Provider {
                message: "boom".into(),
                status_code: Some(500),
            })
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
            Err(KovaError::Provider {
                message: "boom".into(),
                status_code: Some(500),
            })
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn failed_chat_turn_leaves_memory_unchanged() {
        let memory = Arc::new(InMemoryStore::new());
        let agent = AgentBuilder::new()
            .provider(Arc::new(FailingProvider))
            .memory(memory.clone())
            .retry_config(kova_sdk::provider::RetryConfig::disabled())
            .build()
            .unwrap();

        let err = agent.chat("conv-1", "hello").await.unwrap_err();
        assert!(matches!(err, KovaError::Provider { .. }));

        // No dangling user message — the failed turn was never persisted.
        let history = memory.get_history("conv-1").await.unwrap();
        assert!(
            history.is_empty(),
            "failed turn must not corrupt the conversation, got {history:?}"
        );

        // A subsequent successful chat is unaffected. (Provider always fails
        // here, so just assert the error again for determinism.)
        assert!(agent.chat("conv-1", "hello").await.is_err());
    }

    #[tokio::test]
    async fn chat_response_exposes_usage_and_messages() {
        let provider = Arc::new(MockLlmProvider::with_responses(vec![make_text_response(
            "hi there",
        )]));
        let memory = Arc::new(InMemoryStore::new());
        let agent = AgentBuilder::new()
            .provider(provider)
            .memory(memory.clone())
            .build()
            .unwrap();

        let response = agent.chat_response("conv-1", "hello").await.unwrap();
        assert_eq!(response.text, "hi there");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.new_messages.len(), 1);

        // Session layer persisted user + assistant.
        let history = memory.get_history("conv-1").await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, Role::User);
        assert_eq!(history[1].role, Role::Assistant);
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

    use crate::mock::provider::make_text_response;

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
                Err(KovaError::Provider {
                    message: "rate limited".into(),
                    status_code: Some(429),
                })
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
            Err(KovaError::Provider {
                message: "unused".into(),
                status_code: None,
            })
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

        let reply = agent.chat("conv-1", "hi").await.unwrap();
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

        let err = agent.chat("conv-1", "hi").await.unwrap_err();
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
                Err(KovaError::Provider {
                    message: "bad request".into(),
                    status_code: Some(400),
                })
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
                Err(KovaError::Provider {
                    message: "unused".into(),
                    status_code: None,
                })
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

        let err = agent.chat("conv-1", "hi").await.unwrap_err();
        assert!(!err.is_retryable());
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            1,
            "no retries on 400"
        );
    }
}
