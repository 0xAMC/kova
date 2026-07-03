mod mock;

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use tokio::sync::Mutex;

use kova_sdk::agent::AgentBuilder;
use kova_sdk::error::KovaError;
use kova_sdk::models::*;
use kova_sdk::provider::LlmProvider;
use kova_sdk::streaming::StreamingHandler;

// ── MockStreamingHandler ───────────────────────────────────────────

/// Collects all stream events for later inspection.
struct MockStreamingHandler {
    chunks: Arc<Mutex<Vec<StreamEvent>>>,
    completed: Arc<Mutex<bool>>,
    errors: Arc<Mutex<Vec<String>>>,
}

impl MockStreamingHandler {
    fn new() -> Self {
        Self {
            chunks: Arc::new(Mutex::new(Vec::new())),
            completed: Arc::new(Mutex::new(false)),
            errors: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl StreamingHandler for MockStreamingHandler {
    async fn on_chunk(&self, event: &StreamEvent) -> Result<(), KovaError> {
        self.chunks.lock().await.push(event.clone());
        Ok(())
    }

    async fn on_complete(&self) -> Result<(), KovaError> {
        *self.completed.lock().await = true;
        Ok(())
    }

    async fn on_error(&self, error: &KovaError) {
        self.errors.lock().await.push(error.to_string());
    }
}

// ── StreamingMockProvider ──────────────────────────────────────────

/// A mock provider that returns a pre-configured stream of StreamEvents.
/// Events are taken (drained) on each call since KovaError is not Clone.
struct StreamingMockProvider {
    events: Mutex<Vec<Result<StreamEvent, KovaError>>>,
}

impl StreamingMockProvider {
    fn new(events: Vec<Result<StreamEvent, KovaError>>) -> Self {
        Self {
            events: Mutex::new(events),
        }
    }
}

#[async_trait]
impl LlmProvider for StreamingMockProvider {
    async fn chat_completion(
        &self,
        _messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        Err(KovaError::provider_invalid("use chat_completion_stream"))
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
        let events = std::mem::take(&mut *self.events.lock().await);
        Ok(Box::pin(stream::iter(events)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        Ok(vec![])
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_streaming_delivers_chunks_in_order() {
    let events = vec![
        Ok(StreamEvent::ContentDelta {
            text: "Hello".into(),
        }),
        Ok(StreamEvent::ContentDelta {
            text: " world".into(),
        }),
        Ok(StreamEvent::ContentDelta { text: "!".into() }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let result = agent.chat_stream("conv1", "hi").await.unwrap();
    assert_eq!(result, "Hello world!");

    let chunks = handler.chunks.lock().await;
    assert_eq!(chunks.len(), 4);
    assert_eq!(
        chunks[0],
        StreamEvent::ContentDelta {
            text: "Hello".into()
        }
    );
    assert_eq!(
        chunks[1],
        StreamEvent::ContentDelta {
            text: " world".into()
        }
    );
    assert_eq!(chunks[2], StreamEvent::ContentDelta { text: "!".into() });
    assert!(matches!(chunks[3], StreamEvent::StopEvent { .. }));

    assert!(*handler.completed.lock().await);
}

#[tokio::test]
async fn test_streaming_on_complete_called() {
    let events = vec![
        Ok(StreamEvent::ContentDelta {
            text: "done".into(),
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let _ = agent.chat_stream("conv2", "test").await.unwrap();
    assert!(
        *handler.completed.lock().await,
        "on_complete should have been called"
    );
}

#[tokio::test]
async fn test_streaming_on_error_on_stream_failure() {
    // Simulate a connection drop mid-stream by returning an error item.
    let events: Vec<Result<StreamEvent, KovaError>> = vec![
        Ok(StreamEvent::ContentDelta {
            text: "partial".into(),
        }),
        Err(KovaError::Stream("connection dropped".into())),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let result = agent.chat_stream("conv3", "test").await;
    assert!(result.is_err());

    let errors = handler.errors.lock().await;
    assert!(!errors.is_empty(), "on_error should have been called");
    assert!(errors[0].contains("connection dropped"));
}

#[tokio::test]
async fn test_streaming_requires_handler() {
    // Building an agent without a streaming handler and calling chat_stream
    // should return a Build error.
    let events = vec![
        Ok(StreamEvent::ContentDelta { text: "hi".into() }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .build()
        .unwrap();

    let result = agent.chat_stream("conv4", "test").await;
    assert!(matches!(result, Err(KovaError::Build(_))));
}

#[tokio::test]
async fn test_streaming_accumulates_text_correctly() {
    let chunks: Vec<&str> = vec!["The ", "quick ", "brown ", "fox"];
    let events: Vec<Result<StreamEvent, KovaError>> = chunks
        .iter()
        .map(|t| {
            Ok(StreamEvent::ContentDelta {
                text: t.to_string(),
            })
        })
        .chain(std::iter::once(Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        })))
        .collect();

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let result = agent.chat_stream("conv5", "test").await.unwrap();
    assert_eq!(result, "The quick brown fox");
}

#[tokio::test]
async fn test_streaming_thinking_delta_delivered_to_handler_not_in_text() {
    let events = vec![
        Ok(StreamEvent::ThinkingDelta {
            text: "internal thought".into(),
        }),
        Ok(StreamEvent::ContentDelta {
            text: "visible response".into(),
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let result = agent.chat_stream("conv6", "hi").await.unwrap();

    // Only content text is returned — thinking does not contribute
    assert_eq!(result, "visible response");

    // ThinkingDelta was delivered to the handler
    let chunks = handler.chunks.lock().await;
    assert!(
        chunks.iter().any(
            |e| matches!(e, StreamEvent::ThinkingDelta { text } if text == "internal thought")
        ),
        "ThinkingDelta should be delivered to the handler"
    );
}

#[tokio::test]
async fn test_streaming_thinking_delta_only_stream_returns_empty_text() {
    // A stream that only has thinking and no actual content should return ""
    let events = vec![
        Ok(StreamEvent::ThinkingDelta {
            text: "pure thought".into(),
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let result = agent.chat_stream("conv7", "think").await.unwrap();
    assert_eq!(result, "", "thinking-only stream should produce empty text");

    let chunks = handler.chunks.lock().await;
    assert!(
        chunks
            .iter()
            .any(|e| matches!(e, StreamEvent::ThinkingDelta { .. }))
    );
}

#[tokio::test]
async fn test_streaming_multiple_thinking_deltas_none_in_text() {
    let events = vec![
        Ok(StreamEvent::ThinkingDelta {
            text: "step 1 ".into(),
        }),
        Ok(StreamEvent::ThinkingDelta {
            text: "step 2".into(),
        }),
        Ok(StreamEvent::ContentDelta {
            text: "answer".into(),
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let result = agent.chat_stream("conv8", "q").await.unwrap();
    assert_eq!(result, "answer");

    let chunks = handler.chunks.lock().await;
    let thinking_chunks: Vec<_> = chunks
        .iter()
        .filter(|e| matches!(e, StreamEvent::ThinkingDelta { .. }))
        .collect();
    assert_eq!(
        thinking_chunks.len(),
        2,
        "both ThinkingDelta events should reach handler"
    );
}

#[tokio::test]
async fn test_streaming_usage_event_updates_last_turn_input_tokens() {
    let events = vec![
        Ok(StreamEvent::ContentDelta { text: "hi".into() }),
        Ok(StreamEvent::UsageEvent {
            input_tokens: 42,
            output_tokens: 7,
            thinking_tokens: None,
        cache_read_tokens: None,
        cache_creation_tokens: None,
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let _ = agent.chat_stream("conv9", "test").await.unwrap();
    assert_eq!(
        agent.last_turn_input_tokens(),
        42,
        "last_turn_input_tokens should be updated from UsageEvent"
    );
}

#[tokio::test]
async fn test_streaming_usage_event_delivered_to_handler() {
    let events = vec![
        Ok(StreamEvent::UsageEvent {
            input_tokens: 100,
            output_tokens: 50,
            thinking_tokens: None,
        cache_read_tokens: None,
        cache_creation_tokens: None,
        }),
        Ok(StreamEvent::ContentDelta {
            text: "done".into(),
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let _ = agent.chat_stream("conv10", "test").await.unwrap();

    let chunks = handler.chunks.lock().await;
    assert!(
        chunks.iter().any(|e| matches!(
            e,
            StreamEvent::UsageEvent {
                input_tokens: 100,
                output_tokens: 50,
                ..
            }
        )),
        "UsageEvent should be delivered to the handler"
    );
}

// ── run_stream (pull-based AgentEvent API) ─────────────────────────

mod run_stream_api {
    use super::*;
    use kova_sdk::agent::AgentEvent;
    use kova_sdk::models::ToolResult;
    use kova_sdk::tool::Tool;
    use std::collections::VecDeque;

    /// Mock provider returning one pre-configured stream per call.
    struct MultiStreamMockProvider {
        rounds: Mutex<VecDeque<Vec<Result<StreamEvent, KovaError>>>>,
    }

    impl MultiStreamMockProvider {
        fn new(rounds: Vec<Vec<Result<StreamEvent, KovaError>>>) -> Self {
            Self {
                rounds: Mutex::new(rounds.into()),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MultiStreamMockProvider {
        async fn chat_completion(
            &self,
            _messages: &[ConversationMessage],
            _tools: &[ToolDefinition],
            _config: &InferenceConfig,
        ) -> Result<ModelResponse, KovaError> {
            Err(KovaError::provider_invalid("use chat_completion_stream"))
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
            let round = self.rounds.lock().await.pop_front().unwrap_or_default();
            Ok(Box::pin(stream::iter(round)))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
            Ok(vec![])
        }
    }

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "greet"
        }
        fn description(&self) -> &str {
            "greets"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult, KovaError> {
            Ok(ToolResult {
                content: "hello!".to_string(),
                is_error: false,
            })
        }
    }

    fn user(text: &str) -> ConversationMessage {
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    #[tokio::test]
    async fn run_stream_yields_tool_events_deltas_and_completion() {
        let provider = Arc::new(MultiStreamMockProvider::new(vec![
            vec![
                Ok(StreamEvent::ToolUseDelta {
                    id: "tc-1".into(),
                    name: Some("greet".into()),
                    input_delta: Some("{}".into()),
                    provider_metadata: None,
                    index: None,
                }),
                Ok(StreamEvent::StopEvent {
                    stop_reason: StopReason::ToolUse,
                }),
            ],
            vec![
                Ok(StreamEvent::ContentDelta { text: "all".into() }),
                Ok(StreamEvent::ContentDelta {
                    text: " done".into(),
                }),
                Ok(StreamEvent::UsageEvent {
                    input_tokens: 7,
                    output_tokens: 3,
                    thinking_tokens: None,
        cache_read_tokens: None,
        cache_creation_tokens: None,
                }),
                Ok(StreamEvent::StopEvent {
                    stop_reason: StopReason::EndTurn,
                }),
            ],
        ]));

        let agent = AgentBuilder::new()
            .provider(provider)
            .tool(Arc::new(EchoTool))
            .build()
            .unwrap();

        let history = vec![user("hi")];
        let events: Vec<AgentEvent> = {
            use futures::StreamExt;
            let stream = agent.run_stream(&history);
            futures::pin_mut!(stream);
            stream.map(|r| r.expect("no stream errors")).collect().await
        };

        assert!(matches!(
            &events[0],
            AgentEvent::ToolCallStarted { id, name, .. } if id == "tc-1" && name == "greet"
        ));
        assert!(matches!(
            &events[1],
            AgentEvent::ToolCallFinished { id, name, result, is_error: false }
                if id == "tc-1" && name == "greet" && result == "hello!"
        ));
        assert!(matches!(&events[2], AgentEvent::TextDelta { text } if text == "all"));
        assert!(matches!(&events[3], AgentEvent::TextDelta { text } if text == " done"));
        match &events[4] {
            AgentEvent::TurnCompleted { response } => {
                assert_eq!(response.text, "all done");
                assert_eq!(response.new_messages.len(), 3);
                assert_eq!(response.llm_calls, 2);
                assert_eq!(response.usage.input_tokens, 7);
                assert_eq!(response.usage.output_tokens, 3);
            }
            other => panic!("expected TurnCompleted, got {other:?}"),
        }
        assert_eq!(events.len(), 5);
    }

    #[tokio::test]
    async fn run_stream_propagates_stream_errors() {
        let provider = Arc::new(MultiStreamMockProvider::new(vec![vec![
            Ok(StreamEvent::ContentDelta {
                text: "partial".into(),
            }),
            Err(KovaError::Stream("connection dropped".into())),
        ]]));

        let agent = AgentBuilder::new().provider(provider).build().unwrap();
        let history = vec![user("hi")];

        use futures::StreamExt;
        let stream = agent.run_stream(&history);
        futures::pin_mut!(stream);
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_error = true;
                break;
            }
        }
        assert!(saw_error, "stream error must surface to the consumer");
    }
}
