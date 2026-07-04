//! Integration tests for the pull-based streaming API (`Agent::run_stream`).
//!
//! `run_stream` is the streaming surface hosts consume: it yields
//! [`AgentEvent`]s (text/thinking deltas, tool lifecycle, and a final
//! `TurnCompleted`) as a turn executes, over caller-supplied history.

mod mock;

use std::sync::Arc;

use async_trait::async_trait;
use futures::{StreamExt, stream};
use tokio::sync::Mutex;

use kova_sdk::agent::{AgentBuilder, AgentEvent};
use kova_sdk::error::KovaError;
use kova_sdk::models::*;
use kova_sdk::provider::LlmProvider;

use mock::tool::MockTool;

fn user(text: &str) -> ConversationMessage {
    ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
    }
}

// ── StreamingMockProvider ──────────────────────────────────────────

/// A mock provider that returns pre-configured `StreamEvent`s per call.
/// Each call drains the next batch (KovaError is not Clone, so events can't
/// be replayed).
struct StreamingMockProvider {
    batches: Mutex<Vec<Vec<Result<StreamEvent, KovaError>>>>,
}

impl StreamingMockProvider {
    fn one(events: Vec<Result<StreamEvent, KovaError>>) -> Self {
        Self {
            batches: Mutex::new(vec![events]),
        }
    }

    fn many(batches: Vec<Vec<Result<StreamEvent, KovaError>>>) -> Self {
        Self {
            batches: Mutex::new(batches),
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
        let mut batches = self.batches.lock().await;
        let events = if batches.is_empty() {
            Vec::new()
        } else {
            batches.remove(0)
        };
        Ok(Box::pin(stream::iter(events)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        Ok(vec![])
    }
}

/// Collect every event from a `run_stream` turn.
async fn collect(
    agent: &kova_sdk::agent::Agent,
    history: &[ConversationMessage],
) -> Vec<AgentEvent> {
    let stream = agent.run_stream(history);
    futures::pin_mut!(stream);
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.expect("stream event"));
    }
    events
}

// ── Tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn run_stream_delivers_text_deltas_in_order_then_completes() {
    let provider = Arc::new(StreamingMockProvider::one(vec![
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
    ]));
    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let events = collect(&agent, &[user("hi")]).await;

    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::TextDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["Hello", " world", "!"]);

    // The final event carries the assembled response.
    match events.last().unwrap() {
        AgentEvent::TurnCompleted { response } => {
            assert_eq!(response.text, "Hello world!");
            assert_eq!(response.stop_reason, StopReason::EndTurn);
        }
        other => panic!("expected TurnCompleted, got {other:?}"),
    }
}

#[tokio::test]
async fn run_stream_surfaces_thinking_deltas() {
    let provider = Arc::new(StreamingMockProvider::one(vec![
        Ok(StreamEvent::ThinkingDelta {
            text: "let me think".into(),
        }),
        Ok(StreamEvent::ContentDelta {
            text: "answer".into(),
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ]));
    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let events = collect(&agent, &[user("hi")]).await;
    assert!(events.contains(&AgentEvent::ThinkingDelta {
        text: "let me think".into()
    }));
}

#[tokio::test]
async fn run_stream_reports_usage_on_completion() {
    let provider = Arc::new(StreamingMockProvider::one(vec![
        Ok(StreamEvent::ContentDelta { text: "ok".into() }),
        Ok(StreamEvent::UsageEvent {
            input_tokens: 12,
            output_tokens: 3,
            thinking_tokens: None,
            cache_read_tokens: Some(8),
            cache_creation_tokens: None,
        }),
        Ok(StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn,
        }),
    ]));
    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let events = collect(&agent, &[user("hi")]).await;
    match events.last().unwrap() {
        AgentEvent::TurnCompleted { response } => {
            assert_eq!(response.usage.input_tokens, 12);
            assert_eq!(response.usage.output_tokens, 3);
            assert_eq!(response.usage.cache_read_tokens, Some(8));
        }
        other => panic!("expected TurnCompleted, got {other:?}"),
    }
}

#[tokio::test]
async fn run_stream_drives_a_tool_call_loop() {
    // First batch: the model streams a tool call. Second batch: the final text.
    let provider = Arc::new(StreamingMockProvider::many(vec![
        vec![
            Ok(StreamEvent::ToolUseDelta {
                id: "tc-1".into(),
                name: Some("greet".into()),
                input_delta: Some("{}".into()),
                provider_metadata: None,
                index: Some(0),
            }),
            Ok(StreamEvent::StopEvent {
                stop_reason: StopReason::ToolUse,
            }),
        ],
        vec![
            Ok(StreamEvent::ContentDelta {
                text: "done".into(),
            }),
            Ok(StreamEvent::StopEvent {
                stop_reason: StopReason::EndTurn,
            }),
        ],
    ]));
    let agent = AgentBuilder::new()
        .provider(provider)
        .tool(Arc::new(MockTool::new("greet", "hi")))
        .build()
        .unwrap();

    let events = collect(&agent, &[user("greet")]).await;

    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::ToolCallStarted { name, .. } if name == "greet"
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::ToolCallFinished { name, is_error: false, .. } if name == "greet"
    )));
    match events.last().unwrap() {
        AgentEvent::TurnCompleted { response } => {
            assert_eq!(response.text, "done");
            assert_eq!(response.llm_calls, 2);
            // assistant tool-use + tool result + final assistant message
            assert_eq!(response.new_messages.len(), 3);
        }
        other => panic!("expected TurnCompleted, got {other:?}"),
    }
}

#[tokio::test]
async fn run_stream_propagates_provider_stream_error() {
    let provider = Arc::new(StreamingMockProvider::one(vec![
        Ok(StreamEvent::ContentDelta {
            text: "partial".into(),
        }),
        Err(KovaError::Stream("connection dropped".into())),
    ]));
    let agent = AgentBuilder::new()
        .provider(provider)
        .retry_config(kova_sdk::provider::RetryConfig::disabled())
        .build()
        .unwrap();

    let history = [user("hi")];
    let stream = agent.run_stream(&history);
    futures::pin_mut!(stream);
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if item.is_err() {
            saw_error = true;
        }
    }
    assert!(saw_error, "provider stream error should surface to the caller");
}
