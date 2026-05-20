mod mock;

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use tokio::sync::Mutex;

use kova::agent::AgentBuilder;
use kova::error::KovaError;
use kova::models::*;
use kova::provider::LlmProvider;
use kova::streaming::StreamingHandler;

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
        Err(KovaError::Provider {
            message: "use chat_completion_stream".into(),
            status_code: None,
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
        Ok(StreamEvent::ContentDelta { text: "Hello".into() }),
        Ok(StreamEvent::ContentDelta { text: " world".into() }),
        Ok(StreamEvent::ContentDelta { text: "!".into() }),
        Ok(StreamEvent::StopEvent { stop_reason: StopReason::EndTurn }),
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
    assert_eq!(chunks[0], StreamEvent::ContentDelta { text: "Hello".into() });
    assert_eq!(chunks[1], StreamEvent::ContentDelta { text: " world".into() });
    assert_eq!(chunks[2], StreamEvent::ContentDelta { text: "!".into() });
    assert!(matches!(chunks[3], StreamEvent::StopEvent { .. }));

    assert!(*handler.completed.lock().await);
}

#[tokio::test]
async fn test_streaming_on_complete_called() {
    let events = vec![
        Ok(StreamEvent::ContentDelta { text: "done".into() }),
        Ok(StreamEvent::StopEvent { stop_reason: StopReason::EndTurn }),
    ];

    let provider = Arc::new(StreamingMockProvider::new(events));
    let handler = Arc::new(MockStreamingHandler::new());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .streaming_handler(handler.clone() as Arc<dyn StreamingHandler>)
        .build()
        .unwrap();

    let _ = agent.chat_stream("conv2", "test").await.unwrap();
    assert!(*handler.completed.lock().await, "on_complete should have been called");
}

#[tokio::test]
async fn test_streaming_on_error_on_stream_failure() {
    // Simulate a connection drop mid-stream by returning an error item.
    let events: Vec<Result<StreamEvent, KovaError>> = vec![
        Ok(StreamEvent::ContentDelta { text: "partial".into() }),
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
        Ok(StreamEvent::StopEvent { stop_reason: StopReason::EndTurn }),
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
        .map(|t| Ok(StreamEvent::ContentDelta { text: t.to_string() }))
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
