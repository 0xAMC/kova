use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::Stream;
use tokio::sync::Mutex;

use kova_sdk::error::KovaError;
use kova_sdk::models::*;
use kova_sdk::provider::LlmProvider;

/// A mock LLM provider that returns configurable responses.
///
/// Supports a sequence of responses — each call to `chat_completion`
/// pops the next response from the list. If the list is exhausted the
/// last response is reused. Tracks the number of calls received.
pub struct MockLlmProvider {
    responses: Vec<ModelResponse>,
    call_count: AtomicUsize,
}

impl MockLlmProvider {
    /// Create a mock that always returns the given response.
    pub fn with_response(response: ModelResponse) -> Self {
        Self {
            responses: vec![response],
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create a mock that returns responses in order, reusing the last
    /// one once the list is exhausted.
    pub fn with_responses(responses: Vec<ModelResponse>) -> Self {
        assert!(!responses.is_empty(), "must provide at least one response");
        Self {
            responses,
            call_count: AtomicUsize::new(0),
        }
    }

    /// How many times `chat_completion` has been called.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn chat_completion(
        &self,
        _messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let response_idx = idx.min(self.responses.len() - 1);
        Ok(self.responses[response_idx].clone())
    }

    async fn chat_completion_stream(
        &self,
        _messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError> {
        Err(KovaError::Stream("not yet implemented".into()))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        Ok(vec![])
    }
}

/// Helper to build a simple assistant text response.
pub fn make_text_response(text: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: Some(UsageStats {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            thinking_tokens: None,
            cache_read_tokens: None,
            cache_creation_tokens: None,
        }),
        thinking: None,
    }
}

/// Helper to build an assistant response containing tool calls.
pub fn make_tool_call_response(
    tool_calls: Vec<(String, String, serde_json::Value)>,
) -> ModelResponse {
    let content = tool_calls
        .into_iter()
        .map(|(id, name, input)| ContentBlock::ToolUse {
            id,
            name,
            input,
            provider_metadata: None,
        })
        .collect();
    ModelResponse {
        content,
        stop_reason: StopReason::ToolUse,
        usage: Some(UsageStats {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            thinking_tokens: None,
            cache_read_tokens: None,
            cache_creation_tokens: None,
        }),
        thinking: None,
    }
}

/// A mock LLM provider that captures every request it receives.
///
/// Like `MockLlmProvider` but stores a clone of each set of messages
/// so tests can inspect the messages the agent sent back (e.g. tool-role messages).
pub struct CapturingMockProvider {
    responses: Vec<ModelResponse>,
    call_count: AtomicUsize,
    captured: Arc<Mutex<Vec<Vec<ConversationMessage>>>>,
}

impl CapturingMockProvider {
    pub fn new(responses: Vec<ModelResponse>) -> Self {
        assert!(!responses.is_empty(), "must provide at least one response");
        Self {
            responses,
            call_count: AtomicUsize::new(0),
            captured: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return all captured message lists.
    pub async fn captured_requests(&self) -> Vec<Vec<ConversationMessage>> {
        self.captured.lock().await.clone()
    }
}

#[async_trait]
impl LlmProvider for CapturingMockProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        self.captured.lock().await.push(messages.to_vec());
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let response_idx = idx.min(self.responses.len() - 1);
        Ok(self.responses[response_idx].clone())
    }

    async fn chat_completion_stream(
        &self,
        _messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError> {
        Err(KovaError::Stream("not yet implemented".into()))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        Ok(vec![])
    }
}
