#[cfg(feature = "bedrock")]
pub mod bedrock;
pub mod config;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(any(
    feature = "openai",
    feature = "gemini",
    feature = "ollama",
    feature = "bedrock"
))]
pub(crate) mod http;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;

use crate::error::KovaError;
use crate::models::{
    ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, StreamEvent, ToolDefinition,
};

/// Retry policy for provider calls made by the agent.
///
/// Applied to every `chat_completion` call and to *establishing* a
/// streaming response; failures mid-stream are never retried. Only errors
/// where [`KovaError::is_retryable`](crate::error::KovaError::is_retryable)
/// returns true (connection failures, timeouts, 408/429/5xx) trigger a
/// retry. Backoff doubles per attempt, capped at `max_backoff`.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryConfig {
    /// Maximum retries after the initial attempt (default: 2).
    pub max_retries: u32,
    /// Backoff before the first retry; doubles each retry (default: 500ms).
    pub initial_backoff: Duration,
    /// Upper bound on a single backoff sleep (default: 10s).
    pub max_backoff: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(10),
        }
    }
}

impl RetryConfig {
    /// A policy that never retries.
    pub fn disabled() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Backoff duration before retry number `attempt` (0-based).
    pub(crate) fn backoff(&self, attempt: u32) -> Duration {
        self.initial_backoff
            .saturating_mul(2u32.saturating_pow(attempt))
            .min(self.max_backoff)
    }
}

/// LLM provider abstraction.
///
/// Implementations must be `Send + Sync` to allow safe concurrent usage
/// across async tasks and threads. The trait is object-safe, supporting
/// dynamic dispatch via `dyn LlmProvider`.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a chat completion request and receive a full response.
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError>;

    /// Send a chat completion request and receive a stream of response chunks.
    async fn chat_completion_stream(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError>;

    /// List available models from the provider.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError>;
}
