#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "bedrock")]
pub mod bedrock;
pub mod config;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(any(
    feature = "openai",
    feature = "gemini",
    feature = "ollama",
    feature = "bedrock",
    feature = "anthropic"
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
    /// Approximate the prompt size of `messages` + `tools` in tokens.
    ///
    /// The default is a cheap offline heuristic (~4 chars/token plus per-item
    /// overhead) — good enough for budget guards and compaction triggers.
    /// Providers with a native counting endpoint (Anthropic) override this
    /// with an exact, network-backed count.
    async fn count_tokens(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
    ) -> Result<u32, KovaError> {
        Ok(heuristic_count_tokens(messages, tools))
    }

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

/// Cheap offline token estimate: ~4 characters per token over all message
/// text/tool payloads and tool schemas, plus a small per-message overhead.
///
/// Deliberately provider-agnostic and slightly conservative — callers use it
/// for guardrails (context budgets, compaction triggers), not billing.
pub fn heuristic_count_tokens(
    messages: &[crate::models::ConversationMessage],
    tools: &[crate::models::ToolDefinition],
) -> u32 {
    use crate::models::ContentBlock;

    let mut chars: usize = 0;
    for msg in messages {
        chars += 8; // per-message formatting overhead (~2 tokens)
        for block in &msg.content {
            chars += match block {
                ContentBlock::Text { text } => text.len(),
                ContentBlock::ToolUse { name, input, .. } => {
                    name.len() + serde_json::to_string(input).map(|s| s.len()).unwrap_or(0) + 16
                }
                ContentBlock::ToolResult { content, .. } => content.len() + 16,
                ContentBlock::Thinking { thinking, .. } => thinking.len(),
            };
        }
    }
    for tool in tools {
        chars += tool.name.len()
            + tool.description.len()
            + serde_json::to_string(&tool.parameters)
                .map(|s| s.len())
                .unwrap_or(0)
            + 16;
    }
    (chars / 4) as u32
}
