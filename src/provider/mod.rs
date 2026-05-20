pub mod bedrock;
pub mod config;
pub mod openai;
pub(crate) mod http;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::KovaError;
use crate::models::{ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, StreamEvent, ToolDefinition};

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
