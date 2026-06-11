#[cfg(any(feature = "openai", feature = "gemini", feature = "ollama"))]
pub(crate) mod line_stream;
#[cfg(any(feature = "openai", feature = "gemini"))]
pub(crate) mod sse;

use async_trait::async_trait;

use crate::error::KovaError;
use crate::models::StreamEvent;

/// Callback-based handler for processing streaming LLM response events.
///
/// Consumers implement this trait to receive events as they arrive from
/// the provider stream. The handler must be `Send + Sync` to support
/// concurrent usage across async tasks.
#[async_trait]
pub trait StreamingHandler: Send + Sync {
    /// Called for each stream event as it arrives.
    async fn on_chunk(&self, event: &StreamEvent) -> Result<(), KovaError>;

    /// Called when the stream completes successfully.
    async fn on_complete(&self) -> Result<(), KovaError>;

    /// Called when the stream encounters an error.
    async fn on_error(&self, error: &KovaError);
}
