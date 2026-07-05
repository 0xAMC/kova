//! Text embeddings: a provider-agnostic trait plus OpenAI-compatible and
//! Ollama implementations.
//!
//! kova ships no vector store — this is the input seam for retrieval systems
//! built on top (chunking, indexing, and search belong to the host
//! application).

#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;

use async_trait::async_trait;

use crate::error::KovaError;

/// A source of text embeddings.
///
/// Implementations batch server-side where the API allows it; inputs map 1:1
/// onto output vectors, in order.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed `texts`, returning one vector per input in the same order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KovaError>;

    /// Vector dimensionality, when the provider knows it up front
    /// (e.g. a fixed-dimension model). `None` = discover from the first call.
    fn dimensions(&self) -> Option<usize> {
        None
    }
}
