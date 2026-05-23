pub mod in_memory;

use async_trait::async_trait;

use crate::error::KovaError;
use crate::models::ConversationMessage;

/// Trait for storing and retrieving conversation history.
///
/// Implementations must be `Send + Sync` to allow safe concurrent usage.
/// All methods are async to support future persistent storage backends.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Add a message to the conversation.
    async fn add_message(
        &self,
        conversation_id: &str,
        message: ConversationMessage,
    ) -> Result<(), KovaError>;

    /// Retrieve the full message history for a conversation.
    async fn get_history(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>, KovaError>;

    /// Clear all messages for a conversation.
    async fn clear(&self, conversation_id: &str) -> Result<(), KovaError>;
}
