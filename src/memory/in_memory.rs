use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::Instrument;

use crate::error::KovaError;
use crate::models::{ConversationMessage, Role};

use super::MemoryStore;

/// In-memory conversation store backed by a `HashMap`.
///
/// Thread-safe via `Arc<RwLock<...>>`. When `max_messages` is set,
/// `get_history` truncates older messages while preserving the system
/// prompt if present.
pub struct InMemoryStore {
    conversations: Arc<RwLock<HashMap<String, Vec<ConversationMessage>>>>,
    max_messages: Option<usize>,
}

impl InMemoryStore {
    /// Create a new `InMemoryStore` with no message limit.
    pub fn new() -> Self {
        Self {
            conversations: Arc::new(RwLock::new(HashMap::new())),
            max_messages: None,
        }
    }

    /// Create a new `InMemoryStore` with a maximum message count.
    ///
    /// When the history exceeds this limit, older messages are truncated
    /// on retrieval, preserving the system prompt if present.
    pub fn with_max_messages(max_messages: usize) -> Self {
        Self {
            conversations: Arc::new(RwLock::new(HashMap::new())),
            max_messages: Some(max_messages),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn add_message(
        &self,
        conversation_id: &str,
        message: ConversationMessage,
    ) -> Result<(), KovaError> {
        let span = tracing::info_span!(
            "memory.add_message",
            conversation_id = conversation_id,
        );
        self.add_message_inner(conversation_id, message)
            .instrument(span)
            .await
    }

    async fn get_history(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>, KovaError> {
        let span = tracing::info_span!(
            "memory.get_history",
            conversation_id = conversation_id,
        );
        self.get_history_inner(conversation_id)
            .instrument(span)
            .await
    }

    async fn clear(&self, conversation_id: &str) -> Result<(), KovaError> {
        let span = tracing::info_span!(
            "memory.clear",
            conversation_id = conversation_id,
        );
        self.clear_inner(conversation_id)
            .instrument(span)
            .await
    }
}

impl InMemoryStore {
    async fn add_message_inner(
        &self,
        conversation_id: &str,
        message: ConversationMessage,
    ) -> Result<(), KovaError> {
        let mut convos = self.conversations.write().await;
        convos
            .entry(conversation_id.to_string())
            .or_default()
            .push(message);
        Ok(())
    }

    async fn get_history_inner(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>, KovaError> {
        let convos = self.conversations.read().await;
        let messages = match convos.get(conversation_id) {
            Some(msgs) => msgs.clone(),
            None => return Ok(Vec::new()),
        };

        let Some(max) = self.max_messages else {
            return Ok(messages);
        };

        if messages.len() <= max {
            return Ok(messages);
        }

        // Check if the first message is a system prompt.
        let has_system = matches!(messages.first(), Some(m) if m.role == Role::System);

        if has_system {
            // Keep system prompt + most recent (max - 1) messages.
            let system = messages[0].clone();
            let keep = max.saturating_sub(1);
            let start = messages.len().saturating_sub(keep);
            let mut result = vec![system];
            result.extend_from_slice(&messages[start..]);
            Ok(result)
        } else {
            // Keep most recent `max` messages.
            let start = messages.len().saturating_sub(max);
            Ok(messages[start..].to_vec())
        }
    }

    async fn clear_inner(&self, conversation_id: &str) -> Result<(), KovaError> {
        let mut convos = self.conversations.write().await;
        convos.remove(conversation_id);
        Ok(())
    }
}

#[cfg(test)]
#[path = "in_memory_tests.rs"]
mod in_memory_tests;

