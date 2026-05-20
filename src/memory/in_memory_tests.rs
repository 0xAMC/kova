#[cfg(test)]
mod tests {
    use crate::memory::in_memory::InMemoryStore;
    use crate::memory::MemoryStore;
    use crate::models::{ContentBlock, ConversationMessage, Role};

    fn msg(role: Role, text: &str) -> ConversationMessage {
        ConversationMessage {
            role,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    // ── Add, Get, Clear ────────────────────────────────────────────

    #[tokio::test]
    async fn get_history_returns_empty_for_unknown_conversation() {
        let store = InMemoryStore::new();
        let history = store.get_history("unknown").await.unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn add_and_get_single_message() {
        let store = InMemoryStore::new();
        store
            .add_message("c1", msg(Role::User, "hello"))
            .await
            .unwrap();

        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::User);
    }

    #[tokio::test]
    async fn messages_returned_in_insertion_order() {
        let store = InMemoryStore::new();
        store
            .add_message("c1", msg(Role::User, "first"))
            .await
            .unwrap();
        store
            .add_message("c1", msg(Role::Assistant, "second"))
            .await
            .unwrap();
        store
            .add_message("c1", msg(Role::User, "third"))
            .await
            .unwrap();

        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, Role::User);
        assert_eq!(history[1].role, Role::Assistant);
        assert_eq!(history[2].role, Role::User);

        // Verify content ordering
        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match &m.content[0] {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
    }


    #[tokio::test]
    async fn separate_conversations_are_isolated() {
        let store = InMemoryStore::new();
        store
            .add_message("c1", msg(Role::User, "conv1"))
            .await
            .unwrap();
        store
            .add_message("c2", msg(Role::User, "conv2"))
            .await
            .unwrap();

        let h1 = store.get_history("c1").await.unwrap();
        let h2 = store.get_history("c2").await.unwrap();
        assert_eq!(h1.len(), 1);
        assert_eq!(h2.len(), 1);
    }

    #[tokio::test]
    async fn clear_removes_all_messages() {
        let store = InMemoryStore::new();
        store
            .add_message("c1", msg(Role::User, "a"))
            .await
            .unwrap();
        store
            .add_message("c1", msg(Role::Assistant, "b"))
            .await
            .unwrap();

        store.clear("c1").await.unwrap();
        let history = store.get_history("c1").await.unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn clear_only_affects_target_conversation() {
        let store = InMemoryStore::new();
        store
            .add_message("c1", msg(Role::User, "keep"))
            .await
            .unwrap();
        store
            .add_message("c2", msg(Role::User, "remove"))
            .await
            .unwrap();

        store.clear("c2").await.unwrap();
        assert_eq!(store.get_history("c1").await.unwrap().len(), 1);
        assert!(store.get_history("c2").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn clear_nonexistent_conversation_is_ok() {
        let store = InMemoryStore::new();
        // Should not error
        store.clear("nonexistent").await.unwrap();
    }

    // ── Truncation with max_messages ───────────────────────────────

    #[tokio::test]
    async fn no_truncation_when_under_limit() {
        let store = InMemoryStore::with_max_messages(5);
        for i in 0..3 {
            store
                .add_message("c1", msg(Role::User, &format!("msg{i}")))
                .await
                .unwrap();
        }
        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 3);
    }

    #[tokio::test]
    async fn no_truncation_when_at_limit() {
        let store = InMemoryStore::with_max_messages(3);
        for i in 0..3 {
            store
                .add_message("c1", msg(Role::User, &format!("msg{i}")))
                .await
                .unwrap();
        }
        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 3);
    }

    #[tokio::test]
    async fn truncation_keeps_most_recent_messages() {
        let store = InMemoryStore::with_max_messages(3);
        for i in 0..6 {
            store
                .add_message("c1", msg(Role::User, &format!("msg{i}")))
                .await
                .unwrap();
        }
        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 3);

        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match &m.content[0] {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["msg3", "msg4", "msg5"]);
    }

    // ── System prompt preservation during truncation ───────────────

    #[tokio::test]
    async fn truncation_preserves_system_prompt() {
        let store = InMemoryStore::with_max_messages(3);
        store
            .add_message("c1", msg(Role::System, "system"))
            .await
            .unwrap();
        for i in 0..5 {
            store
                .add_message("c1", msg(Role::User, &format!("msg{i}")))
                .await
                .unwrap();
        }

        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 3);
        // First message must be the system prompt
        assert_eq!(history[0].role, Role::System);

        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match &m.content[0] {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        // system + 2 most recent user messages
        assert_eq!(texts, vec!["system", "msg3", "msg4"]);
    }

    #[tokio::test]
    async fn system_prompt_preserved_when_exactly_at_limit() {
        let store = InMemoryStore::with_max_messages(3);
        store
            .add_message("c1", msg(Role::System, "sys"))
            .await
            .unwrap();
        store
            .add_message("c1", msg(Role::User, "u1"))
            .await
            .unwrap();
        store
            .add_message("c1", msg(Role::Assistant, "a1"))
            .await
            .unwrap();

        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, Role::System);
    }

    #[tokio::test]
    async fn no_system_prompt_truncation_keeps_recent() {
        // Without a system prompt, truncation just keeps the N most recent
        let store = InMemoryStore::with_max_messages(2);
        store
            .add_message("c1", msg(Role::User, "old"))
            .await
            .unwrap();
        store
            .add_message("c1", msg(Role::Assistant, "older"))
            .await
            .unwrap();
        store
            .add_message("c1", msg(Role::User, "new"))
            .await
            .unwrap();

        let history = store.get_history("c1").await.unwrap();
        assert_eq!(history.len(), 2);

        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match &m.content[0] {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["older", "new"]);
    }
}
