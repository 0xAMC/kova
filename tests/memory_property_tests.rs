use proptest::prelude::*;

use kova::memory::MemoryStore;
use kova::memory::in_memory::InMemoryStore;
use kova::models::{ContentBlock, ConversationMessage, Role};

fn make_msg(role: Role, text: &str) -> ConversationMessage {
    ConversationMessage {
        role,
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
    }
}

fn arb_user_message() -> impl Strategy<Value = ConversationMessage> {
    "[a-zA-Z0-9 ]{1,50}".prop_map(|text| make_msg(Role::User, &text))
}

fn arb_non_system_messages() -> impl Strategy<Value = Vec<ConversationMessage>> {
    proptest::collection::vec(arb_user_message(), 1..20)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_memory_store_message_ordering(messages in arb_non_system_messages()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = InMemoryStore::new();
            let conv_id = "test_conv";

            for msg in &messages {
                store.add_message(conv_id, msg.clone()).await.unwrap();
            }

            let history = store.get_history(conv_id).await.unwrap();
            prop_assert_eq!(history.len(), messages.len());

            for (i, (got, expected)) in history.iter().zip(messages.iter()).enumerate() {
                prop_assert_eq!(
                    got, expected,
                    "message at index {} differs", i
                );
            }

            Ok(())
        })?;
    }

    #[test]
    fn prop_memory_store_clear(messages in arb_non_system_messages()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = InMemoryStore::new();
            let conv_id = "test_conv";

            for msg in &messages {
                store.add_message(conv_id, msg.clone()).await.unwrap();
            }

            store.clear(conv_id).await.unwrap();
            let history = store.get_history(conv_id).await.unwrap();
            prop_assert!(history.is_empty(), "history should be empty after clear");

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_context_window_truncation_no_system(
        max_messages in 2usize..=10,
        msg_count in 1usize..=30,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = InMemoryStore::with_max_messages(max_messages);
            let conv_id = "trunc";

            for i in 0..msg_count {
                store
                    .add_message(conv_id, make_msg(Role::User, &format!("msg_{}", i)))
                    .await
                    .unwrap();
            }

            let history = store.get_history(conv_id).await.unwrap();

            prop_assert!(
                history.len() <= max_messages,
                "history length {} exceeds max {}",
                history.len(),
                max_messages
            );

            if msg_count > max_messages {
                prop_assert_eq!(history.len(), max_messages);
                let expected_start = msg_count - max_messages;
                for (i, msg) in history.iter().enumerate() {
                    let expected_text = format!("msg_{}", expected_start + i);
                    let got_text = msg.content.iter().find_map(|b| {
                        if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
                    }).unwrap_or("");
                    prop_assert_eq!(got_text, expected_text.as_str());
                }
            }

            Ok(())
        })?;
    }

    #[test]
    fn prop_context_window_truncation_with_system(
        max_messages in 3usize..=10,
        extra_count in 1usize..=20,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = InMemoryStore::with_max_messages(max_messages);
            let conv_id = "trunc_sys";

            store
                .add_message(conv_id, make_msg(Role::System, "You are helpful."))
                .await
                .unwrap();

            let total_non_system = max_messages + extra_count;
            for i in 0..total_non_system {
                store
                    .add_message(conv_id, make_msg(Role::User, &format!("msg_{}", i)))
                    .await
                    .unwrap();
            }

            let history = store.get_history(conv_id).await.unwrap();

            prop_assert!(
                history.len() <= max_messages,
                "history length {} exceeds max {}",
                history.len(),
                max_messages
            );
            prop_assert_eq!(history.len(), max_messages);

            prop_assert_eq!(&history[0].role, &Role::System);
            let sys_text = history[0].content.iter().find_map(|b| {
                if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
            }).unwrap_or("");
            prop_assert_eq!(sys_text, "You are helpful.");

            let keep = max_messages - 1;
            let expected_start = total_non_system - keep;
            for (i, msg) in history[1..].iter().enumerate() {
                let expected_text = format!("msg_{}", expected_start + i);
                let got_text = msg.content.iter().find_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
                }).unwrap_or("");
                prop_assert_eq!(got_text, expected_text.as_str());
            }

            Ok(())
        })?;
    }
}

fn _assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<InMemoryStore>();
}
