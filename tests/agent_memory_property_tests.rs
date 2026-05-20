use proptest::prelude::*;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use tokio::sync::Mutex;

use kova::agent::AgentBuilder;
use kova::error::KovaError;
use kova::memory::in_memory::InMemoryStore;
use kova::memory::MemoryStore;
use kova::models::*;
use kova::provider::LlmProvider;

struct CapturingMock {
    responses: Vec<ModelResponse>,
    call_count: AtomicUsize,
    captured: Arc<Mutex<Vec<Vec<ConversationMessage>>>>,
}

impl CapturingMock {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
            captured: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn captured_requests(&self) -> Vec<Vec<ConversationMessage>> {
        self.captured.lock().await.clone()
    }
}

#[async_trait]
impl LlmProvider for CapturingMock {
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
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>,
        KovaError,
    > {
        Err(KovaError::Stream("not implemented".into()))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        Ok(vec![])
    }
}

fn make_text_response(text: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: None,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_agent_appends_user_and_assistant_to_memory(
        user_text in "[a-zA-Z0-9 ]{1,50}",
        assistant_text in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let memory: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
            let provider = Arc::new(CapturingMock::new(vec![
                make_text_response(&assistant_text),
            ]));

            let agent = AgentBuilder::new()
                .provider(provider as Arc<dyn LlmProvider>)
                .memory(memory.clone())
                .build()
                .unwrap();

            let result = agent.chat("conv1", &user_text).await.unwrap();
            prop_assert_eq!(&result, &assistant_text);

            let history = memory.get_history("conv1").await.unwrap();
            prop_assert_eq!(history.len(), 2, "expected 2 messages in memory, got {}", history.len());

            prop_assert_eq!(&history[0].role, &Role::User);
            let user_content = history[0].content.iter().find_map(|b| {
                if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
            }).unwrap_or("");
            prop_assert_eq!(user_content, user_text.as_str());

            prop_assert_eq!(&history[1].role, &Role::Assistant);
            let asst_content = history[1].content.iter().find_map(|b| {
                if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
            }).unwrap_or("");
            prop_assert_eq!(asst_content, assistant_text.as_str());

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_agent_includes_full_history_in_requests(
        turn_count in 1usize..=5,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let memory: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
            let provider = Arc::new(CapturingMock::new(vec![
                make_text_response("reply"),
            ]));

            let agent = AgentBuilder::new()
                .provider(provider.clone() as Arc<dyn LlmProvider>)
                .memory(memory.clone())
                .system_prompt("You are helpful.")
                .build()
                .unwrap();

            for i in 0..turn_count {
                let _ = agent.chat("conv", &format!("msg_{}", i)).await.unwrap();
            }

            let requests = provider.captured_requests().await;
            prop_assert_eq!(
                requests.len(), turn_count,
                "expected {} LLM calls, got {}", turn_count, requests.len()
            );

            let last_req = &requests[turn_count - 1];

            prop_assert_eq!(&last_req[0].role, &Role::System);

            let expected_count = 1 + (turn_count - 1) * 2 + 1;
            prop_assert_eq!(
                last_req.len(), expected_count,
                "expected {} messages in request, got {}", expected_count, last_req.len()
            );

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_system_prompt_prepended(
        system_prompt in "[a-zA-Z0-9 ]{1,50}",
        user_text in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let provider = Arc::new(CapturingMock::new(vec![
                make_text_response("ok"),
            ]));

            let agent = AgentBuilder::new()
                .provider(provider.clone() as Arc<dyn LlmProvider>)
                .system_prompt(&system_prompt)
                .build()
                .unwrap();

            let _ = agent.chat("conv", &user_text).await.unwrap();

            let requests = provider.captured_requests().await;
            prop_assert!(!requests.is_empty());

            for req in &requests {
                prop_assert!(!req.is_empty(), "request should not be empty");
                prop_assert_eq!(&req[0].role, &Role::System, "first message should be system");
                let sys_text = req[0].content.iter().find_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
                }).unwrap_or("");
                prop_assert_eq!(sys_text, system_prompt.as_str());
            }

            Ok(())
        })?;
    }
}
