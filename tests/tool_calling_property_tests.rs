use proptest::prelude::*;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::Stream;
use serde_json::json;
use tokio::sync::Mutex;

use kova::agent::AgentBuilder;
use kova::error::KovaError;
use kova::models::*;
use kova::provider::LlmProvider;
use kova::tool::Tool;

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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError> {
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

fn make_tool_call_response(tool_calls: Vec<(String, String, serde_json::Value)>) -> ModelResponse {
    let content = tool_calls
        .into_iter()
        .map(|(id, name, input)| ContentBlock::ToolUse { id, name, input })
        .collect();
    ModelResponse {
        content,
        stop_reason: StopReason::ToolUse,
        usage: None,
    }
}

fn make_tool_call(id: &str, name: &str, args: &str) -> (String, String, serde_json::Value) {
    let input: serde_json::Value = serde_json::from_str(args).unwrap_or(json!({}));
    (id.to_string(), name.to_string(), input)
}

struct ConfigurableTool {
    tool_name: String,
    should_fail: bool,
}

#[async_trait]
impl Tool for ConfigurableTool {
    fn name(&self) -> &str {
        &self.tool_name
    }
    fn description(&self) -> &str {
        "a configurable mock tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult, KovaError> {
        if self.should_fail {
            Err(KovaError::ToolExecution {
                tool_name: self.tool_name.clone(),
                message: "intentional failure".into(),
            })
        } else {
            Ok(ToolResult {
                content: format!("result from {}", self.tool_name),
                is_error: false,
            })
        }
    }
}

fn arb_tool_call_ids(count: usize) -> impl Strategy<Value = Vec<String>> {
    proptest::collection::vec("[a-z0-9]{4,8}", count..=count).prop_map(|mut ids| {
        for (i, id) in ids.iter_mut().enumerate() {
            id.push_str(&format!("_{}", i));
        }
        ids
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_tool_errors_forwarded_to_llm(
        unknown_name in "[a-z]{3,10}_unknown",
        call_id_unknown in "[a-z0-9]{4,8}",
        failing_name in "[a-z]{3,10}_fail",
        call_id_fail in "[a-z0-9]{4,8}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tc_unknown = make_tool_call(&call_id_unknown, &unknown_name, "{}");
            let tc_fail = make_tool_call(&call_id_fail, &failing_name, "{}");

            let provider = Arc::new(CapturingMock::new(vec![
                make_tool_call_response(vec![tc_unknown, tc_fail]),
                make_text_response("done"),
            ]));

            let failing_tool: Arc<dyn Tool> = Arc::new(ConfigurableTool {
                tool_name: failing_name.clone(),
                should_fail: true,
            });

            let agent = AgentBuilder::new()
                .provider(provider.clone() as Arc<dyn LlmProvider>)
                .tool(failing_tool)
                .max_iterations(5)
                .build()
                .unwrap();

            let _result = agent.chat("conv", "hi").await.unwrap();

            let requests = provider.captured_requests().await;
            prop_assert!(
                requests.len() >= 2,
                "expected at least 2 LLM calls, got {}",
                requests.len()
            );

            let second_msgs = &requests[1];
            let tool_msgs: Vec<&ConversationMessage> = second_msgs
                .iter()
                .filter(|m| m.role == Role::Tool)
                .collect();

            prop_assert_eq!(
                tool_msgs.len(),
                2,
                "expected 2 tool-role messages, got {}",
                tool_msgs.len()
            );

            let unknown_msg = tool_msgs
                .iter()
                .find(|m| {
                    m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == &call_id_unknown))
                })
                .expect("missing tool-role message for unknown tool");
            let content = unknown_msg.content.iter().find_map(|b| {
                if let ContentBlock::ToolResult { content, .. } = b { Some(content.as_str()) } else { None }
            }).unwrap_or("");
            prop_assert!(
                content.contains(&unknown_name) || content.to_lowercase().contains("not found"),
                "unknown tool error should mention tool name or 'not found', got: {}",
                content
            );

            let fail_msg = tool_msgs
                .iter()
                .find(|m| {
                    m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == &call_id_fail))
                })
                .expect("missing tool-role message for failing tool");
            let fail_content = fail_msg.content.iter().find_map(|b| {
                if let ContentBlock::ToolResult { content, .. } = b { Some(content.as_str()) } else { None }
            }).unwrap_or("");
            prop_assert!(
                !fail_content.is_empty(),
                "failing tool error message should be non-empty"
            );

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_tool_results_appended_as_messages(n in 1usize..=5, ids in arb_tool_call_ids(5)) {
        let ids: Vec<String> = ids.into_iter().take(n).collect();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tool_calls: Vec<(String, String, serde_json::Value)> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let name = format!("tool_{}", i);
                    make_tool_call(id, &name, "{}")
                })
                .collect();

            let provider = Arc::new(CapturingMock::new(vec![
                make_tool_call_response(tool_calls.clone()),
                make_text_response("done"),
            ]));

            let mut builder = AgentBuilder::new()
                .provider(provider.clone() as Arc<dyn LlmProvider>)
                .max_iterations(5);

            for i in 0..n {
                let tool: Arc<dyn Tool> = Arc::new(ConfigurableTool {
                    tool_name: format!("tool_{}", i),
                    should_fail: false,
                });
                builder = builder.tool(tool);
            }

            let agent = builder.build().unwrap();
            let _result = agent.chat("conv", "hi").await.unwrap();

            let requests = provider.captured_requests().await;
            prop_assert!(
                requests.len() >= 2,
                "expected at least 2 LLM calls, got {}",
                requests.len()
            );

            let second_msgs = &requests[1];
            let tool_msgs: Vec<&ConversationMessage> = second_msgs
                .iter()
                .filter(|m| m.role == Role::Tool)
                .collect();

            prop_assert_eq!(
                tool_msgs.len(),
                n,
                "expected {} tool-role messages, got {}",
                n,
                tool_msgs.len()
            );

            for id in &ids {
                let count = tool_msgs
                    .iter()
                    .filter(|m| {
                        m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id))
                    })
                    .count();
                prop_assert_eq!(
                    count,
                    1,
                    "tool_use_id '{}' should appear exactly once, found {}",
                    id,
                    count
                );
            }

            for msg in &tool_msgs {
                for block in &msg.content {
                    if let ContentBlock::ToolResult { content, .. } = block {
                        prop_assert!(
                            !content.is_empty(),
                            "tool result content should be non-empty"
                        );
                    }
                }
            }

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_agent_text_return_on_end_turn_or_max_tokens(
        text in "[a-zA-Z0-9 .,!?]{1,100}",
        use_end_turn in any::<bool>(),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let stop_reason = if use_end_turn {
                StopReason::EndTurn
            } else {
                StopReason::MaxTokens
            };

            let response = ModelResponse {
                content: vec![ContentBlock::Text { text: text.clone() }],
                stop_reason,
                usage: None,
            };

            let provider = Arc::new(CapturingMock::new(vec![response]));

            let agent = AgentBuilder::new()
                .provider(provider.clone() as Arc<dyn LlmProvider>)
                .build()
                .unwrap();

            let result = agent.chat("conv", "hi").await.unwrap();
            prop_assert_eq!(&result, &text);

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_max_iterations_terminates_tool_loop(max_iter in 1usize..=8) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let always_tool_call = make_tool_call_response(vec![
                make_tool_call("tc_1", "echo", "{}"),
            ]);

            let provider = Arc::new(CapturingMock::new(vec![always_tool_call]));

            let echo_tool: Arc<dyn Tool> = Arc::new(ConfigurableTool {
                tool_name: "echo".to_string(),
                should_fail: false,
            });

            let agent = AgentBuilder::new()
                .provider(provider.clone() as Arc<dyn LlmProvider>)
                .tool(echo_tool)
                .max_iterations(max_iter)
                .build()
                .unwrap();

            let result = agent.chat("conv", "loop forever").await;

            match &result {
                Err(KovaError::MaxIterations(n)) => {
                    prop_assert_eq!(
                        *n,
                        max_iter,
                        "MaxIterations should carry the configured limit"
                    );
                }
                Err(other) => {
                    prop_assert!(false, "expected MaxIterations, got: {:?}", other);
                }
                Ok(text) => {
                    prop_assert!(
                        false,
                        "expected MaxIterations error, got Ok(\"{}\")",
                        text
                    );
                }
            }

            let requests = provider.captured_requests().await;
            prop_assert_eq!(
                requests.len(),
                max_iter + 1,
                "expected {} LLM calls (1 initial + {} iterations), got {}",
                max_iter + 1,
                max_iter,
                requests.len()
            );

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_parallel_tool_partial_failure(
        n in 1usize..=5,
        ids in arb_tool_call_ids(5),
        k_ratio in 0.0f64..=1.0,
    ) {
        let ids: Vec<String> = ids.into_iter().take(n).collect();
        let k = ((k_ratio * n as f64).floor() as usize).min(n);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tool_calls: Vec<(String, String, serde_json::Value)> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let name = format!("ptool_{}", i);
                    make_tool_call(id, &name, "{}")
                })
                .collect();

            let provider = Arc::new(CapturingMock::new(vec![
                make_tool_call_response(tool_calls.clone()),
                make_text_response("done"),
            ]));

            let mut builder = AgentBuilder::new()
                .provider(provider.clone() as Arc<dyn LlmProvider>)
                .max_iterations(5);

            for i in 0..n {
                let tool: Arc<dyn Tool> = Arc::new(ConfigurableTool {
                    tool_name: format!("ptool_{}", i),
                    should_fail: i < k,
                });
                builder = builder.tool(tool);
            }

            let agent = builder.build().unwrap();
            let result = agent.chat("conv", "run tools").await.unwrap();
            prop_assert_eq!(&result, "done");

            let requests = provider.captured_requests().await;
            prop_assert!(
                requests.len() >= 2,
                "expected at least 2 LLM calls, got {}",
                requests.len()
            );

            let second_msgs = &requests[1];
            let tool_msgs: Vec<&ConversationMessage> = second_msgs
                .iter()
                .filter(|m| m.role == Role::Tool)
                .collect();

            prop_assert_eq!(
                tool_msgs.len(),
                n,
                "expected {} tool-role messages, got {}",
                n,
                tool_msgs.len()
            );

            for id in &ids {
                let count = tool_msgs
                    .iter()
                    .filter(|m| {
                        m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id))
                    })
                    .count();
                prop_assert_eq!(
                    count,
                    1,
                    "tool_use_id '{}' should appear exactly once, found {}",
                    id,
                    count
                );
            }

            for (i, id) in ids.iter().enumerate() {
                let msg = tool_msgs
                    .iter()
                    .find(|m| {
                        m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id))
                    })
                    .expect("missing tool-role message");

                let (content, is_error) = msg.content.iter().find_map(|b| {
                    if let ContentBlock::ToolResult { content, is_error, tool_use_id } = b
                        && tool_use_id == id
                    {
                        return Some((content.clone(), *is_error));
                    }
                    None
                }).expect("missing ToolResult block");

                prop_assert!(
                    !content.is_empty(),
                    "tool result content should be non-empty for tool_use_id '{}'",
                    id
                );

                if i < k {
                    prop_assert!(
                        is_error,
                        "tool at index {} should have is_error=true (failed), got false",
                        i
                    );
                } else {
                    prop_assert!(
                        !is_error,
                        "tool at index {} should have is_error=false (success), got true",
                        i
                    );
                }
            }

            Ok(())
        })?;
    }
}
