use proptest::prelude::*;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;

use kova_sdk::agent::AgentBuilder;
use kova_sdk::error::KovaError;
use kova_sdk::models::*;
use kova_sdk::orchestrator::{Orchestrator, OrchestratorOutput, OrchestratorPattern};
use kova_sdk::provider::{LlmProvider, RetryConfig};

struct SuffixProvider {
    suffix: String,
}

#[async_trait]
impl LlmProvider for SuffixProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let input = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.content.first())
            .map(|c| match c {
                ContentBlock::Text { text } => text.clone(),
                _ => String::new(),
            })
            .unwrap_or_default();

        Ok(ModelResponse {
            content: vec![ContentBlock::Text {
                text: format!("{}{}", input, self.suffix),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
            thinking: None,
        })
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

struct FailingProvider {
    message: String,
}

#[async_trait]
impl LlmProvider for FailingProvider {
    async fn chat_completion(
        &self,
        _messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        Err(KovaError::Provider {
            message: self.message.clone(),
            status_code: Some(500),
        })
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

struct CapturingEchoProvider {
    captured: Arc<tokio::sync::Mutex<Vec<String>>>,
    suffix: String,
}

#[async_trait]
impl LlmProvider for CapturingEchoProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let input = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.content.first())
            .map(|c| match c {
                ContentBlock::Text { text } => text.clone(),
                _ => String::new(),
            })
            .unwrap_or_default();

        self.captured.lock().await.push(input.clone());

        Ok(ModelResponse {
            content: vec![ContentBlock::Text {
                text: format!("{}{}", input, self.suffix),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
            thinking: None,
        })
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

fn build_suffix_agent(suffix: &str) -> Arc<kova_sdk::agent::Agent> {
    let provider = Arc::new(SuffixProvider {
        suffix: suffix.to_string(),
    });
    Arc::new(
        AgentBuilder::new()
            .provider(provider)
            .build()
            .expect("build suffix agent"),
    )
}

fn build_failing_agent(msg: &str) -> Arc<kova_sdk::agent::Agent> {
    let provider = Arc::new(FailingProvider {
        message: msg.to_string(),
    });
    Arc::new(
        AgentBuilder::new()
            .provider(provider)
            .retry_config(RetryConfig::disabled())
            .build()
            .expect("build failing agent"),
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_sequential_output_chaining(
        n in 1usize..=5,
        input in "[a-zA-Z0-9 ]{1,20}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut agents_map = HashMap::new();
            let mut names = Vec::new();
            for i in 0..n {
                let name = format!("agent_{}", i);
                let suffix = format!("_{}", i);
                agents_map.insert(name.clone(), build_suffix_agent(&suffix));
                names.push(name);
            }

            let orch = Orchestrator::new(agents_map, Duration::from_secs(10));
            let result = orch
                .execute(OrchestratorPattern::Sequential(names.clone()), &input)
                .await;

            let output = match result {
                Ok(OrchestratorOutput::Single(s)) => s,
                Ok(OrchestratorOutput::Parallel(_)) => {
                    return Err(proptest::test_runner::TestCaseError::fail(
                        "expected Single output from sequential",
                    ));
                }
                Err(e) => {
                    return Err(proptest::test_runner::TestCaseError::fail(
                        format!("sequential execution failed: {}", e),
                    ));
                }
            };

            let mut expected = input.clone();
            for i in 0..n {
                expected = format!("{}_{}", expected, i);
            }
            prop_assert_eq!(output, expected);
            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_sequential_failure_stops_pipeline(
        total in 2usize..=5,
        fail_idx in 0usize..5,
        input in "[a-zA-Z0-9 ]{1,20}",
    ) {
        let fail_idx = fail_idx % total;

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut agents_map = HashMap::new();
            let mut names = Vec::new();

            let post_fail_captures: Vec<Arc<tokio::sync::Mutex<Vec<String>>>> = (0..total)
                .map(|_| Arc::new(tokio::sync::Mutex::new(Vec::new())))
                .collect();

            for (i, captured_slot) in post_fail_captures.iter().enumerate() {
                let name = format!("agent_{}", i);
                if i == fail_idx {
                    agents_map.insert(name.clone(), build_failing_agent("intentional failure"));
                } else {
                    let captured = captured_slot.clone();
                    let provider = Arc::new(CapturingEchoProvider {
                        captured,
                        suffix: format!("_{}", i),
                    });
                    let agent = Arc::new(
                        AgentBuilder::new()
                            .provider(provider)
                            .build()
                            .expect("build capturing agent"),
                    );
                    agents_map.insert(name.clone(), agent);
                }
                names.push(name);
            }

            let orch = Orchestrator::new(agents_map, Duration::from_secs(10));
            let result = orch
                .execute(OrchestratorPattern::Sequential(names), &input)
                .await;

            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => {
                    return Err(proptest::test_runner::TestCaseError::fail(
                        "expected error from failing agent, got Ok",
                    ));
                }
            };
            prop_assert!(
                err_msg.contains("intentional failure"),
                "error should contain the failing agent's message, got: {}",
                err_msg
            );

            for (i, captured_slot) in post_fail_captures.iter().enumerate().skip(fail_idx + 1) {
                let captured = captured_slot.lock().await;
                prop_assert!(
                    captured.is_empty(),
                    "agent_{} should not have been invoked after agent_{} failed",
                    i,
                    fail_idx
                );
            }

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_parallel_same_input(
        n in 1usize..=5,
        input in "[a-zA-Z0-9 ]{1,20}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut agents_map = HashMap::new();
            let mut names = Vec::new();
            let captures: Vec<Arc<tokio::sync::Mutex<Vec<String>>>> = (0..n)
                .map(|_| Arc::new(tokio::sync::Mutex::new(Vec::new())))
                .collect();

            for (i, captured_slot) in captures.iter().enumerate() {
                let name = format!("agent_{}", i);
                let captured = captured_slot.clone();
                let provider = Arc::new(CapturingEchoProvider {
                    captured,
                    suffix: format!("_{}", i),
                });
                let agent = Arc::new(
                    AgentBuilder::new()
                        .provider(provider)
                        .build()
                        .expect("build capturing agent"),
                );
                agents_map.insert(name.clone(), agent);
                names.push(name);
            }

            let orch = Orchestrator::new(agents_map, Duration::from_secs(10));
            let input_ref = input.as_str();
            let result = orch
                .execute(OrchestratorPattern::Parallel(names), input_ref)
                .await;

            if let Err(e) = &result {
                return Err(proptest::test_runner::TestCaseError::fail(
                    format!("parallel execution should succeed, got: {}", e),
                ));
            }

            for (i, captured_slot) in captures.iter().enumerate() {
                let captured = captured_slot.lock().await;
                prop_assert_eq!(
                    captured.len(),
                    1,
                    "agent_{} should have been called exactly once",
                    i
                );
                prop_assert_eq!(
                    &captured[0],
                    input_ref,
                    "agent_{} should have received the original input",
                    i
                );
            }

            Ok(())
        })?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_parallel_partial_failure(
        total in 2usize..=5,
        fail_mask in 0u32..32,
        input in "[a-zA-Z0-9 ]{1,20}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut agents_map = HashMap::new();
            let mut names = Vec::new();
            let mut expected_fail_count = 0usize;
            let mut expected_success_names = Vec::new();

            for i in 0..total {
                let name = format!("agent_{}", i);
                let should_fail = (fail_mask >> (i as u32)) & 1 == 1;

                if should_fail {
                    agents_map.insert(name.clone(), build_failing_agent(&format!("fail_{}", i)));
                    expected_fail_count += 1;
                } else {
                    agents_map.insert(name.clone(), build_suffix_agent(&format!("_{}", i)));
                    expected_success_names.push(name.clone());
                }
                names.push(name);
            }

            let orch = Orchestrator::new(agents_map, Duration::from_secs(10));
            let result = orch
                .execute(OrchestratorPattern::Parallel(names), &input)
                .await;

            let parallel_result = match result {
                Ok(OrchestratorOutput::Parallel(pr)) => pr,
                Ok(OrchestratorOutput::Single(_)) => {
                    return Err(proptest::test_runner::TestCaseError::fail(
                        "expected Parallel output",
                    ));
                }
                Err(e) => {
                    return Err(proptest::test_runner::TestCaseError::fail(
                        format!("parallel should return Ok with ParallelResult, got: {}", e),
                    ));
                }
            };

            prop_assert_eq!(
                parallel_result.failures.len(),
                expected_fail_count,
                "expected {} failures, got {}",
                expected_fail_count,
                parallel_result.failures.len()
            );

            prop_assert_eq!(
                parallel_result.successes.len(),
                expected_success_names.len(),
                "expected {} successes, got {}",
                expected_success_names.len(),
                parallel_result.successes.len()
            );

            let success_agent_names: Vec<String> = parallel_result
                .successes
                .iter()
                .map(|(name, _)| name.clone())
                .collect();
            for expected_name in &expected_success_names {
                prop_assert!(
                    success_agent_names.contains(expected_name),
                    "expected {} in successes",
                    expected_name
                );
            }

            Ok(())
        })?;
    }
}
