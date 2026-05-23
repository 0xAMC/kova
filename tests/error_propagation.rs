//! Integration tests for error propagation.
//!
//! Tests that provider failures, tool failures, and timeout conditions
//! produce the correct error variants.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use serde_json::json;

use kova::agent::AgentBuilder;
use kova::error::KovaError;
use kova::models::*;
use kova::provider::LlmProvider;
use kova::provider::openai::{OpenAiCompatibleProvider, OpenAiProviderConfig};
use kova::tool::Tool;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Failing mock provider ──────────────────────────────────────────

struct FailingProvider {
    error: KovaError,
}

impl FailingProvider {
    fn provider_error() -> Self {
        Self {
            error: KovaError::Provider {
                message: "service unavailable".to_string(),
                status_code: Some(503),
            },
        }
    }

    fn connection_error() -> Self {
        Self {
            error: KovaError::Connection("connection refused".to_string()),
        }
    }
}

#[async_trait]
impl LlmProvider for FailingProvider {
    async fn chat_completion(
        &self,
        _messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        // Recreate the error each time since KovaError isn't Clone.
        match &self.error {
            KovaError::Provider {
                message,
                status_code,
            } => Err(KovaError::Provider {
                message: message.clone(),
                status_code: *status_code,
            }),
            KovaError::Connection(msg) => Err(KovaError::Connection(msg.clone())),
            _ => Err(KovaError::Provider {
                message: "unknown".to_string(),
                status_code: None,
            }),
        }
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

// ── Mock provider that returns tool calls then text ────────────────

struct ToolCallThenTextProvider;

#[async_trait]
impl LlmProvider for ToolCallThenTextProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        // If we already have tool results in messages, return text.
        let has_tool_results = messages.iter().any(|m| m.role == Role::Tool);
        if has_tool_results {
            Ok(ModelResponse {
                content: vec![ContentBlock::Text {
                    text: "done".to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        } else {
            // First call: request a tool call.
            Ok(ModelResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "tc_1".to_string(),
                    name: "failing_tool".to_string(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: None,
            })
        }
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

// ── Failing tool ───────────────────────────────────────────────────

struct AlwaysFailingTool;

#[async_trait]
impl Tool for AlwaysFailingTool {
    fn name(&self) -> &str {
        "failing_tool"
    }
    fn description(&self) -> &str {
        "A tool that always fails"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult, KovaError> {
        Err(KovaError::ToolExecution {
            tool_name: "failing_tool".to_string(),
            message: "intentional failure".to_string(),
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────

/// Provider failure returns the correct error variant.
#[tokio::test]
async fn test_provider_failure_returns_provider_error() {
    let provider = Arc::new(FailingProvider::provider_error());
    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .build()
        .unwrap();

    let result = agent.chat("conv", "hello").await;
    match result {
        Err(KovaError::Provider {
            status_code: Some(503),
            ..
        }) => {} // expected
        other => panic!("Expected Provider error with 503, got: {:?}", other),
    }
}

/// Connection failure returns the correct error variant.
#[tokio::test]
async fn test_connection_failure_returns_connection_error() {
    let provider = Arc::new(FailingProvider::connection_error());
    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .build()
        .unwrap();

    let result = agent.chat("conv", "hello").await;
    match result {
        Err(KovaError::Connection(msg)) => {
            assert!(msg.contains("connection refused"));
        }
        other => panic!("Expected Connection error, got: {:?}", other),
    }
}

/// Tool execution failure is forwarded to the LLM as a tool result
/// (the agent doesn't crash — it sends the error back to the LLM).
#[tokio::test]
async fn test_tool_failure_forwarded_to_llm_as_tool_result() {
    let provider = Arc::new(ToolCallThenTextProvider);
    let failing_tool: Arc<dyn Tool> = Arc::new(AlwaysFailingTool);

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .tool(failing_tool)
        .max_iterations(5)
        .build()
        .unwrap();

    // The agent should NOT return an error — it should forward the tool
    // error to the LLM and get a text response back.
    let result = agent.chat("conv", "use the tool").await;
    assert!(
        result.is_ok(),
        "Agent should forward tool errors to LLM, not propagate them. Got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), "done");
}

/// Missing provider in AgentBuilder returns Build error.
#[tokio::test]
async fn test_missing_provider_returns_build_error() {
    let result = AgentBuilder::new().build();
    match result {
        Err(KovaError::Build(msg)) => {
            assert!(msg.contains("LlmProvider"));
        }
        Err(other) => panic!("Expected Build error, got: {:?}", other),
        Ok(_) => panic!("Expected Build error, got Ok"),
    }
}

/// HTTP timeout from a real provider returns the correct error variant.
#[tokio::test]
async fn test_timeout_returns_timeout_or_connection_error() {
    let server = MockServer::start().await;

    // Mock that delays longer than the provider timeout.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0u64,
                    "model": "m",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop"
                    }]
                }))
                .set_delay(Duration::from_secs(5)),
        )
        .mount(&server)
        .await;

    let config = OpenAiProviderConfig::new(server.uri(), "test-model")
        .with_timeout(Duration::from_millis(100));
    let provider = Arc::new(OpenAiCompatibleProvider::new(config).unwrap());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .build()
        .unwrap();

    let result = agent.chat("conv", "hello").await;
    match result {
        Err(KovaError::Timeout(_)) | Err(KovaError::Connection(_)) | Err(KovaError::Http(_)) => {
            // Any of these is acceptable for a timeout scenario.
        }
        other => panic!(
            "Expected Timeout, Connection, or Http error, got: {:?}",
            other
        ),
    }
}

/// HTTP error from a real provider propagates through the agent.
#[tokio::test]
async fn test_http_error_propagates_through_agent() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
        .mount(&server)
        .await;

    let config = OpenAiProviderConfig::new(server.uri(), "test-model");
    let provider = Arc::new(OpenAiCompatibleProvider::new(config).unwrap());

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .build()
        .unwrap();

    let result = agent.chat("conv", "hello").await;
    match result {
        Err(KovaError::Provider {
            status_code: Some(500),
            message,
        }) => {
            assert_eq!(message, "internal server error");
        }
        other => panic!("Expected Provider error with 500, got: {:?}", other),
    }
}
