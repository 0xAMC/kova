use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use kova_sdk::error::KovaError;
use kova_sdk::models::ToolResult;
use kova_sdk::tool::Tool;

/// A configurable mock tool for integration tests.
///
/// Can be set up to:
/// - Return a fixed successful result
/// - Fail on every invocation
/// - Introduce an artificial delay before returning
/// - Track how many times it was called
pub struct MockTool {
    tool_name: String,
    tool_description: String,
    schema: Value,
    /// When `Some`, `execute` returns this error message.
    fail_with: Option<String>,
    /// The content returned on success.
    result_content: String,
    /// Optional delay before returning from `execute`.
    delay: Option<Duration>,
    /// Number of times `execute` has been called.
    call_count: AtomicUsize,
}

impl MockTool {
    /// Create a mock tool that succeeds with the given content.
    pub fn new(name: &str, result_content: &str) -> Self {
        Self {
            tool_name: name.to_string(),
            tool_description: format!("Mock tool: {name}"),
            schema: json!({"type": "object"}),
            fail_with: None,
            result_content: result_content.to_string(),
            delay: None,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create a mock tool that always fails with the given message.
    pub fn failing(name: &str, error_message: &str) -> Self {
        Self {
            tool_name: name.to_string(),
            tool_description: format!("Failing mock tool: {name}"),
            schema: json!({"type": "object"}),
            fail_with: Some(error_message.to_string()),
            result_content: String::new(),
            delay: None,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create a mock tool that succeeds after an artificial delay.
    pub fn with_delay(name: &str, result_content: &str, delay: Duration) -> Self {
        Self {
            tool_name: name.to_string(),
            tool_description: format!("Delayed mock tool: {name}"),
            schema: json!({"type": "object"}),
            fail_with: None,
            result_content: result_content.to_string(),
            delay: Some(delay),
            call_count: AtomicUsize::new(0),
        }
    }

    /// How many times `execute` has been called.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Tool for MockTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, _args: Value) -> Result<ToolResult, KovaError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        if let Some(d) = self.delay {
            tokio::time::sleep(d).await;
        }

        if let Some(ref msg) = self.fail_with {
            return Err(KovaError::ToolExecution {
                tool_name: self.tool_name.clone(),
                message: msg.clone(),
            });
        }

        Ok(ToolResult {
            content: self.result_content.clone(),
            is_error: false,
        })
    }
}
