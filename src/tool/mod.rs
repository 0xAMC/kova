pub mod approval;
pub mod registry;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::KovaError;
use crate::models::ToolResult;

/// A callable tool that the LLM can invoke to perform actions.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique name for this tool.
    fn name(&self) -> &str;

    /// Human-readable description for the LLM.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given JSON arguments.
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError>;
}

/// Optional lifecycle hook for observing tool execution.
///
/// Implement this to display progress, log, or apply custom
/// side-effects around each tool call without modifying tools.
#[async_trait]
pub trait ToolLifecycleHook: Send + Sync {
    /// Called immediately before a tool is executed.
    async fn on_tool_start(&self, tool_name: &str, args: &Value);

    /// Called immediately after a tool returns its result.
    async fn on_tool_end(&self, tool_name: &str, result: &str, is_error: bool);
}
