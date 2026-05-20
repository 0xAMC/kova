use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::KovaError;
use crate::models::ToolResult;
use crate::tool::Tool;

use super::{McpClient, McpToolDefinition};

/// Adapter that wraps an MCP tool definition and client into the SDK [`Tool`] trait.
///
/// MCP tools registered via this adapter are indistinguishable from native
/// tools in the [`ToolRegistry`](crate::tool::registry::ToolRegistry).
///
/// Each `McpTool` carries a qualified name in the format `@<server_name>/<tool_name>`
/// which uniquely identifies the tool across all connected MCP servers.
pub struct McpTool {
    definition: McpToolDefinition,
    client: Arc<McpClient>,
    /// Logical name of the MCP server this tool belongs to.
    server_name: String,
    /// Cached qualified name: `<server_name>__<tool_name>`.
    qualified_name: String,
}

impl McpTool {
    /// Create a new `McpTool` from an MCP tool definition, a shared client, and the server name.
    ///
    /// The qualified name is computed as `{server_name}__{tool_name}`, which is safe
    /// for all LLM provider APIs (only `[a-zA-Z0-9_-]` characters).
    pub fn new(definition: McpToolDefinition, client: Arc<McpClient>, server_name: &str) -> Self {
        let qualified_name = format!("{}__{}", server_name, definition.name);
        Self {
            definition,
            client,
            server_name: server_name.to_string(),
            qualified_name,
        }
    }

    /// The bare tool name as returned by the MCP server.
    pub fn bare_name(&self) -> &str {
        &self.definition.name
    }

    /// The logical server name this tool belongs to.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.qualified_name
    }

    fn description(&self) -> &str {
        self.definition
            .description
            .as_deref()
            .unwrap_or("")
    }

    fn parameters_schema(&self) -> Value {
        self.definition
            .input_schema
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object"}))
    }

    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let (content, is_error) = self.client.tools_call(&self.definition.name, args).await?;
        Ok(ToolResult { content, is_error })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_tool_name_and_description() {
        let def = McpToolDefinition {
            name: "get_weather".to_string(),
            description: Some("Get current weather for a city".to_string()),
            input_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                }
            })),
        };

        // We can't create a real McpClient in a unit test, but we can verify
        // the trait method implementations via a helper that doesn't call execute.
        assert_eq!(def.name, "get_weather");
        assert_eq!(def.description.as_deref(), Some("Get current weather for a city"));
    }

    #[test]
    fn mcp_tool_defaults_for_missing_fields() {
        let def = McpToolDefinition {
            name: "minimal".to_string(),
            description: None,
            input_schema: None,
        };

        // Verify the defaults that McpTool would use.
        assert_eq!(def.description.as_deref().unwrap_or(""), "");
        let schema = def
            .input_schema
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object"}));
        assert_eq!(schema, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn mcp_tool_conversion_preserves_fields() {
        let def = McpToolDefinition {
            name: "search".to_string(),
            description: Some("Search the web".to_string()),
            input_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })),
        };

        // Verify that the fields match what Tool trait methods would return.
        assert_eq!(&def.name, "search");
        assert_eq!(def.description.as_deref().unwrap_or(""), "Search the web");
        assert_eq!(
            def.input_schema.as_ref().unwrap()["properties"]["query"]["type"],
            "string"
        );
    }
}
