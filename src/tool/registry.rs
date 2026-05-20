use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::models::ToolDefinition;

use super::Tool;

/// Thread-safe registry for storing and resolving tools by name.
pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>,
}

impl ToolRegistry {
    /// Create a new empty `ToolRegistry`.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a `ToolRegistry` pre-populated with the given tools.
    ///
    /// This is a synchronous constructor that avoids the need for async
    /// `register` calls — useful in `AgentBuilder::build()`.
    pub fn from_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map = HashMap::new();
        for tool in tools {
            map.insert(tool.name().to_string(), tool);
        }
        Self {
            tools: Arc::new(RwLock::new(map)),
        }
    }

    /// Register a tool. Overwrites any existing tool with the same name.
    pub async fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        let mut tools = self.tools.write().await;
        tools.insert(name, tool);
    }

    /// Look up a tool by name.
    pub async fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().await;
        tools.get(name).cloned()
    }

    /// List all registered tool names.
    pub async fn list(&self) -> Vec<String> {
        let tools = self.tools.read().await;
        tools.keys().cloned().collect()
    }

    /// Return `ToolDefinition` entries for all registered tools.
    pub async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        tools
            .values()
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: Arc::clone(&self.tools),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KovaError;
    use crate::models::ToolResult;
    use async_trait::async_trait;
    use serde_json::json;

    /// A simple test tool for unit tests.
    struct DummyTool {
        tool_name: String,
        tool_description: String,
        schema: serde_json::Value,
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            &self.tool_description
        }
        fn parameters_schema(&self) -> serde_json::Value {
            self.schema.clone()
        }
        async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult, KovaError> {
            Ok(ToolResult {
                content: "ok".to_string(),
                is_error: false,
            })
        }
    }

    fn make_tool(name: &str, desc: &str) -> Arc<dyn Tool> {
        Arc::new(DummyTool {
            tool_name: name.to_string(),
            tool_description: desc.to_string(),
            schema: json!({"type": "object"}),
        })
    }

    #[tokio::test]
    async fn register_and_get() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("calc", "A calculator")).await;

        let tool = registry.get("calc").await;
        assert!(tool.is_some());
        let tool = tool.unwrap();
        assert_eq!(tool.name(), "calc");
        assert_eq!(tool.description(), "A calculator");
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn list_returns_all_names() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("a", "tool a")).await;
        registry.register(make_tool("b", "tool b")).await;

        let mut names = registry.list().await;
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn tool_definitions_format() {
        let registry = ToolRegistry::new();
        registry
            .register(make_tool("search", "Search the web"))
            .await;

        let defs = registry.tool_definitions().await;
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "search");
        assert_eq!(defs[0].description, "Search the web");
        assert_eq!(defs[0].parameters, json!({"type": "object"}));
    }

    #[tokio::test]
    async fn register_overwrites_existing() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("calc", "old desc")).await;
        registry.register(make_tool("calc", "new desc")).await;

        let tool = registry.get("calc").await.unwrap();
        assert_eq!(tool.description(), "new desc");
        assert_eq!(registry.list().await.len(), 1);
    }

    #[tokio::test]
    async fn clone_shares_state() {
        let registry = ToolRegistry::new();
        let cloned = registry.clone();

        registry.register(make_tool("shared", "shared tool")).await;
        assert!(cloned.get("shared").await.is_some());
    }

    #[tokio::test]
    async fn from_tools_populates_registry() {
        let tools = vec![
            make_tool("alpha", "tool alpha"),
            make_tool("beta", "tool beta"),
        ];
        let registry = ToolRegistry::from_tools(tools);

        assert!(registry.get("alpha").await.is_some());
        assert!(registry.get("beta").await.is_some());
        assert_eq!(registry.get("alpha").await.unwrap().description(), "tool alpha");

        let mut names = registry.list().await;
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn from_tools_empty_vec() {
        let registry = ToolRegistry::from_tools(vec![]);
        assert!(registry.list().await.is_empty());
    }

    #[tokio::test]
    async fn from_tools_last_wins_on_duplicate() {
        let tools = vec![
            make_tool("dup", "first"),
            make_tool("dup", "second"),
        ];
        let registry = ToolRegistry::from_tools(tools);

        let tool = registry.get("dup").await.unwrap();
        assert_eq!(tool.description(), "second");
        assert_eq!(registry.list().await.len(), 1);
    }

    /// Compile-time assertion that ToolRegistry is Send + Sync.
    fn _assert_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolRegistry>();
    }
}
