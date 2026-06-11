use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::models::ToolDefinition;

use super::Tool;

struct RegistryInner {
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Lazily-built definitions for all registered tools. Building one
    /// requires cloning every tool's JSON schema, and the agent needs the
    /// list on every LLM call, so it is cached until the next `register`.
    cached_definitions: Option<Arc<Vec<ToolDefinition>>>,
}

/// Thread-safe registry for storing and resolving tools by name.
///
/// Lookups are synchronous: the internal lock is only held for map access
/// and never across an `.await` point.
pub struct ToolRegistry {
    inner: Arc<RwLock<RegistryInner>>,
}

impl ToolRegistry {
    /// Create a new empty `ToolRegistry`.
    pub fn new() -> Self {
        Self::from_tools(Vec::new())
    }

    /// Create a `ToolRegistry` pre-populated with the given tools.
    pub fn from_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map = HashMap::new();
        for tool in tools {
            map.insert(tool.name().to_string(), tool);
        }
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                tools: map,
                cached_definitions: None,
            })),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, RegistryInner> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, RegistryInner> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Register a tool. Overwrites any existing tool with the same name.
    pub fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        let mut inner = self.write();
        inner.tools.insert(name, tool);
        inner.cached_definitions = None;
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.read().tools.get(name).cloned()
    }

    /// List all registered tool names.
    pub fn list(&self) -> Vec<String> {
        self.read().tools.keys().cloned().collect()
    }

    /// Return `ToolDefinition` entries for all registered tools.
    ///
    /// The list is built once and cached; subsequent calls are cheap until
    /// a new tool is registered.
    pub fn tool_definitions(&self) -> Arc<Vec<ToolDefinition>> {
        if let Some(defs) = &self.read().cached_definitions {
            return Arc::clone(defs);
        }

        let mut inner = self.write();
        if let Some(defs) = &inner.cached_definitions {
            return Arc::clone(defs);
        }
        let defs = Arc::new(
            inner
                .tools
                .values()
                .map(|tool| ToolDefinition {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.parameters_schema(),
                })
                .collect::<Vec<_>>(),
        );
        inner.cached_definitions = Some(Arc::clone(&defs));
        defs
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
            inner: Arc::clone(&self.inner),
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

    #[test]
    fn register_and_get() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("calc", "A calculator"));

        let tool = registry.get("calc");
        assert!(tool.is_some());
        let tool = tool.unwrap();
        assert_eq!(tool.name(), "calc");
        assert_eq!(tool.description(), "A calculator");
    }

    #[test]
    fn get_missing_returns_none() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn list_returns_all_names() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("a", "tool a"));
        registry.register(make_tool("b", "tool b"));

        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn tool_definitions_format() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("search", "Search the web"));

        let defs = registry.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "search");
        assert_eq!(defs[0].description, "Search the web");
        assert_eq!(defs[0].parameters, json!({"type": "object"}));
    }

    #[test]
    fn tool_definitions_cached_until_register() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("a", "tool a"));

        let first = registry.tool_definitions();
        let second = registry.tool_definitions();
        assert!(Arc::ptr_eq(&first, &second), "definitions are cached");

        registry.register(make_tool("b", "tool b"));
        let third = registry.tool_definitions();
        assert!(!Arc::ptr_eq(&first, &third), "register invalidates cache");
        assert_eq!(third.len(), 2);
    }

    #[test]
    fn register_overwrites_existing() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("calc", "old desc"));
        registry.register(make_tool("calc", "new desc"));

        let tool = registry.get("calc").unwrap();
        assert_eq!(tool.description(), "new desc");
        assert_eq!(registry.list().len(), 1);
    }

    #[test]
    fn clone_shares_state() {
        let registry = ToolRegistry::new();
        let cloned = registry.clone();

        registry.register(make_tool("shared", "shared tool"));
        assert!(cloned.get("shared").is_some());
    }

    #[test]
    fn from_tools_populates_registry() {
        let tools = vec![
            make_tool("alpha", "tool alpha"),
            make_tool("beta", "tool beta"),
        ];
        let registry = ToolRegistry::from_tools(tools);

        assert!(registry.get("alpha").is_some());
        assert!(registry.get("beta").is_some());
        assert_eq!(registry.get("alpha").unwrap().description(), "tool alpha");

        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn from_tools_empty_vec() {
        let registry = ToolRegistry::from_tools(vec![]);
        assert!(registry.list().is_empty());
    }

    #[test]
    fn from_tools_last_wins_on_duplicate() {
        let tools = vec![make_tool("dup", "first"), make_tool("dup", "second")];
        let registry = ToolRegistry::from_tools(tools);

        let tool = registry.get("dup").unwrap();
        assert_eq!(tool.description(), "second");
        assert_eq!(registry.list().len(), 1);
    }

    /// Compile-time assertion that ToolRegistry is Send + Sync.
    fn _assert_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolRegistry>();
    }
}
