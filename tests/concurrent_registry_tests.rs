//! Property test for concurrent ToolRegistry reads.
//!
//! For any ToolRegistry with registered tools, concurrent reads from
//! multiple async tasks return consistent results without panics.

use proptest::prelude::*;
use std::sync::Arc;

use async_trait::async_trait;
use kova_sdk::error::KovaError;
use kova_sdk::models::ToolResult;
use kova_sdk::tool::Tool;
use kova_sdk::tool::registry::ToolRegistry;
use serde_json::json;

struct ConcurrentTestTool {
    tool_name: String,
    tool_description: String,
    schema: serde_json::Value,
}

#[async_trait]
impl Tool for ConcurrentTestTool {
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
            content: format!("result from {}", self.tool_name),
            is_error: false,
        })
    }
}

fn make_tool(name: &str, desc: &str) -> Arc<dyn Tool> {
    Arc::new(ConcurrentTestTool {
        tool_name: name.to_string(),
        tool_description: desc.to_string(),
        schema: json!({"type": "object"}),
    })
}

fn arb_tool_names() -> impl Strategy<Value = Vec<String>> {
    proptest::collection::vec("[a-z][a-z0-9_]{1,12}", 1..=10).prop_map(|names| {
        let mut seen = std::collections::HashSet::new();
        names
            .into_iter()
            .filter(|n| seen.insert(n.clone()))
            .collect()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_concurrent_registry_reads(
        tool_names in arb_tool_names(),
        num_readers in 4usize..=16,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let registry = ToolRegistry::new();

            // Register all tools.
            for name in &tool_names {
                registry.register(make_tool(name, &format!("desc for {}", name)));
            }

            let registry = Arc::new(registry);

            // Spawn concurrent readers that each look up every tool.
            let mut handles = Vec::new();
            for _ in 0..num_readers {
                let reg = Arc::clone(&registry);
                let names = tool_names.clone();
                handles.push(tokio::spawn(async move {
                    let mut results = Vec::new();
                    for name in &names {
                        let tool = reg.get(name);
                        results.push((name.clone(), tool));
                    }
                    // Also call list() and tool_definitions() concurrently.
                    let listed = reg.list();
                    let defs = reg.tool_definitions();
                    (results, listed, defs)
                }));
            }

            // Collect all results — no panics should occur.
            for handle in handles {
                let (lookups, listed, defs) = handle.await.expect("task should not panic");

                // Every registered tool should be found.
                for (name, tool_opt) in &lookups {
                    prop_assert!(
                        tool_opt.is_some(),
                        "Tool '{}' should be found in concurrent read",
                        name
                    );
                    let tool = tool_opt.as_ref().unwrap();
                    prop_assert_eq!(
                        tool.name(),
                        name.as_str(),
                        "Tool name should match"
                    );
                }

                // list() should return all registered names.
                prop_assert_eq!(
                    listed.len(),
                    tool_names.len(),
                    "list() should return all registered tools"
                );

                // tool_definitions() should return all definitions.
                prop_assert_eq!(
                    defs.len(),
                    tool_names.len(),
                    "tool_definitions() should return all registered tools"
                );
            }

            Ok(())
        })?;
    }
}
