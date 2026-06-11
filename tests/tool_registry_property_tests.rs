use proptest::prelude::*;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use kova_sdk::error::KovaError;
use kova_sdk::models::ToolResult;
use kova_sdk::tool::Tool;
use kova_sdk::tool::registry::ToolRegistry;

struct PropTool {
    tool_name: String,
    tool_description: String,
    schema: serde_json::Value,
}

#[async_trait]
impl Tool for PropTool {
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

fn arb_tool_spec() -> impl Strategy<Value = (String, String, serde_json::Value)> {
    (
        "[a-z][a-z0-9_]{0,19}",
        "[a-zA-Z0-9 .,]{0,50}",
        prop_oneof![
            Just(json!({"type": "object"})),
            Just(json!({"type": "object", "properties": {"x": {"type": "string"}}})),
            Just(
                json!({"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]})
            ),
        ],
    )
}

fn arb_tool_specs() -> impl Strategy<Value = Vec<(String, String, serde_json::Value)>> {
    proptest::collection::vec(arb_tool_spec(), 1..10).prop_map(|specs| {
        let mut seen = std::collections::HashSet::new();
        specs
            .into_iter()
            .filter(|(name, _, _)| seen.insert(name.clone()))
            .collect()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_tool_registry_lookup_roundtrip(specs in arb_tool_specs()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let registry = ToolRegistry::new();

            let mut expected: Vec<(String, String, serde_json::Value)> = Vec::new();
            for (name, desc, schema) in &specs {
                let tool: Arc<dyn Tool> = Arc::new(PropTool {
                    tool_name: name.clone(),
                    tool_description: desc.clone(),
                    schema: schema.clone(),
                });
                registry.register(tool);
                expected.push((name.clone(), desc.clone(), schema.clone()));
            }

            for (name, desc, schema) in &expected {
                let found = registry.get(name);
                prop_assert!(found.is_some(), "Tool '{}' not found after registration", name);
                let found = found.unwrap();
                prop_assert_eq!(found.name(), name.as_str());
                prop_assert_eq!(found.description(), desc.as_str());
                prop_assert_eq!(found.parameters_schema(), schema.clone());
            }

            let defs = registry.tool_definitions();
            prop_assert_eq!(defs.len(), expected.len());
            for def in defs.iter() {
                let matching = expected.iter().find(|(n, _, _)| n == &def.name);
                prop_assert!(matching.is_some(), "Unexpected tool definition: {}", def.name);
                let (_, desc, schema) = matching.unwrap();
                prop_assert_eq!(&def.description, desc);
                prop_assert_eq!(&def.parameters, schema);
            }

            Ok(())
        })?;
    }
}
