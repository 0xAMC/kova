use proptest::prelude::*;
use std::sync::Arc;

use kova_sdk::mcp::tool::McpTool;
use kova_sdk::mcp::{McpClient, McpToolDefinition};
use kova_sdk::tool::Tool;

fn arb_mcp_tool_definition() -> impl Strategy<Value = McpToolDefinition> {
    (
        "[a-z_]{1,20}",
        proptest::option::of("[A-Za-z0-9 ]{1,50}"),
        proptest::option::of(Just(serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            }
        }))),
    )
        .prop_map(|(name, description, input_schema)| McpToolDefinition {
            name,
            description,
            input_schema,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_mcp_tool_conversion(def in arb_mcp_tool_definition()) {
        let expected_name = def.name.clone();
        let expected_desc = def.description.clone().unwrap_or_default();
        let expected_schema = def.input_schema.clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object"}));

        let client = Arc::new(McpClient::new_for_test());
        let server_name = "test_server";
        let mcp_tool = McpTool::new(def, client, server_name);

        let expected_qualified = format!("{}__{}", server_name, expected_name);
        prop_assert_eq!(mcp_tool.name(), expected_qualified.as_str());
        prop_assert_eq!(mcp_tool.bare_name(), expected_name.as_str());
        prop_assert_eq!(mcp_tool.description(), expected_desc.as_str());
        prop_assert_eq!(mcp_tool.parameters_schema(), expected_schema);
    }
}
