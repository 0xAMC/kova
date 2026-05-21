use super::*;
use crate::provider::openai::types::{
    OaiChatCompletionResponse, OaiChoice, OaiFunctionCall, OaiMessage, OaiToolCall, OaiUsage,
};
use proptest::prelude::*;
use serde_json::json;

fn sample_messages() -> Vec<ConversationMessage> {
    vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "Hello".to_string(),
        }],
    }]
}

fn sample_config() -> InferenceConfig {
    InferenceConfig {
        model: Some("test-model".to_string()),
        max_tokens: Some(100),
        temperature: Some(0.7),
    }
}

fn sample_oai_response() -> OaiChatCompletionResponse {
    OaiChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "test-model".to_string(),
        choices: vec![OaiChoice {
            index: 0,
            message: OaiMessage {
                role: "assistant".to_string(),
                content: Some("Hi there!".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(OaiUsage {
            prompt_tokens: 5,
            completion_tokens: 3,
            total_tokens: 8,
        }),
    }
}

// ── format_request tests ───────────────────────────────────────

#[test]
fn test_format_request_basic_text_message() {
    let req = format_request(&sample_messages(), &[], &sample_config());
    assert_eq!(req.model, "test-model");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "user");
    assert_eq!(req.messages[0].content, Some("Hello".to_string()));
    assert!(req.messages[0].tool_calls.is_none());
    assert_eq!(req.max_tokens, Some(100));
    assert_eq!(req.temperature, Some(0.7));
}

#[test]
fn test_format_request_omits_none_fields() {
    let config = InferenceConfig {
        model: Some("m".to_string()),
        max_tokens: None,
        temperature: None,
    };
    let req = format_request(&sample_messages(), &[], &config);
    let json_val = serde_json::to_value(&req).unwrap();
    assert!(json_val.get("max_tokens").is_none());
    assert!(json_val.get("temperature").is_none());
    assert!(json_val.get("tools").is_none());
    assert!(json_val.get("stream").is_none());
}

#[test]
fn test_format_request_with_tools() {
    let tools = vec![ToolDefinition {
        name: "search".to_string(),
        description: "Search the web".to_string(),
        parameters: json!({"type": "object", "properties": {}}),
    }];
    let req = format_request(&sample_messages(), &tools, &sample_config());
    let oai_tools = req.tools.unwrap();
    assert_eq!(oai_tools.len(), 1);
    assert_eq!(oai_tools[0].tool_type, "function");
    assert_eq!(oai_tools[0].function.name, "search");
    assert_eq!(oai_tools[0].function.description, "Search the web");
}

#[test]
fn test_format_request_tool_use_content_block() {
    let messages = vec![ConversationMessage {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            input: json!({"city": "Seattle"}),
        }],
    }];
    let req = format_request(&messages, &[], &sample_config());
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "assistant");
    assert!(req.messages[0].content.is_none());
    let tc = req.messages[0].tool_calls.as_ref().unwrap();
    assert_eq!(tc.len(), 1);
    assert_eq!(tc[0].id, "call_1");
    assert_eq!(tc[0].function.name, "get_weather");
    assert_eq!(tc[0].function.arguments, r#"{"city":"Seattle"}"#);
}

#[test]
fn test_format_request_tool_result_content_block() {
    let messages = vec![ConversationMessage {
        role: Role::Tool,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: "call_1".to_string(),
            content: "72°F".to_string(),
            is_error: false,
        }],
    }];
    let req = format_request(&messages, &[], &sample_config());
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "tool");
    assert_eq!(req.messages[0].content, Some("72°F".to_string()));
    assert_eq!(req.messages[0].tool_call_id, Some("call_1".to_string()));
}

// ── format_response tests ──────────────────────────────────────

#[test]
fn test_format_response_basic() {
    let resp = format_response(sample_oai_response()).unwrap();
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(resp.content.len(), 1);
    assert_eq!(
        resp.content[0],
        ContentBlock::Text {
            text: "Hi there!".to_string()
        }
    );
    let usage = resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 5);
    assert_eq!(usage.output_tokens, 3);
    assert_eq!(usage.total_tokens, 8);
}

#[test]
fn test_format_response_without_usage() {
    let mut resp = sample_oai_response();
    resp.usage = None;
    let model_resp = format_response(resp).unwrap();
    assert!(model_resp.usage.is_none());
}

#[test]
fn test_format_response_tool_calls_finish_reason() {
    let oai_resp = OaiChatCompletionResponse {
        id: "id".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "m".to_string(),
        choices: vec![OaiChoice {
            index: 0,
            message: OaiMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![OaiToolCall {
                    id: "tc_1".to_string(),
                    call_type: "function".to_string(),
                    function: OaiFunctionCall {
                        name: "search".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some("tool_calls".to_string()),
        }],
        usage: None,
    };
    let resp = format_response(oai_resp).unwrap();
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    match &resp.content[0] {
        ContentBlock::ToolUse { id, name, .. } => {
            assert_eq!(id, "tc_1");
            assert_eq!(name, "search");
        }
        other => panic!("Expected ToolUse, got {:?}", other),
    }
}

#[test]
fn test_format_response_unknown_finish_reason() {
    let mut resp = sample_oai_response();
    resp.choices[0].finish_reason = Some("content_filter".to_string());
    let model_resp = format_response(resp).unwrap();
    assert_eq!(
        model_resp.stop_reason,
        StopReason::Unknown("content_filter".to_string())
    );
}

#[test]
fn test_format_response_empty_choices_error() {
    let oai_resp = OaiChatCompletionResponse {
        id: "id".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "m".to_string(),
        choices: vec![],
        usage: None,
    };
    let err = format_response(oai_resp).unwrap_err();
    match err {
        KovaError::Provider { message, .. } => assert!(message.contains("No choices")),
        other => panic!("Expected Provider error, got {:?}", other),
    }
}

// ── Property tests ─────────────────────────────────────────────

fn arb_nonempty_text() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 .,!?]{1,80}"
}

fn arb_tool_use_block() -> impl Strategy<Value = ContentBlock> {
    (
        "[a-z0-9]{4,12}",
        "[a-z_]{2,15}",
        Just(json!({"key": "value"})),
    )
        .prop_map(|(id, name, input)| ContentBlock::ToolUse { id, name, input })
}

fn arb_text_only_assistant() -> impl Strategy<Value = (ConversationMessage, StopReason)> {
    proptest::collection::vec(arb_nonempty_text(), 1..=3).prop_map(|texts| {
        let content = texts
            .into_iter()
            .map(|text| ContentBlock::Text { text })
            .collect();
        (
            ConversationMessage {
                role: Role::Assistant,
                content,
            },
            StopReason::EndTurn,
        )
    })
}

fn arb_tool_use_only_assistant() -> impl Strategy<Value = (ConversationMessage, StopReason)> {
    proptest::collection::vec(arb_tool_use_block(), 1..=3).prop_map(|blocks| {
        (
            ConversationMessage {
                role: Role::Assistant,
                content: blocks,
            },
            StopReason::ToolUse,
        )
    })
}

fn arb_mixed_assistant() -> impl Strategy<Value = (ConversationMessage, StopReason)> {
    (
        arb_nonempty_text(),
        proptest::collection::vec(arb_tool_use_block(), 1..=2),
    )
        .prop_map(|(text, tool_blocks)| {
            let mut content = vec![ContentBlock::Text { text }];
            content.extend(tool_blocks);
            (
                ConversationMessage {
                    role: Role::Assistant,
                    content,
                },
                StopReason::ToolUse,
            )
        })
}

fn arb_assistant_message_with_stop() -> impl Strategy<Value = (ConversationMessage, StopReason)> {
    prop_oneof![
        arb_text_only_assistant(),
        arb_tool_use_only_assistant(),
        arb_mixed_assistant()
    ]
}

fn stop_reason_to_finish_reason(sr: &StopReason) -> String {
    match sr {
        StopReason::EndTurn => "stop".to_string(),
        StopReason::ToolUse => "tool_calls".to_string(),
        StopReason::MaxTokens => "length".to_string(),
        StopReason::Unknown(s) => s.clone(),
    }
}

fn arb_tool_result_block() -> impl Strategy<Value = ContentBlock> {
    ("[a-z0-9]{4,12}", arb_nonempty_text(), proptest::bool::ANY).prop_map(
        |(tool_use_id, content, is_error)| ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        },
    )
}

fn arb_user_text_message() -> impl Strategy<Value = ConversationMessage> {
    proptest::collection::vec(arb_nonempty_text(), 1..=3).prop_map(|texts| ConversationMessage {
        role: Role::User,
        content: texts
            .into_iter()
            .map(|text| ContentBlock::Text { text })
            .collect(),
    })
}

fn arb_tool_result_message() -> impl Strategy<Value = ConversationMessage> {
    proptest::collection::vec(arb_tool_result_block(), 1..=3).prop_map(|blocks| {
        ConversationMessage {
            role: Role::Tool,
            content: blocks,
        }
    })
}

fn arb_unrecognized_finish_reason() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_. -]{1,50}".prop_filter("must not be a recognized finish_reason", |s| {
        s != "stop" && s != "tool_calls" && s != "length"
    })
}

fn arb_unrecognized_role() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_. -]{1,50}".prop_filter("must not be a recognized role", |s| {
        s != "system" && s != "user" && s != "assistant" && s != "tool"
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_format_conversion_round_trip(
        (assistant_msg, expected_stop) in arb_assistant_message_with_stop(),
        config in (
            proptest::option::of("[a-z0-9-]{1,15}"),
            proptest::option::of(1..4096u32),
            proptest::option::of(0.0f32..2.0f32),
        ).prop_map(|(model, max_tokens, temperature)| InferenceConfig { model, max_tokens, temperature }),
    ) {
        let oai_req = format_request(std::slice::from_ref(&assistant_msg), &[], &config);
        prop_assert_eq!(oai_req.messages.len(), 1);
        let oai_msg = &oai_req.messages[0];
        prop_assert_eq!(&oai_msg.role, "assistant");

        let oai_response = OaiChatCompletionResponse {
            id: "test-id".to_string(), object: "chat.completion".to_string(), created: 0,
            model: config.model.clone().unwrap_or_default(),
            choices: vec![OaiChoice {
                index: 0,
                message: oai_msg.clone(),
                finish_reason: Some(stop_reason_to_finish_reason(&expected_stop)),
            }],
            usage: None,
        };

        let model_response = format_response(oai_response).unwrap();
        prop_assert_eq!(&model_response.stop_reason, &expected_stop);

        let expected_texts: Vec<&str> = assistant_msg.content.iter().filter_map(|b| {
            if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
        }).collect();
        let expected_tool_uses: Vec<(&str, &str, &serde_json::Value)> = assistant_msg.content.iter().filter_map(|b| {
            if let ContentBlock::ToolUse { id, name, input } = b { Some((id.as_str(), name.as_str(), input)) } else { None }
        }).collect();

        let actual_texts: Vec<&str> = model_response.content.iter().filter_map(|b| {
            if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
        }).collect();
        let actual_tool_uses: Vec<(&str, &str, &serde_json::Value)> = model_response.content.iter().filter_map(|b| {
            if let ContentBlock::ToolUse { id, name, input } = b { Some((id.as_str(), name.as_str(), input)) } else { None }
        }).collect();

        prop_assert_eq!(actual_texts.join(""), expected_texts.join(""));
        prop_assert_eq!(actual_tool_uses.len(), expected_tool_uses.len());
        for (expected, actual) in expected_tool_uses.iter().zip(actual_tool_uses.iter()) {
            prop_assert_eq!(actual.0, expected.0);
            prop_assert_eq!(actual.1, expected.1);
            prop_assert_eq!(actual.2, expected.2);
        }
    }

    #[test]
    fn prop_format_request_text_maps_to_content_field(msg in arb_user_text_message()) {
        let oai_req = format_request(std::slice::from_ref(&msg), &[], &InferenceConfig::default());
        prop_assert_eq!(oai_req.messages.len(), 1);
        let oai_msg = &oai_req.messages[0];
        let expected_text: String = msg.content.iter().filter_map(|b| {
            if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
        }).collect::<Vec<_>>().join("");
        prop_assert!(oai_msg.content.is_some());
        prop_assert_eq!(oai_msg.content.as_deref().unwrap(), &expected_text);
        prop_assert!(oai_msg.tool_calls.is_none());
    }

    #[test]
    fn prop_format_request_tool_use_maps_to_tool_calls(msg in arb_tool_use_only_assistant()) {
        let (assistant_msg, _stop) = msg;
        let oai_req = format_request(std::slice::from_ref(&assistant_msg), &[], &InferenceConfig::default());
        prop_assert_eq!(oai_req.messages.len(), 1);
        let oai_msg = &oai_req.messages[0];
        let expected_tool_uses: Vec<(&str, &str, &serde_json::Value)> = assistant_msg.content.iter().filter_map(|b| {
            if let ContentBlock::ToolUse { id, name, input } = b { Some((id.as_str(), name.as_str(), input)) } else { None }
        }).collect();
        prop_assert!(oai_msg.tool_calls.is_some());
        let tool_calls = oai_msg.tool_calls.as_ref().unwrap();
        prop_assert_eq!(tool_calls.len(), expected_tool_uses.len());
        for (tc, (exp_id, exp_name, exp_input)) in tool_calls.iter().zip(expected_tool_uses.iter()) {
            prop_assert_eq!(&tc.call_type, "function");
            prop_assert_eq!(&tc.id, exp_id);
            prop_assert_eq!(&tc.function.name, exp_name);
            let parsed: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
            prop_assert_eq!(&parsed, *exp_input);
        }
    }

    #[test]
    fn prop_format_request_tool_result_maps_to_tool_role(msg in arb_tool_result_message()) {
        let oai_req = format_request(std::slice::from_ref(&msg), &[], &InferenceConfig::default());
        let expected_results: Vec<(&str, &str)> = msg.content.iter().filter_map(|b| {
            if let ContentBlock::ToolResult { tool_use_id, content, .. } = b {
                Some((tool_use_id.as_str(), content.as_str()))
            } else { None }
        }).collect();
        prop_assert_eq!(oai_req.messages.len(), expected_results.len());
        for (oai_msg, (exp_id, exp_content)) in oai_req.messages.iter().zip(expected_results.iter()) {
            prop_assert_eq!(&oai_msg.role, "tool");
            prop_assert_eq!(oai_msg.tool_call_id.as_deref().unwrap(), *exp_id);
            prop_assert_eq!(oai_msg.content.as_deref().unwrap(), *exp_content);
        }
    }

    #[test]
    fn prop_unrecognized_finish_reason_maps_to_unknown(reason in arb_unrecognized_finish_reason()) {
        let oai_response = OaiChatCompletionResponse {
            id: "test-id".to_string(), object: "chat.completion".to_string(), created: 0,
            model: "test-model".to_string(),
            choices: vec![OaiChoice {
                index: 0,
                message: OaiMessage { role: "assistant".to_string(), content: Some("hello".to_string()),
                    tool_calls: None, tool_call_id: None, name: None },
                finish_reason: Some(reason.clone()),
            }],
            usage: None,
        };
        let model_response = format_response(oai_response).unwrap();
        prop_assert_eq!(model_response.stop_reason, StopReason::Unknown(reason.clone()));
    }

    #[test]
    fn prop_unrecognized_role_returns_provider_error(role in arb_unrecognized_role()) {
        let oai_response = OaiChatCompletionResponse {
            id: "test-id".to_string(), object: "chat.completion".to_string(), created: 0,
            model: "test-model".to_string(),
            choices: vec![OaiChoice {
                index: 0,
                message: OaiMessage { role: role.clone(), content: Some("hello".to_string()),
                    tool_calls: None, tool_call_id: None, name: None },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        let result = format_response(oai_response);
        prop_assert!(result.is_err());
        match result.unwrap_err() {
            KovaError::Provider { message, status_code } => {
                prop_assert!(message.contains("Unrecognized role"));
                prop_assert!(message.contains(&role));
                prop_assert_eq!(status_code, None);
            }
            other => prop_assert!(false, "Expected KovaError::Provider, got {:?}", other),
        }
    }
}
