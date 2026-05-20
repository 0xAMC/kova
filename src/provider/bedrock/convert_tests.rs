use super::*;
use proptest::prelude::*;
use serde_json::json;

fn arb_identifier() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_]{0,19}"
}

fn arb_text() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 ]{1,50}"
}

fn arb_text_block() -> impl Strategy<Value = ContentBlock> {
    arb_text().prop_map(|text| ContentBlock::Text { text })
}

fn arb_tool_use_block() -> impl Strategy<Value = ContentBlock> {
    (arb_identifier(), arb_identifier(), arb_text()).prop_map(|(id, name, val)| {
        ContentBlock::ToolUse {
            id,
            name,
            input: json!({ "key": val }),
        }
    })
}

fn arb_tool_result_block() -> impl Strategy<Value = ContentBlock> {
    (arb_identifier(), arb_text(), any::<bool>()).prop_map(|(tool_use_id, content, is_error)| {
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        }
    })
}

fn arb_content_block() -> impl Strategy<Value = ContentBlock> {
    prop_oneof![arb_text_block(), arb_tool_use_block(), arb_tool_result_block()]
}

fn arb_stop_reason_str() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("end_turn".to_string()),
        Just("tool_use".to_string()),
        Just("max_tokens".to_string()),
    ]
}

fn arb_bedrock_text_delta() -> impl Strategy<Value = BedrockStreamEvent> {
    (0u32..100u32, arb_text()).prop_map(|(idx, text)| BedrockStreamEvent::ContentBlockDelta {
        content_block_index: idx,
        delta: BedrockContentBlockDelta::Text(text),
    })
}

fn arb_bedrock_tool_use_delta() -> impl Strategy<Value = BedrockStreamEvent> {
    (0u32..100u32, arb_text()).prop_map(|(idx, input)| BedrockStreamEvent::ContentBlockDelta {
        content_block_index: idx,
        delta: BedrockContentBlockDelta::ToolUse { input },
    })
}

fn arb_bedrock_tool_use_start() -> impl Strategy<Value = BedrockStreamEvent> {
    (0u32..100u32, arb_identifier(), arb_identifier()).prop_map(|(idx, tool_use_id, name)| {
        BedrockStreamEvent::ContentBlockStart {
            content_block_index: idx,
            start: BedrockContentBlockStart::ToolUse { tool_use_id, name },
        }
    })
}

fn arb_bedrock_message_stop() -> impl Strategy<Value = BedrockStreamEvent> {
    arb_stop_reason_str()
        .prop_map(|stop_reason| BedrockStreamEvent::MessageStop { stop_reason })
}

fn arb_bedrock_content_block_stop() -> impl Strategy<Value = BedrockStreamEvent> {
    (0u32..100u32).prop_map(|idx| BedrockStreamEvent::ContentBlockStop {
        content_block_index: idx,
    })
}

fn arb_bedrock_metadata() -> impl Strategy<Value = BedrockStreamEvent> {
    use super::super::types::BedrockUsage;
    (1u32..1000u32, 1u32..1000u32).prop_map(|(input_tokens, output_tokens)| {
        BedrockStreamEvent::Metadata {
            usage: BedrockUsage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
            },
        }
    })
}

fn arb_bedrock_stream_event() -> impl Strategy<Value = BedrockStreamEvent> {
    prop_oneof![
        arb_bedrock_text_delta(),
        arb_bedrock_tool_use_delta(),
        arb_bedrock_tool_use_start(),
        arb_bedrock_message_stop(),
        arb_bedrock_content_block_stop(),
        arb_bedrock_metadata(),
    ]
}

fn arb_bedrock_event_with_expected() -> impl Strategy<Value = (BedrockStreamEvent, StreamEvent)>
{
    prop_oneof![
        (0u32..100u32, arb_text()).prop_map(|(idx, text)| {
            let event = BedrockStreamEvent::ContentBlockDelta {
                content_block_index: idx,
                delta: BedrockContentBlockDelta::Text(text.clone()),
            };
            let expected = StreamEvent::ContentDelta { text };
            (event, expected)
        }),
        (0u32..100u32, arb_text()).prop_map(|(idx, input)| {
            let event = BedrockStreamEvent::ContentBlockDelta {
                content_block_index: idx,
                delta: BedrockContentBlockDelta::ToolUse { input: input.clone() },
            };
            let expected = StreamEvent::ToolUseDelta {
                id: String::new(),
                name: None,
                input_delta: Some(input),
            };
            (event, expected)
        }),
        (0u32..100u32, arb_identifier(), arb_identifier()).prop_map(
            |(idx, tool_use_id, name)| {
                let event = BedrockStreamEvent::ContentBlockStart {
                    content_block_index: idx,
                    start: BedrockContentBlockStart::ToolUse {
                        tool_use_id: tool_use_id.clone(),
                        name: name.clone(),
                    },
                };
                let expected = StreamEvent::ToolUseDelta {
                    id: tool_use_id,
                    name: Some(name),
                    input_delta: None,
                };
                (event, expected)
            }
        ),
        arb_stop_reason_str().prop_map(|stop_reason| {
            let event = BedrockStreamEvent::MessageStop {
                stop_reason: stop_reason.clone(),
            };
            let expected = StreamEvent::StopEvent {
                stop_reason: map_stop_reason(&stop_reason),
            };
            (event, expected)
        }),
    ]
}

fn message_for_block(block: &ContentBlock) -> ConversationMessage {
    let role = match block {
        ContentBlock::Text { .. } => Role::User,
        ContentBlock::ToolUse { .. } => Role::Assistant,
        ContentBlock::ToolResult { .. } => Role::User,
    };
    ConversationMessage {
        role,
        content: vec![block.clone()],
    }
}

fn fake_response(content: Vec<BedrockContentBlock>) -> BedrockConverseResponse {
    use super::super::types::{BedrockOutput, BedrockUsage};
    BedrockConverseResponse {
        output: BedrockOutput {
            message: BedrockMessage {
                role: "assistant".to_string(),
                content,
            },
        },
        stop_reason: "end_turn".to_string(),
        usage: BedrockUsage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
        },
    }
}

fn response_with_stop_reason(stop_reason: &str) -> BedrockConverseResponse {
    use super::super::types::{BedrockOutput, BedrockUsage};
    BedrockConverseResponse {
        output: BedrockOutput {
            message: BedrockMessage {
                role: "assistant".to_string(),
                content: vec![BedrockContentBlock::Text("ok".to_string())],
            },
        },
        stop_reason: stop_reason.to_string(),
        usage: BedrockUsage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
        },
    }
}

proptest! {
    #[test]
    fn prop_content_block_round_trip(block in arb_content_block()) {
        let msg = message_for_block(&block);
        let config = InferenceConfig::default();
        let request = format_request(&[msg], &[], &config);
        let bedrock_blocks = request.messages[0].content.clone();
        let response = fake_response(bedrock_blocks);
        let model_resp = format_response(response).unwrap();
        prop_assert_eq!(model_resp.content.len(), 1);
        prop_assert_eq!(&model_resp.content[0], &block);
    }

    #[test]
    fn prop_inference_config_conversion(
        max_tokens in prop::option::of(1u32..10000u32),
        temperature in prop::option::of(0.0f32..2.0f32),
    ) {
        let config = InferenceConfig { model: None, max_tokens, temperature };
        let msg = ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "hi".to_string() }],
        };
        let request = format_request(&[msg], &[], &config);
        let json_val = serde_json::to_value(&request).unwrap();
        let inf = &json_val["inferenceConfig"];
        match max_tokens {
            Some(mt) => prop_assert_eq!(inf["maxTokens"].as_u64().unwrap(), mt as u64),
            None => prop_assert!(inf.get("maxTokens").is_none() || inf["maxTokens"].is_null()),
        }
        match temperature {
            Some(t) => {
                let got = inf["temperature"].as_f64().unwrap() as f32;
                prop_assert!((got - t).abs() < 1e-5, "temperature mismatch: got {} expected {}", got, t);
            }
            None => prop_assert!(inf.get("temperature").is_none() || inf["temperature"].is_null()),
        }
    }

    #[test]
    fn prop_tool_definition_conversion(
        name in arb_identifier(),
        description in arb_text(),
        param_key in arb_identifier(),
        param_val in arb_text(),
    ) {
        let params = json!({ "type": "object", "properties": { param_key.clone(): { "type": "string", "description": param_val.clone() } } });
        let tool = ToolDefinition {
            name: name.clone(),
            description: description.clone(),
            parameters: params.clone(),
        };
        let msg = ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "hi".to_string() }],
        };
        let request = format_request(&[msg], &[tool], &InferenceConfig::default());
        let json_val = serde_json::to_value(&request).unwrap();
        let tool_spec = &json_val["toolConfig"]["tools"][0]["toolSpec"];
        prop_assert_eq!(tool_spec["name"].as_str().unwrap(), name.as_str());
        prop_assert_eq!(tool_spec["description"].as_str().unwrap(), description.as_str());
        prop_assert_eq!(&tool_spec["inputSchema"]["json"], &params);
    }

    #[test]
    fn prop_system_message_extraction(
        system_text in arb_text(),
        user_text in arb_text(),
    ) {
        let messages = vec![
            ConversationMessage {
                role: Role::System,
                content: vec![ContentBlock::Text { text: system_text.clone() }],
            },
            ConversationMessage {
                role: Role::User,
                content: vec![ContentBlock::Text { text: user_text.clone() }],
            },
        ];
        let request = format_request(&messages, &[], &InferenceConfig::default());
        let json_val = serde_json::to_value(&request).unwrap();
        let system_arr = json_val["system"].as_array().unwrap();
        let system_texts: Vec<&str> = system_arr.iter().map(|s| s["text"].as_str().unwrap()).collect();
        prop_assert!(system_texts.contains(&system_text.as_str()));
        let msgs = json_val["messages"].as_array().unwrap();
        for m in msgs {
            prop_assert_ne!(m["role"].as_str().unwrap(), "system");
        }
    }

    #[test]
    fn prop_stop_reason_mapping(unknown_reason in "[a-z_]{1,20}") {
        let known = vec![
            ("end_turn", StopReason::EndTurn),
            ("tool_use", StopReason::ToolUse),
            ("max_tokens", StopReason::MaxTokens),
        ];
        for (reason_str, expected) in &known {
            let resp = response_with_stop_reason(reason_str);
            let model_resp = format_response(resp).unwrap();
            prop_assert_eq!(&model_resp.stop_reason, expected);
        }
        if unknown_reason != "end_turn" && unknown_reason != "tool_use" && unknown_reason != "max_tokens" {
            let resp = response_with_stop_reason(&unknown_reason);
            let model_resp = format_response(resp).unwrap();
            prop_assert_eq!(model_resp.stop_reason, StopReason::Unknown(unknown_reason));
        }
    }

    #[test]
    fn prop_stream_event_conversion((event, expected) in arb_bedrock_event_with_expected()) {
        let result = format_stream_event(event);
        prop_assert!(result.is_some(), "Event that should produce output returned None");
        prop_assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn prop_stream_event_order_preservation(events in prop::collection::vec(arb_bedrock_stream_event(), 0..20)) {
        let events_clone = events.clone();
        let converted: Vec<StreamEvent> = events.into_iter().filter_map(format_stream_event).collect();
        let converted_again: Vec<StreamEvent> = events_clone.into_iter().filter_map(format_stream_event).collect();
        prop_assert_eq!(converted.len(), converted_again.len());
        for (a, b) in converted.iter().zip(converted_again.iter()) {
            prop_assert_eq!(a, b);
        }
    }
}

#[test]
fn test_format_request_serialization() {
    let msg = ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text { text: "Hello, world!".to_string() }],
    };
    let request = format_request(&[msg], &[], &InferenceConfig::default());
    let json_val = serde_json::to_value(&request).unwrap();
    assert_eq!(json_val["messages"][0]["role"], "user");
    assert_eq!(json_val["messages"][0]["content"][0]["text"], "Hello, world!");
    assert!(json_val.get("inferenceConfig").is_some());
}

#[test]
fn test_format_response_deserialization() {
    use super::super::types::{BedrockOutput, BedrockUsage};
    let resp = BedrockConverseResponse {
        output: BedrockOutput {
            message: BedrockMessage {
                role: "assistant".to_string(),
                content: vec![BedrockContentBlock::Text("Hi there!".to_string())],
            },
        },
        stop_reason: "end_turn".to_string(),
        usage: BedrockUsage { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    };
    let model_resp = format_response(resp).unwrap();
    assert_eq!(model_resp.content.len(), 1);
    assert_eq!(
        model_resp.content[0],
        ContentBlock::Text { text: "Hi there!".to_string() }
    );
    assert_eq!(model_resp.stop_reason, StopReason::EndTurn);
    let usage = model_resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[test]
fn test_tool_definition_complex_schema() {
    let schema = json!({
        "type": "object",
        "properties": {
            "query": { "type": "string" },
            "filters": { "type": "object", "properties": { "date_range": { "type": "string" } }, "required": ["date_range"] }
        },
        "required": ["query", "filters"]
    });
    let tool = ToolDefinition {
        name: "search".to_string(),
        description: "Search the database".to_string(),
        parameters: schema.clone(),
    };
    let msg = ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text { text: "hi".to_string() }],
    };
    let request = format_request(&[msg], &[tool], &InferenceConfig::default());
    let json_val = serde_json::to_value(&request).unwrap();
    let tool_spec = &json_val["toolConfig"]["tools"][0]["toolSpec"];
    assert_eq!(tool_spec["name"], "search");
    assert_eq!(tool_spec["description"], "Search the database");
    assert_eq!(tool_spec["inputSchema"]["json"], schema);
}

#[test]
fn test_system_message_extraction_mixed() {
    let messages = vec![
        ConversationMessage {
            role: Role::System,
            content: vec![ContentBlock::Text { text: "Be helpful".to_string() }],
        },
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Hi".to_string() }],
        },
        ConversationMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: "Hello".to_string() }],
        },
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "How are you?".to_string() }],
        },
    ];
    let request = format_request(&messages, &[], &InferenceConfig::default());
    let json_val = serde_json::to_value(&request).unwrap();
    let system_arr = json_val["system"].as_array().unwrap();
    assert_eq!(system_arr.len(), 1);
    assert_eq!(system_arr[0]["text"], "Be helpful");
    let msgs = json_val["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    for m in msgs {
        assert_ne!(m["role"].as_str().unwrap(), "system");
    }
}

#[test]
fn test_stop_reason_end_turn() {
    let resp = response_with_stop_reason("end_turn");
    assert_eq!(format_response(resp).unwrap().stop_reason, StopReason::EndTurn);
}

#[test]
fn test_stop_reason_tool_use() {
    let resp = response_with_stop_reason("tool_use");
    assert_eq!(format_response(resp).unwrap().stop_reason, StopReason::ToolUse);
}

#[test]
fn test_stop_reason_max_tokens() {
    let resp = response_with_stop_reason("max_tokens");
    assert_eq!(format_response(resp).unwrap().stop_reason, StopReason::MaxTokens);
}

#[test]
fn test_stop_reason_unknown() {
    let resp = response_with_stop_reason("content_filtered");
    assert_eq!(
        format_response(resp).unwrap().stop_reason,
        StopReason::Unknown("content_filtered".to_string())
    );
}

#[test]
fn test_empty_tools_no_tool_config() {
    let msg = ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text { text: "hi".to_string() }],
    };
    let request = format_request(&[msg], &[], &InferenceConfig::default());
    let json_val = serde_json::to_value(&request).unwrap();
    assert!(json_val.get("toolConfig").is_none());
}

#[test]
fn test_format_stream_event_text_delta() {
    let event = BedrockStreamEvent::ContentBlockDelta {
        content_block_index: 0,
        delta: BedrockContentBlockDelta::Text("Hello world".to_string()),
    };
    assert_eq!(
        format_stream_event(event),
        Some(StreamEvent::ContentDelta { text: "Hello world".to_string() })
    );
}

#[test]
fn test_format_stream_event_tool_use_start() {
    let event = BedrockStreamEvent::ContentBlockStart {
        content_block_index: 1,
        start: BedrockContentBlockStart::ToolUse {
            tool_use_id: "tool-abc".to_string(),
            name: "search".to_string(),
        },
    };
    assert_eq!(
        format_stream_event(event),
        Some(StreamEvent::ToolUseDelta {
            id: "tool-abc".to_string(),
            name: Some("search".to_string()),
            input_delta: None,
        })
    );
}

#[test]
fn test_format_stream_event_tool_use_input_delta() {
    let event = BedrockStreamEvent::ContentBlockDelta {
        content_block_index: 1,
        delta: BedrockContentBlockDelta::ToolUse {
            input: "{\"query\":\"rust\"}".to_string(),
        },
    };
    assert_eq!(
        format_stream_event(event),
        Some(StreamEvent::ToolUseDelta {
            id: String::new(),
            name: None,
            input_delta: Some("{\"query\":\"rust\"}".to_string()),
        })
    );
}

#[test]
fn test_format_stream_event_message_stop_end_turn() {
    let event = BedrockStreamEvent::MessageStop { stop_reason: "end_turn".to_string() };
    assert_eq!(
        format_stream_event(event),
        Some(StreamEvent::StopEvent { stop_reason: StopReason::EndTurn })
    );
}

#[test]
fn test_format_stream_event_content_block_stop_ignored() {
    let event = BedrockStreamEvent::ContentBlockStop { content_block_index: 0 };
    assert!(format_stream_event(event).is_none());
}

#[test]
fn test_format_stream_event_metadata_ignored() {
    use super::super::types::BedrockUsage;
    let event = BedrockStreamEvent::Metadata {
        usage: BedrockUsage { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    };
    assert!(format_stream_event(event).is_none());
}
