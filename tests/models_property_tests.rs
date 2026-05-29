use proptest::prelude::*;
use std::time::Duration;

use kova_sdk::error::KovaError;
use kova_sdk::models::*;

fn arb_role() -> impl Strategy<Value = Role> {
    prop_oneof![
        Just(Role::System),
        Just(Role::User),
        Just(Role::Assistant),
        Just(Role::Tool),
    ]
}

fn arb_stop_reason() -> impl Strategy<Value = StopReason> {
    prop_oneof![
        Just(StopReason::EndTurn),
        Just(StopReason::ToolUse),
        Just(StopReason::MaxTokens),
        "[a-z_]{3,15}".prop_map(StopReason::Unknown),
    ]
}

fn arb_content_block() -> impl Strategy<Value = ContentBlock> {
    prop_oneof![
        "[a-zA-Z0-9 .,!?]{0,100}".prop_map(|text| ContentBlock::Text { text }),
        (
            "[a-z0-9_]{1,20}",
            "[a-z_]{1,20}",
            Just(serde_json::json!({"key": "value"})),
        )
            .prop_map(|(id, name, input)| ContentBlock::ToolUse { id, name, input, provider_metadata: None }),
        ("[a-z0-9_]{1,20}", "[a-zA-Z0-9 ]{0,50}", any::<bool>(),).prop_map(
            |(tool_use_id, content, is_error)| ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            }
        ),
    ]
}

fn arb_conversation_message() -> impl Strategy<Value = ConversationMessage> {
    (
        arb_role(),
        proptest::collection::vec(arb_content_block(), 1..4),
    )
        .prop_map(|(role, content)| ConversationMessage { role, content })
}

fn arb_usage_stats() -> impl Strategy<Value = UsageStats> {
    (any::<u32>(), any::<u32>()).prop_map(|(input_tokens, output_tokens)| UsageStats {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens.saturating_add(output_tokens),
    })
}

fn arb_model_response() -> impl Strategy<Value = ModelResponse> {
    (
        proptest::collection::vec(arb_content_block(), 1..4),
        arb_stop_reason(),
        proptest::option::of(arb_usage_stats()),
    )
        .prop_map(|(content, stop_reason, usage)| ModelResponse {
            content,
            stop_reason,
            usage,
        })
}

fn arb_inference_config() -> impl Strategy<Value = InferenceConfig> {
    (
        proptest::option::of("[a-z0-9-]{1,20}"),
        proptest::option::of(1..4096u32),
        proptest::option::of(0.0f32..2.0f32),
    )
        .prop_map(|(model, max_tokens, temperature)| InferenceConfig {
            model,
            max_tokens,
            temperature,
        })
}

fn arb_tool_definition() -> impl Strategy<Value = ToolDefinition> {
    ("[a-z_]{1,20}", "[a-zA-Z0-9 ]{0,50}").prop_map(|(name, description)| ToolDefinition {
        name,
        description,
        parameters: serde_json::json!({"type": "object"}),
    })
}

fn arb_stream_event() -> impl Strategy<Value = StreamEvent> {
    prop_oneof![
        "[a-zA-Z0-9 ]{0,50}".prop_map(|text| StreamEvent::ContentDelta { text }),
        (
            "[a-z0-9_]{1,20}",
            proptest::option::of("[a-z_]{1,20}"),
            proptest::option::of("[a-zA-Z0-9 {}]{0,30}"),
        )
            .prop_map(|(id, name, input_delta)| StreamEvent::ToolUseDelta {
                id,
                name,
                input_delta,
                provider_metadata: None,
            }),
        arb_stop_reason().prop_map(|stop_reason| StreamEvent::StopEvent { stop_reason }),
        "[a-zA-Z0-9 ]{0,50}".prop_map(|message| StreamEvent::Error { message }),
    ]
}

fn arb_tool_result() -> impl Strategy<Value = ToolResult> {
    ("[a-zA-Z0-9 ]{0,50}", any::<bool>())
        .prop_map(|(content, is_error)| ToolResult { content, is_error })
}

fn arb_model_info() -> impl Strategy<Value = ModelInfo> {
    (
        "[a-z0-9-]{1,20}",
        Just("model".to_string()),
        any::<u64>(),
        "[a-z0-9-]{1,20}",
    )
        .prop_map(|(id, object, created, owned_by)| ModelInfo {
            id,
            object,
            created,
            owned_by,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_serde_roundtrip_conversation_message(msg in arb_conversation_message()) {
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: ConversationMessage = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&msg, &decoded);
    }

    #[test]
    fn prop_serde_roundtrip_model_response(resp in arb_model_response()) {
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: ModelResponse = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&resp, &decoded);
    }

    #[test]
    fn prop_serde_roundtrip_tool_result(tr in arb_tool_result()) {
        let json = serde_json::to_string(&tr).unwrap();
        let decoded: ToolResult = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&tr, &decoded);
    }

    #[test]
    fn prop_serde_roundtrip_tool_definition(td in arb_tool_definition()) {
        let json = serde_json::to_string(&td).unwrap();
        let decoded: ToolDefinition = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&td, &decoded);
    }

    #[test]
    fn prop_serde_roundtrip_inference_config(cfg in arb_inference_config()) {
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: InferenceConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&cfg, &decoded);
    }

    #[test]
    fn prop_serde_roundtrip_stream_event(evt in arb_stream_event()) {
        let json = serde_json::to_string(&evt).unwrap();
        let decoded: StreamEvent = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&evt, &decoded);
    }

    #[test]
    fn prop_serde_roundtrip_model_info(info in arb_model_info()) {
        let json = serde_json::to_string(&info).unwrap();
        let decoded: ModelInfo = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&info, &decoded);
    }
}

#[test]
fn prop_error_display_non_empty() {
    let variants: Vec<KovaError> = vec![
        KovaError::Provider {
            message: "test".into(),
            status_code: Some(500),
        },
        KovaError::Provider {
            message: "test".into(),
            status_code: None,
        },
        KovaError::Connection("timeout".into()),
        KovaError::ToolExecution {
            tool_name: "calc".into(),
            message: "failed".into(),
        },
        KovaError::ToolNotFound("missing".into()),
        KovaError::Mcp("connection refused".into()),
        KovaError::Memory("full".into()),
        KovaError::Orchestration("timeout".into()),
        KovaError::Build("missing provider".into()),
        KovaError::Stream("eof".into()),
        KovaError::MaxIterations(10),
        KovaError::Timeout(Duration::from_secs(30)),
    ];

    for err in &variants {
        let display = err.to_string();
        assert!(
            !display.is_empty(),
            "KovaError variant {:?} produced empty Display string",
            err
        );
    }
}
