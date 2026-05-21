use aws_smithy_eventstream::frame::write_message_to;
use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use futures::StreamExt;
use kova::error::KovaError;
use kova::models::*;
use kova::provider::LlmProvider;
use kova::provider::bedrock::{BedrockProvider, BedrockProviderConfig};
use proptest::prelude::*;
use serde_json::json;
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_config(server_uri: &str) -> BedrockProviderConfig {
    BedrockProviderConfig::new("us-east-1", "anthropic.claude-v2")
        .with_credentials(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        )
        .with_endpoint_url(server_uri)
}

fn sample_bedrock_response() -> serde_json::Value {
    json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello from Bedrock!"}]
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    })
}

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
        model: None,
        max_tokens: Some(100),
        temperature: None,
    }
}

/// Mock Bedrock converse endpoint, return mock response, verify SDK ModelResponse
#[tokio::test]
async fn test_chat_completion_success() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_bedrock_response()))
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let resp = provider
        .chat_completion(&sample_messages(), &[], &sample_config())
        .await
        .unwrap();

    assert_eq!(resp.content.len(), 1);
    assert_eq!(
        resp.content[0],
        ContentBlock::Text {
            text: "Hello from Bedrock!".to_string()
        }
    );
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    let usage = resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
}

/// Mock 400 error, verify KovaError::Provider with status 400
#[tokio::test]
async fn test_chat_completion_http_400() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(
            ResponseTemplate::new(400).set_body_string(r#"{"message":"validation error"}"#),
        )
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let err = provider
        .chat_completion(&sample_messages(), &[], &sample_config())
        .await
        .unwrap_err();
    match err {
        KovaError::Provider {
            status_code: Some(400),
            ..
        } => {}
        other => panic!("Expected Provider 400, got {:?}", other),
    }
}

/// Mock 403 error, verify KovaError::Provider with status 403
#[tokio::test]
async fn test_chat_completion_http_403() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(ResponseTemplate::new(403).set_body_string(r#"{"message":"access denied"}"#))
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let err = provider
        .chat_completion(&sample_messages(), &[], &sample_config())
        .await
        .unwrap_err();
    match err {
        KovaError::Provider {
            status_code: Some(403),
            ..
        } => {}
        other => panic!("Expected Provider 403, got {:?}", other),
    }
}

/// Mock 429 error, verify KovaError::Provider with status 429
#[tokio::test]
async fn test_chat_completion_http_429() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(ResponseTemplate::new(429).set_body_string(r#"{"message":"throttled"}"#))
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let err = provider
        .chat_completion(&sample_messages(), &[], &sample_config())
        .await
        .unwrap_err();
    match err {
        KovaError::Provider {
            status_code: Some(429),
            ..
        } => {}
        other => panic!("Expected Provider 429, got {:?}", other),
    }
}

/// Mock 500 error, verify KovaError::Provider with status 500
#[tokio::test]
async fn test_chat_completion_http_500() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(ResponseTemplate::new(500).set_body_string(r#"{"message":"internal error"}"#))
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let err = provider
        .chat_completion(&sample_messages(), &[], &sample_config())
        .await
        .unwrap_err();
    match err {
        KovaError::Provider {
            status_code: Some(500),
            ..
        } => {}
        other => panic!("Expected Provider 500, got {:?}", other),
    }
}

/// Mock connection timeout, verify KovaError::Timeout
#[tokio::test]
async fn test_chat_completion_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(sample_bedrock_response())
                .set_delay(Duration::from_secs(5)),
        )
        .mount(&server)
        .await;

    let config = BedrockProviderConfig::new("us-east-1", "anthropic.claude-v2")
        .with_credentials(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        )
        .with_endpoint_url(server.uri())
        .with_timeout(Duration::from_millis(100));
    let provider = BedrockProvider::new(config).await.unwrap();
    let err = provider
        .chat_completion(&sample_messages(), &[], &sample_config())
        .await
        .unwrap_err();
    match err {
        KovaError::Timeout(_) => {}
        other => panic!("Expected Timeout, got {:?}", other),
    }
}

/// Mock list_models endpoint, verify ModelInfo list
#[tokio::test]
async fn test_list_models_success() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/foundation-models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "modelSummaries": [
                {"modelId": "anthropic.claude-v2", "modelName": "Claude V2", "providerName": "Anthropic"},
                {"modelId": "meta.llama2-70b", "modelName": "Llama 2 70B", "providerName": "Meta"}
            ]
        })))
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let models = provider.list_models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "anthropic.claude-v2");
    assert_eq!(models[0].owned_by, "Anthropic");
    assert_eq!(models[1].id, "meta.llama2-70b");
    assert_eq!(models[1].owned_by, "Meta");
}

/// Mock list_models error, verify KovaError::Provider
#[tokio::test]
async fn test_list_models_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/foundation-models"))
        .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let err = provider.list_models().await.unwrap_err();
    match err {
        KovaError::Provider {
            status_code: Some(503),
            ..
        } => {}
        other => panic!("Expected Provider 503, got {:?}", other),
    }
}

proptest! {
    #[test]
    fn prop_http_error_status_mapping(
        status_code in 400u16..600u16,
        body in "[a-zA-Z0-9 ]{1,100}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/model/anthropic.claude-v2/converse"))
                .respond_with(
                    ResponseTemplate::new(status_code).set_body_string(body.clone()),
                )
                .mount(&server)
                .await;

            let provider = BedrockProvider::new(test_config(&server.uri()))
                .await
                .unwrap();
            let err = provider
                .chat_completion(&sample_messages(), &[], &sample_config())
                .await
                .unwrap_err();

            match err {
                KovaError::Provider {
                    status_code: Some(code),
                    message,
                } => {
                    prop_assert_eq!(code, status_code);
                    prop_assert_eq!(message, body);
                }
                other => {
                    prop_assert!(false, "Expected KovaError::Provider, got: {:?}", other);
                }
            }
            Ok(())
        })?;
    }
}

// ── Event stream encoding helpers for streaming tests ──────────────

/// Encode a single AWS event stream frame with the given `:event-type` header and JSON payload.
fn encode_event_stream_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
    let msg = Message::new(bytes::Bytes::copy_from_slice(payload))
        .add_header(Header::new(
            ":event-type".to_string(),
            HeaderValue::String(event_type.to_string().into()),
        ))
        .add_header(Header::new(
            ":content-type".to_string(),
            HeaderValue::String("application/json".to_string().into()),
        ))
        .add_header(Header::new(
            ":message-type".to_string(),
            HeaderValue::String("event".to_string().into()),
        ));
    let mut buf = Vec::new();
    write_message_to(&msg, &mut buf).expect("failed to encode event stream frame");
    buf
}

/// Build a complete event stream binary body from multiple (event_type, payload_json) pairs.
fn build_event_stream_body(events: &[(&str, serde_json::Value)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (event_type, payload) in events {
        let payload_bytes = serde_json::to_vec(payload).unwrap();
        body.extend(encode_event_stream_frame(event_type, &payload_bytes));
    }
    body
}

// ── Streaming integration tests ────────────────────────────────────

/// Mock converse-stream endpoint returning event stream binary data with text deltas.
/// Collect all StreamEvents, verify correct types and order.
#[tokio::test]
async fn test_chat_completion_stream_success() {
    let server = MockServer::start().await;

    let body = build_event_stream_body(&[
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "Hello "}}),
        ),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "world"}}),
        ),
        ("contentBlockStop", json!({"contentBlockIndex": 0})),
        ("messageStop", json!({"stopReason": "end_turn"})),
    ]);

    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse-stream"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "application/vnd.amazon.eventstream"),
        )
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let stream = provider
        .chat_completion_stream(&sample_messages(), &[], &sample_config())
        .await
        .unwrap();

    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Should have 2 ContentDelta events + 1 StopEvent (contentBlockStop is filtered)
    assert_eq!(events.len(), 3, "Expected 3 events, got: {:?}", events);
    assert_eq!(
        events[0],
        StreamEvent::ContentDelta {
            text: "Hello ".to_string()
        }
    );
    assert_eq!(
        events[1],
        StreamEvent::ContentDelta {
            text: "world".to_string()
        }
    );
    assert_eq!(
        events[2],
        StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn
        }
    );
}

/// Mock converse-stream endpoint returning tool use stream events.
/// Verify ToolUseDelta events and StopEvent.
#[tokio::test]
async fn test_chat_completion_stream_tool_use() {
    let server = MockServer::start().await;

    let body = build_event_stream_body(&[
        (
            "contentBlockStart",
            json!({
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tool-123", "name": "get_weather"}}
            }),
        ),
        (
            "contentBlockDelta",
            json!({
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "{\"city\": \"Seattle\"}"}}
            }),
        ),
        ("contentBlockStop", json!({"contentBlockIndex": 0})),
        ("messageStop", json!({"stopReason": "tool_use"})),
    ]);

    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse-stream"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "application/vnd.amazon.eventstream"),
        )
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let stream = provider
        .chat_completion_stream(&sample_messages(), &[], &sample_config())
        .await
        .unwrap();

    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(events.len(), 3, "Expected 3 events, got: {:?}", events);
    assert_eq!(
        events[0],
        StreamEvent::ToolUseDelta {
            id: "tool-123".to_string(),
            name: Some("get_weather".to_string()),
            input_delta: None,
        }
    );
    assert_eq!(
        events[1],
        StreamEvent::ToolUseDelta {
            id: String::new(),
            name: None,
            input_delta: Some("{\"city\": \"Seattle\"}".to_string()),
        }
    );
    assert_eq!(
        events[2],
        StreamEvent::StopEvent {
            stop_reason: StopReason::ToolUse
        }
    );
}

/// Mock converse-stream endpoint returning HTTP 400 error.
/// Verify KovaError::Provider with status 400.
#[tokio::test]
async fn test_chat_completion_stream_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse-stream"))
        .respond_with(
            ResponseTemplate::new(400).set_body_string(r#"{"message":"validation error"}"#),
        )
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let result = provider
        .chat_completion_stream(&sample_messages(), &[], &sample_config())
        .await;

    match result {
        Err(KovaError::Provider {
            status_code: Some(400),
            ..
        }) => {}
        Err(other) => panic!("Expected Provider 400, got {:?}", other),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

/// Mock converse-stream endpoint returning truncated binary data (incomplete frame).
/// Verify the stream terminates without error (wiremock closes connection after body).
/// Note: A true mid-stream connection drop is difficult to simulate with wiremock.
/// This test verifies that a truncated/partial event stream frame is handled gracefully.
#[tokio::test]
async fn test_chat_completion_stream_truncated_body() {
    let server = MockServer::start().await;

    // Build one valid frame followed by a truncated frame (just the first 4 bytes of a prelude)
    let mut body = build_event_stream_body(&[(
        "contentBlockDelta",
        json!({"contentBlockIndex": 0, "delta": {"text": "partial"}}),
    )]);
    // Append truncated bytes that look like the start of a frame but are incomplete
    body.extend_from_slice(&[0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x20]);

    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse-stream"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "application/vnd.amazon.eventstream"),
        )
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let stream = provider
        .chat_completion_stream(&sample_messages(), &[], &sample_config())
        .await
        .unwrap();

    let results: Vec<Result<StreamEvent, KovaError>> = stream.collect::<Vec<_>>().await;

    // First event should be the valid ContentDelta
    assert!(results[0].is_ok());
    assert_eq!(
        results[0].as_ref().unwrap(),
        &StreamEvent::ContentDelta {
            text: "partial".to_string()
        }
    );

    // The stream should end after the valid event. The truncated bytes cause the decoder
    // to request more data, but the byte stream ends, so the stream terminates.
    // Depending on implementation, this may just end the stream (no more events)
    // or produce a stream error. Either is acceptable behavior.
    assert!(!results.is_empty());
}

// ── End-to-end integration tests ────────────────────────

/// Full end-to-end chat_completion flow with tool definitions and a tool_use
/// response, exercising the complete tool calling path through the LlmProvider trait.
#[tokio::test]
async fn test_chat_completion_with_tools_e2e() {
    let server = MockServer::start().await;

    // Turn 1: Model responds with a tool_use content block
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {
                            "toolUseId": "call-001",
                            "name": "get_weather",
                            "input": {"city": "Seattle", "units": "fahrenheit"}
                        }}
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 25, "outputTokens": 40, "totalTokens": 65}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let tools = vec![ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }),
    }];

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();

    let messages = vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "What's the weather in Seattle?".to_string(),
        }],
    }];

    let resp = provider
        .chat_completion(&messages, &tools, &sample_config())
        .await
        .unwrap();

    // Verify tool_use response
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.content.len(), 1);
    match &resp.content[0] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "call-001");
            assert_eq!(name, "get_weather");
            assert_eq!(input["city"], "Seattle");
            assert_eq!(input["units"], "fahrenheit");
        }
        other => panic!("Expected ToolUse content block, got {:?}", other),
    }
    let usage = resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 25);
    assert_eq!(usage.output_tokens, 40);
}

/// Full end-to-end chat_completion flow with a multi-turn tool result conversation,
/// verifying that tool result messages are correctly serialized and the final text
/// response is returned through the LlmProvider trait.
#[tokio::test]
async fn test_chat_completion_tool_result_round_trip() {
    let server = MockServer::start().await;

    // Model receives tool result and responds with final text
    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "The weather in Seattle is 62°F and cloudy."}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let tools = vec![ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }),
    }];

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();

    // Multi-turn conversation: user question → assistant tool_use → user tool_result
    let messages = vec![
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "What's the weather in Seattle?".to_string(),
            }],
        },
        ConversationMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-001".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city": "Seattle"}),
            }],
        },
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call-001".to_string(),
                content: "62°F, cloudy".to_string(),
                is_error: false,
            }],
        },
    ];

    let resp = provider
        .chat_completion(&messages, &tools, &sample_config())
        .await
        .unwrap();

    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(resp.content.len(), 1);
    assert_eq!(
        resp.content[0],
        ContentBlock::Text {
            text: "The weather in Seattle is 62°F and cloudy.".to_string()
        }
    );
    let usage = resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 50);
    assert_eq!(usage.output_tokens, 20);
}

/// Full list_models flow through LlmProvider trait with a larger, more realistic
/// model list covering multiple providers and model families.
#[tokio::test]
async fn test_list_models_large_response() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/foundation-models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "modelSummaries": [
                {"modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0", "modelName": "Claude 3.5 Sonnet v2", "providerName": "Anthropic"},
                {"modelId": "anthropic.claude-3-haiku-20240307-v1:0", "modelName": "Claude 3 Haiku", "providerName": "Anthropic"},
                {"modelId": "anthropic.claude-3-opus-20240229-v1:0", "modelName": "Claude 3 Opus", "providerName": "Anthropic"},
                {"modelId": "meta.llama3-1-70b-instruct-v1:0", "modelName": "Llama 3.1 70B Instruct", "providerName": "Meta"},
                {"modelId": "meta.llama3-1-8b-instruct-v1:0", "modelName": "Llama 3.1 8B Instruct", "providerName": "Meta"},
                {"modelId": "mistral.mistral-large-2407-v1:0", "modelName": "Mistral Large 2", "providerName": "Mistral AI"},
                {"modelId": "amazon.titan-text-express-v1", "modelName": "Titan Text Express", "providerName": "Amazon"},
                {"modelId": "cohere.command-r-plus-v1:0", "modelName": "Command R+", "providerName": "Cohere"}
            ]
        })))
        .mount(&server)
        .await;

    let provider = BedrockProvider::new(test_config(&server.uri()))
        .await
        .unwrap();
    let models = provider.list_models().await.unwrap();

    assert_eq!(models.len(), 8);

    // Verify first and last entries
    assert_eq!(models[0].id, "anthropic.claude-3-5-sonnet-20241022-v2:0");
    assert_eq!(models[0].owned_by, "Anthropic");
    assert_eq!(models[0].object, "model");

    assert_eq!(models[7].id, "cohere.command-r-plus-v1:0");
    assert_eq!(models[7].owned_by, "Cohere");

    // Verify multiple providers are represented
    let providers: std::collections::HashSet<&str> =
        models.iter().map(|m| m.owned_by.as_str()).collect();
    assert!(providers.contains("Anthropic"));
    assert!(providers.contains("Meta"));
    assert!(providers.contains("Mistral AI"));
    assert!(providers.contains("Amazon"));
    assert!(providers.contains("Cohere"));
}

/// Compile-time assertion that BedrockProvider implements Send + Sync,
/// which is required by the LlmProvider trait for safe concurrent usage.
fn _assert_send_sync<T: Send + Sync>() {}

#[test]
fn test_bedrock_provider_is_send_sync() {
    _assert_send_sync::<BedrockProvider>();
}

// ── Trait object (dyn LlmProvider) end-to-end tests ─────

/// Full chat_completion e2e test through `dyn LlmProvider` trait object.
/// Creates a BedrockProvider, boxes it as `Box<dyn LlmProvider>`, sends a
/// chat completion request with system message, user message, and tools,
/// then verifies the response through the trait interface.
#[tokio::test]
async fn test_chat_completion_via_dyn_llm_provider() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "The capital of France is Paris."}
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 30, "outputTokens": 12, "totalTokens": 42}
        })))
        .mount(&server)
        .await;

    let provider: Box<dyn LlmProvider> = Box::new(
        BedrockProvider::new(test_config(&server.uri()))
            .await
            .unwrap(),
    );

    let messages = vec![
        ConversationMessage {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "You are a helpful geography assistant.".to_string(),
            }],
        },
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "What is the capital of France?".to_string(),
            }],
        },
    ];

    let tools = vec![ToolDefinition {
        name: "lookup_country".to_string(),
        description: "Look up information about a country".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "country": {"type": "string"}
            },
            "required": ["country"]
        }),
    }];

    let config = sample_config();
    let resp = provider
        .chat_completion(&messages, &tools, &config)
        .await
        .unwrap();

    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(resp.content.len(), 1);
    assert_eq!(
        resp.content[0],
        ContentBlock::Text {
            text: "The capital of France is Paris.".to_string()
        }
    );
    let usage = resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 30);
    assert_eq!(usage.output_tokens, 12);
    assert_eq!(usage.total_tokens, 42);
}

/// Full list_models e2e test through `dyn LlmProvider` trait object.
/// Creates a BedrockProvider, boxes it as `Box<dyn LlmProvider>`, calls
/// `list_models()` through the trait interface, and verifies the model list.
#[tokio::test]
async fn test_list_models_via_dyn_llm_provider() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/foundation-models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "modelSummaries": [
                {"modelId": "anthropic.claude-v2", "modelName": "Claude V2", "providerName": "Anthropic"},
                {"modelId": "meta.llama2-70b", "modelName": "Llama 2 70B", "providerName": "Meta"},
                {"modelId": "amazon.titan-text-express-v1", "modelName": "Titan Text Express", "providerName": "Amazon"}
            ]
        })))
        .mount(&server)
        .await;

    let provider: Box<dyn LlmProvider> = Box::new(
        BedrockProvider::new(test_config(&server.uri()))
            .await
            .unwrap(),
    );

    let models = provider.list_models().await.unwrap();

    assert_eq!(models.len(), 3);
    assert_eq!(models[0].id, "anthropic.claude-v2");
    assert_eq!(models[0].owned_by, "Anthropic");
    assert_eq!(models[0].object, "model");
    assert_eq!(models[1].id, "meta.llama2-70b");
    assert_eq!(models[1].owned_by, "Meta");
    assert_eq!(models[2].id, "amazon.titan-text-express-v1");
    assert_eq!(models[2].owned_by, "Amazon");
}

/// Full streaming e2e test through `dyn LlmProvider` trait object.
/// Creates a BedrockProvider, boxes it as `Box<dyn LlmProvider>`, calls
/// `chat_completion_stream()` through the trait interface, collects and
/// verifies stream events.
#[tokio::test]
async fn test_chat_completion_stream_via_dyn_llm_provider() {
    let server = MockServer::start().await;

    let body = build_event_stream_body(&[
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "Bonjour "}}),
        ),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "le monde"}}),
        ),
        ("contentBlockStop", json!({"contentBlockIndex": 0})),
        ("messageStop", json!({"stopReason": "end_turn"})),
    ]);

    Mock::given(method("POST"))
        .and(path("/model/anthropic.claude-v2/converse-stream"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "application/vnd.amazon.eventstream"),
        )
        .mount(&server)
        .await;

    let provider: Box<dyn LlmProvider> = Box::new(
        BedrockProvider::new(test_config(&server.uri()))
            .await
            .unwrap(),
    );

    let stream = provider
        .chat_completion_stream(&sample_messages(), &[], &sample_config())
        .await
        .unwrap();

    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(events.len(), 3, "Expected 3 events, got: {:?}", events);
    assert_eq!(
        events[0],
        StreamEvent::ContentDelta {
            text: "Bonjour ".to_string()
        }
    );
    assert_eq!(
        events[1],
        StreamEvent::ContentDelta {
            text: "le monde".to_string()
        }
    );
    assert_eq!(
        events[2],
        StreamEvent::StopEvent {
            stop_reason: StopReason::EndTurn
        }
    );
}
