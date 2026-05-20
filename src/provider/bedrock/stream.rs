use aws_smithy_types::event_stream::Message;
use serde::Deserialize;

use crate::error::KovaError;
use super::types::{
    BedrockContentBlockDelta, BedrockContentBlockStart, BedrockStreamEvent, BedrockUsage,
};

// ── Event stream intermediate payload types ────────────────────────
// Bedrock sends each event type as a separate JSON object keyed by the
// `:event-type` header, so we deserialize into these flat structs and
// convert to BedrockStreamEvent.

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockStartPayload {
    content_block_index: u32,
    start: BedrockContentBlockStart,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockDeltaPayload {
    content_block_index: u32,
    delta: BedrockContentBlockDelta,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockStopPayload {
    content_block_index: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageStopPayload {
    stop_reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MetadataPayload {
    usage: BedrockUsage,
}

/// Parse a single AWS event stream `Message` into a `BedrockStreamEvent`.
///
/// Returns `Ok(None)` for unknown event types, `Err` for malformed frames.
pub(super) fn parse_event_stream_frame(
    message: &Message,
) -> Result<Option<BedrockStreamEvent>, KovaError> {
    let event_type = message
        .headers()
        .iter()
        .find(|h| h.name().as_str() == ":event-type")
        .ok_or_else(|| {
            KovaError::Stream("Event stream frame missing :event-type header".to_string())
        })?;

    let event_type_str = event_type.value().as_string().map_err(|_| {
        KovaError::Stream("Event stream :event-type header is not a string".to_string())
    })?;

    let payload = message.payload();

    match event_type_str.as_str() {
        "contentBlockStart" => {
            let p: ContentBlockStartPayload =
                serde_json::from_slice(payload).map_err(|e| {
                    KovaError::Stream(format!(
                        "Failed to deserialize contentBlockStart payload: {e}"
                    ))
                })?;
            Ok(Some(BedrockStreamEvent::ContentBlockStart {
                content_block_index: p.content_block_index,
                start: p.start,
            }))
        }
        "contentBlockDelta" => {
            let p: ContentBlockDeltaPayload =
                serde_json::from_slice(payload).map_err(|e| {
                    KovaError::Stream(format!(
                        "Failed to deserialize contentBlockDelta payload: {e}"
                    ))
                })?;
            Ok(Some(BedrockStreamEvent::ContentBlockDelta {
                content_block_index: p.content_block_index,
                delta: p.delta,
            }))
        }
        "contentBlockStop" => {
            let p: ContentBlockStopPayload =
                serde_json::from_slice(payload).map_err(|e| {
                    KovaError::Stream(format!(
                        "Failed to deserialize contentBlockStop payload: {e}"
                    ))
                })?;
            Ok(Some(BedrockStreamEvent::ContentBlockStop {
                content_block_index: p.content_block_index,
            }))
        }
        "messageStop" => {
            let p: MessageStopPayload =
                serde_json::from_slice(payload).map_err(|e| {
                    KovaError::Stream(format!(
                        "Failed to deserialize messageStop payload: {e}"
                    ))
                })?;
            Ok(Some(BedrockStreamEvent::MessageStop {
                stop_reason: p.stop_reason,
            }))
        }
        "metadata" => {
            let p: MetadataPayload = serde_json::from_slice(payload).map_err(|e| {
                KovaError::Stream(format!("Failed to deserialize metadata payload: {e}"))
            })?;
            Ok(Some(BedrockStreamEvent::Metadata { usage: p.usage }))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_smithy_types::event_stream::{Header, HeaderValue};
    use serde_json::json;

    fn make_event_message(event_type: &str, payload: &[u8]) -> Message {
        Message::new_from_parts(
            vec![Header::new(
                ":event-type",
                HeaderValue::String(event_type.to_string().into()),
            )],
            bytes::Bytes::copy_from_slice(payload),
        )
    }

    #[test]
    fn test_parse_content_block_start_tool_use() {
        let payload = serde_json::to_vec(&json!({
            "contentBlockIndex": 0,
            "start": { "toolUse": { "toolUseId": "tool-1", "name": "search" } }
        }))
        .unwrap();
        let msg = make_event_message("contentBlockStart", &payload);
        let event = parse_event_stream_frame(&msg).unwrap().unwrap();
        match event {
            BedrockStreamEvent::ContentBlockStart {
                content_block_index,
                start,
            } => {
                assert_eq!(content_block_index, 0);
                match start {
                    BedrockContentBlockStart::ToolUse { tool_use_id, name } => {
                        assert_eq!(tool_use_id, "tool-1");
                        assert_eq!(name, "search");
                    }
                }
            }
            other => panic!("Expected ContentBlockStart, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_content_block_delta_text() {
        let payload = serde_json::to_vec(&json!({
            "contentBlockIndex": 1,
            "delta": { "text": "Hello" }
        }))
        .unwrap();
        let msg = make_event_message("contentBlockDelta", &payload);
        let event = parse_event_stream_frame(&msg).unwrap().unwrap();
        match event {
            BedrockStreamEvent::ContentBlockDelta {
                content_block_index,
                delta,
            } => {
                assert_eq!(content_block_index, 1);
                match delta {
                    BedrockContentBlockDelta::Text(text) => assert_eq!(text, "Hello"),
                    other => panic!("Expected Text delta, got: {:?}", other),
                }
            }
            other => panic!("Expected ContentBlockDelta, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_content_block_delta_tool_use() {
        let payload = serde_json::to_vec(&json!({
            "contentBlockIndex": 0,
            "delta": { "toolUse": { "input": "{\"query\":" } }
        }))
        .unwrap();
        let msg = make_event_message("contentBlockDelta", &payload);
        let event = parse_event_stream_frame(&msg).unwrap().unwrap();
        match event {
            BedrockStreamEvent::ContentBlockDelta {
                content_block_index,
                delta,
            } => {
                assert_eq!(content_block_index, 0);
                match delta {
                    BedrockContentBlockDelta::ToolUse { input } => {
                        assert_eq!(input, "{\"query\":");
                    }
                    other => panic!("Expected ToolUse delta, got: {:?}", other),
                }
            }
            other => panic!("Expected ContentBlockDelta, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_content_block_stop() {
        let payload = serde_json::to_vec(&json!({ "contentBlockIndex": 2 })).unwrap();
        let msg = make_event_message("contentBlockStop", &payload);
        let event = parse_event_stream_frame(&msg).unwrap().unwrap();
        match event {
            BedrockStreamEvent::ContentBlockStop { content_block_index } => {
                assert_eq!(content_block_index, 2)
            }
            other => panic!("Expected ContentBlockStop, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_message_stop() {
        let payload = serde_json::to_vec(&json!({ "stopReason": "end_turn" })).unwrap();
        let msg = make_event_message("messageStop", &payload);
        let event = parse_event_stream_frame(&msg).unwrap().unwrap();
        match event {
            BedrockStreamEvent::MessageStop { stop_reason } => {
                assert_eq!(stop_reason, "end_turn");
            }
            other => panic!("Expected MessageStop, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_metadata() {
        let payload = serde_json::to_vec(&json!({
            "usage": { "inputTokens": 10, "outputTokens": 5, "totalTokens": 15 }
        }))
        .unwrap();
        let msg = make_event_message("metadata", &payload);
        let event = parse_event_stream_frame(&msg).unwrap().unwrap();
        match event {
            BedrockStreamEvent::Metadata { usage } => {
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
                assert_eq!(usage.total_tokens, 15);
            }
            other => panic!("Expected Metadata, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_unknown_event_type_returns_none() {
        let msg = make_event_message("someUnknownEvent", b"{}");
        let result = parse_event_stream_frame(&msg).unwrap();
        assert!(result.is_none(), "Unknown event types should return None");
    }

    #[test]
    fn test_parse_missing_event_type_header_returns_stream_error() {
        let msg = Message::new(bytes::Bytes::from_static(b"{}"));
        let result = parse_event_stream_frame(&msg);
        match result {
            Err(KovaError::Stream(msg)) => {
                assert!(msg.contains("missing :event-type header"));
            }
            other => panic!("Expected KovaError::Stream, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_malformed_json_payload_returns_stream_error() {
        let msg = make_event_message("contentBlockDelta", b"not valid json");
        let result = parse_event_stream_frame(&msg);
        match result {
            Err(KovaError::Stream(msg)) => {
                assert!(msg.contains("Failed to deserialize"));
            }
            other => panic!("Expected KovaError::Stream, got: {:?}", other),
        }
    }
}
