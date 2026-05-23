use serde::de::DeserializeOwned;

use crate::error::KovaError;

/// Represents a parsed SSE (Server-Sent Events) line.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SseLine {
    /// A `data:` line with the payload string (after stripping the prefix).
    Data(String),
    /// The `[DONE]` sentinel indicating end-of-stream.
    Done,
    /// An empty or whitespace-only line (SSE event separator).
    Empty,
    /// A comment line (starts with `:`).
    Comment,
}

/// Parse a single SSE line into an [`SseLine`] variant.
///
/// SSE protocol rules applied:
/// - Lines starting with `data: [DONE]` or `data:[DONE]` → `SseLine::Done`
/// - Lines starting with `data: ` or `data:` → `SseLine::Data` with the payload
/// - Lines starting with `:` → `SseLine::Comment`
/// - Empty / whitespace-only lines → `SseLine::Empty`
/// - Anything else → `SseLine::Empty` (ignored per SSE spec)
pub(crate) fn parse_sse_line(line: &str) -> SseLine {
    let trimmed = line.trim();

    if trimmed.is_empty() {
        return SseLine::Empty;
    }

    if trimmed.starts_with(':') {
        return SseLine::Comment;
    }

    if let Some(rest) = trimmed.strip_prefix("data:") {
        let payload = rest.trim_start();
        if payload == "[DONE]" {
            return SseLine::Done;
        }
        return SseLine::Data(payload.to_string());
    }

    // Lines that don't match any SSE field are ignored per the spec.
    SseLine::Empty
}

/// Deserialize a JSON string from an SSE `data:` payload into the target type.
///
/// Returns `KovaError::Stream` if the JSON is malformed or doesn't match `T`.
pub(crate) fn parse_sse_data<T: DeserializeOwned>(data: &str) -> Result<T, KovaError> {
    serde_json::from_str(data).map_err(|e| KovaError::Stream(format!("SSE parse error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde::{Deserialize, Serialize};

    use crate::models::{StopReason, StreamEvent};

    fn arb_stop_reason() -> impl Strategy<Value = StopReason> {
        prop_oneof![
            Just(StopReason::EndTurn),
            Just(StopReason::ToolUse),
            Just(StopReason::MaxTokens),
            "[a-z_]{3,15}".prop_map(StopReason::Unknown),
        ]
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
                }),
            arb_stop_reason().prop_map(|stop_reason| StreamEvent::StopEvent { stop_reason }),
            "[a-zA-Z0-9 ]{0,50}".prop_map(|message| StreamEvent::Error { message }),
        ]
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        #[test]
        fn prop_sse_line_parsing_roundtrip(event in arb_stream_event()) {
            let json = serde_json::to_string(&event).unwrap();
            let sse_line = format!("data: {}", json);
            let parsed = parse_sse_line(&sse_line);
            match parsed {
                SseLine::Data(payload) => {
                    let decoded: StreamEvent = parse_sse_data(&payload).unwrap();
                    prop_assert_eq!(&event, &decoded);
                }
                other => {
                    prop_assert!(false, "expected SseLine::Data, got {:?}", other);
                }
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestPayload {
        id: String,
        value: u32,
    }

    // ── parse_sse_line tests ───────────────────────────────────────

    #[test]
    fn test_data_line_with_space() {
        let result = parse_sse_line("data: {\"id\":\"1\"}");
        assert_eq!(result, SseLine::Data("{\"id\":\"1\"}".to_string()));
    }

    #[test]
    fn test_data_line_without_space() {
        let result = parse_sse_line("data:{\"id\":\"1\"}");
        assert_eq!(result, SseLine::Data("{\"id\":\"1\"}".to_string()));
    }

    #[test]
    fn test_done_sentinel_with_space() {
        assert_eq!(parse_sse_line("data: [DONE]"), SseLine::Done);
    }

    #[test]
    fn test_done_sentinel_without_space() {
        assert_eq!(parse_sse_line("data:[DONE]"), SseLine::Done);
    }

    #[test]
    fn test_empty_line() {
        assert_eq!(parse_sse_line(""), SseLine::Empty);
    }

    #[test]
    fn test_whitespace_only_line() {
        assert_eq!(parse_sse_line("   "), SseLine::Empty);
    }

    #[test]
    fn test_comment_line() {
        assert_eq!(parse_sse_line(": this is a comment"), SseLine::Comment);
    }

    #[test]
    fn test_unknown_field_treated_as_empty() {
        assert_eq!(parse_sse_line("event: message"), SseLine::Empty);
    }

    #[test]
    fn test_data_line_with_leading_whitespace() {
        let result = parse_sse_line("  data: hello");
        assert_eq!(result, SseLine::Data("hello".to_string()));
    }

    // ── parse_sse_data tests ───────────────────────────────────────

    #[test]
    fn test_parse_valid_json() {
        let data = r#"{"id":"abc","value":42}"#;
        let result: TestPayload = parse_sse_data(data).unwrap();
        assert_eq!(
            result,
            TestPayload {
                id: "abc".to_string(),
                value: 42,
            }
        );
    }

    #[test]
    fn test_parse_malformed_json_returns_stream_error() {
        let result: Result<TestPayload, _> = parse_sse_data("not json");
        let err = result.unwrap_err();
        match err {
            KovaError::Stream(msg) => {
                assert!(msg.contains("SSE parse error"));
            }
            other => panic!("Expected Stream error, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_empty_string_returns_stream_error() {
        let result: Result<TestPayload, _> = parse_sse_data("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_wrong_shape_returns_stream_error() {
        let result: Result<TestPayload, _> = parse_sse_data(r#"{"wrong":"shape"}"#);
        assert!(result.is_err());
    }
}
