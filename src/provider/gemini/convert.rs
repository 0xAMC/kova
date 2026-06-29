use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_tool_id() -> String {
    let n = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tooluse_{n:016x}")
}

use futures::Stream;

use super::types::{
    GeminiCandidate, GeminiContent, GeminiFunctionCall, GeminiFunctionDeclaration,
    GeminiFunctionResponse, GeminiGenerationConfig, GeminiPart, GeminiRequest, GeminiResponse,
    GeminiThinkingConfig, GeminiTool,
};
use crate::error::KovaError;
use crate::models::{
    ContentBlock, ConversationMessage, InferenceConfig, ModelResponse, Role, StopReason,
    StreamEvent, ToolDefinition, UsageStats,
};
use crate::streaming::line_stream::{LineOutcome, line_stream_to_events};
use crate::streaming::sse::{SseLine, parse_sse_data, parse_sse_line};

// ── Tool-parameter schema sanitisation ─────────────────────────────
//
// Gemini's function-calling API only accepts a subset of JSON Schema
// (an OpenAPI 3.0 `Schema`). Draft keywords like `$schema`,
// `exclusiveMinimum`/`exclusiveMaximum`, `additionalProperties`, `$ref`,
// etc. are rejected with HTTP 400 "Unknown name ...". Tool schemas may
// carry these (e.g. from `schemars`), so we recursively keep only the
// fields Gemini documents and recurse into nested schemas.
const GEMINI_SCHEMA_KEYS: &[&str] = &[
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "pattern",
    "example",
    "default",
    "anyOf",
    "propertyOrdering",
];

fn sanitize_schema_for_gemini(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, val) in map {
                if !GEMINI_SCHEMA_KEYS.contains(&key.as_str()) {
                    continue;
                }
                let sanitized = match key.as_str() {
                    // `properties` is a map of name → schema.
                    "properties" => match val {
                        serde_json::Value::Object(props) => serde_json::Value::Object(
                            props
                                .iter()
                                .map(|(k, v)| (k.clone(), sanitize_schema_for_gemini(v)))
                                .collect(),
                        ),
                        other => other.clone(),
                    },
                    // `items` is a single nested schema.
                    "items" => sanitize_schema_for_gemini(val),
                    // `anyOf` is an array of nested schemas.
                    "anyOf" => match val {
                        serde_json::Value::Array(variants) => serde_json::Value::Array(
                            variants.iter().map(sanitize_schema_for_gemini).collect(),
                        ),
                        other => other.clone(),
                    },
                    // Remaining keys (enum, required, default, …) are leaf data.
                    _ => val.clone(),
                };
                out.insert(key.clone(), sanitized);
            }
            serde_json::Value::Object(out)
        }
        other => other.clone(),
    }
}

// ── Tool-ID → name lookup ──────────────────────────────────────────
//
// Gemini's FunctionResponse requires the function name, but kova's
// ToolResult only stores tool_use_id. We build this map once per
// request by scanning all ToolUse blocks in the message history.

fn build_tool_id_map(messages: &[ConversationMessage]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for msg in messages {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                map.insert(id.clone(), name.clone());
            }
        }
    }
    map
}

// ── Request formatting ─────────────────────────────────────────────

pub(crate) fn format_request(
    messages: &[ConversationMessage],
    tools: &[ToolDefinition],
    config: &InferenceConfig,
    thinking_budget: Option<i32>,
) -> GeminiRequest {
    let tool_id_map = build_tool_id_map(messages);

    let mut system_instruction: Option<GeminiContent> = None;
    let mut contents: Vec<GeminiContent> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                let parts: Vec<GeminiPart> = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(GeminiPart {
                                text: Some(text.clone()),
                                thought: None,
                                function_call: None,
                                function_response: None,
                                thought_signature: None,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if !parts.is_empty() {
                    // role is present in the Content struct but ignored by the
                    // API for systemInstruction; "user" keeps serde happy.
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
            }

            Role::Tool => {
                // kova Role::Tool messages carry ToolResult blocks.
                // Gemini models consume these as "user"-role FunctionResponse parts.
                let parts: Vec<GeminiPart> = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } = b
                        {
                            let name = tool_id_map
                                .get(tool_use_id)
                                .cloned()
                                .unwrap_or_else(|| "unknown".to_string());
                            let response = if *is_error {
                                serde_json::json!({ "error": content })
                            } else {
                                serde_json::json!({ "output": content })
                            };
                            Some(GeminiPart {
                                text: None,
                                thought: None,
                                function_call: None,
                                function_response: Some(GeminiFunctionResponse {
                                    name,
                                    id: Some(tool_use_id.clone()),
                                    response,
                                }),
                                thought_signature: None,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if !parts.is_empty() {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
            }

            Role::User => {
                let parts: Vec<GeminiPart> = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(GeminiPart {
                                text: Some(text.clone()),
                                thought: None,
                                function_call: None,
                                function_response: None,
                                thought_signature: None,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if !parts.is_empty() {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
            }

            Role::Assistant => {
                let mut parts: Vec<GeminiPart> = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => parts.push(GeminiPart {
                            text: Some(text.clone()),
                            thought: None,
                            function_call: None,
                            function_response: None,
                            thought_signature: None,
                        }),
                        ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            provider_metadata,
                        } => {
                            let thought_signature = provider_metadata
                                .as_ref()
                                .and_then(|m| m["thoughtSignature"].as_str())
                                .map(|s| s.to_string());
                            parts.push(GeminiPart {
                                text: None,
                                thought: None,
                                function_call: Some(GeminiFunctionCall {
                                    id: Some(id.clone()),
                                    name: name.clone(),
                                    args: input.clone(),
                                }),
                                function_response: None,
                                thought_signature,
                            })
                        }
                        ContentBlock::ToolResult { .. } => {}
                    }
                }
                if !parts.is_empty() {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
        }
    }

    let gemini_tools = if tools.is_empty() {
        None
    } else {
        Some(vec![GeminiTool {
            function_declarations: tools
                .iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: sanitize_schema_for_gemini(&t.parameters),
                })
                .collect(),
        }])
    };

    let thinking_config = thinking_budget.map(|budget| GeminiThinkingConfig {
        thinking_budget: budget,
        // budget == 0 disables thinking; any other value means we want thoughts visible.
        include_thoughts: if budget != 0 { Some(true) } else { None },
    });
    let generation_config = if config.max_tokens.is_some()
        || config.temperature.is_some()
        || config.top_p.is_some()
        || config.stop_sequences.is_some()
        || thinking_config.is_some()
    {
        Some(GeminiGenerationConfig {
            max_output_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            stop_sequences: config.stop_sequences.clone(),
            thinking_config,
        })
    } else {
        None
    };

    GeminiRequest {
        contents,
        system_instruction,
        tools: gemini_tools,
        generation_config,
    }
}

// ── Response formatting ────────────────────────────────────────────

fn candidate_stop_reason(candidate: &GeminiCandidate) -> StopReason {
    // Prefer content detection: if any part is a functionCall the model is
    // requesting tool invocation regardless of finishReason (which is often
    // "STOP" even when function calls are present).
    let has_function_calls = candidate
        .content
        .as_ref()
        .map(|c| c.parts.iter().any(|p| p.function_call.is_some()))
        .unwrap_or(false);
    if has_function_calls {
        return StopReason::ToolUse;
    }
    match candidate.finish_reason.as_deref() {
        Some("STOP") | None => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        Some(other) => StopReason::Unknown(other.to_string()),
    }
}

pub(crate) fn format_response(gemini_resp: GeminiResponse) -> Result<ModelResponse, KovaError> {
    let candidate =
        gemini_resp
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| KovaError::Provider {
                message: "No candidates in response".to_string(),
                status_code: None,
            })?;

    let stop_reason = candidate_stop_reason(&candidate);

    let mut content: Vec<ContentBlock> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    for part in candidate.content.map(|c| c.parts).unwrap_or_default() {
        if part.thought == Some(true) {
            if let Some(text) = part.text
                && !text.is_empty()
            {
                thinking_parts.push(text);
            }
            continue;
        }
        if let Some(text) = part.text
            && !text.is_empty()
        {
            content.push(ContentBlock::Text { text });
        }
        if let Some(fc) = part.function_call {
            let id = fc.id.unwrap_or_else(generate_tool_id);
            let provider_metadata = part
                .thought_signature
                .map(|sig| serde_json::json!({ "thoughtSignature": sig }));
            content.push(ContentBlock::ToolUse {
                id,
                name: fc.name,
                input: fc.args,
                provider_metadata,
            });
        }
    }

    let usage = gemini_resp.usage_metadata.map(|u| UsageStats {
        input_tokens: u.prompt_token_count,
        output_tokens: u.candidates_token_count,
        total_tokens: u.total_token_count,
        thinking_tokens: u.thoughts_token_count,
    });

    let thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join(""))
    };

    Ok(ModelResponse {
        content,
        stop_reason,
        usage,
        thinking,
    })
}

// ── Stream event formatting ────────────────────────────────────────

pub(crate) fn format_stream_events(chunk: GeminiResponse) -> Vec<StreamEvent> {
    let mut events: Vec<StreamEvent> = Vec::new();

    if let Some(usage) = chunk.usage_metadata {
        // candidates_token_count can be 0 in streaming even when tokens were used;
        // total - input is always reliable.
        let output_tokens = usage.candidates_token_count.max(
            usage
                .total_token_count
                .saturating_sub(usage.prompt_token_count),
        );
        events.push(StreamEvent::UsageEvent {
            input_tokens: usage.prompt_token_count,
            output_tokens,
            thinking_tokens: usage.thoughts_token_count,
        });
    }

    for candidate in chunk.candidates {
        let has_function_calls = candidate
            .content
            .as_ref()
            .map(|c| {
                c.parts
                    .iter()
                    .any(|p| p.thought != Some(true) && p.function_call.is_some())
            })
            .unwrap_or(false);

        if let Some(reason) = &candidate.finish_reason {
            let stop_reason = if has_function_calls {
                StopReason::ToolUse
            } else {
                match reason.as_str() {
                    "STOP" => StopReason::EndTurn,
                    "MAX_TOKENS" => StopReason::MaxTokens,
                    other => StopReason::Unknown(other.to_string()),
                }
            };
            events.push(StreamEvent::StopEvent { stop_reason });
        }

        for part in candidate.content.map(|c| c.parts).unwrap_or_default() {
            if part.thought == Some(true) {
                if let Some(text) = part.text
                    && !text.is_empty()
                {
                    tracing::debug!(chars = text.len(), "Gemini stream: thought part");
                    events.push(StreamEvent::ThinkingDelta { text });
                }
                continue;
            }
            if let Some(text) = part.text
                && !text.is_empty()
            {
                events.push(StreamEvent::ContentDelta { text });
            }
            if let Some(fc) = part.function_call {
                let id = fc.id.unwrap_or_else(generate_tool_id);
                let input_delta = serde_json::to_string(&fc.args).ok();
                let provider_metadata = part
                    .thought_signature
                    .map(|sig| serde_json::json!({ "thoughtSignature": sig }));
                events.push(StreamEvent::ToolUseDelta {
                    id,
                    name: Some(fc.name),
                    input_delta,
                    provider_metadata,
                    // Gemini sends each function call complete in one chunk.
                    index: None,
                });
            }
        }
    }

    events
}

// ── SSE byte stream adapter ────────────────────────────────────────
//
// Gemini's streamGenerateContent sends SSE-formatted chunks when
// `?alt=sse` is appended to the URL. Each `data:` payload is a full
// GenerateContentResponse JSON object. The stream closes at EOF
// without sending a `[DONE]` sentinel (unlike OpenAI).

pub(crate) fn sse_byte_stream_to_events(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>> {
    line_stream_to_events(byte_stream, |line| match parse_sse_line(line) {
        SseLine::Done => LineOutcome::Done,
        SseLine::Data(data) => match parse_sse_data::<GeminiResponse>(&data) {
            Ok(chunk) => LineOutcome::Events(format_stream_events(chunk)),
            Err(e) => LineOutcome::Fail(e),
        },
        SseLine::Empty | SseLine::Comment => LineOutcome::Events(Vec::new()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContentBlock, ConversationMessage, Role};
    use serde_json::json;

    fn user_msg(text: &str) -> ConversationMessage {
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn assistant_text_msg(text: &str) -> ConversationMessage {
        ConversationMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn system_msg(text: &str) -> ConversationMessage {
        ConversationMessage {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn default_config() -> InferenceConfig {
        InferenceConfig {
            model: Some("gemini-2.0-flash".to_string()),
            max_tokens: None,
            temperature: None,
            ..Default::default()
        }
    }

    // ── format_request ─────────────────────────────────────────────

    #[test]
    fn test_user_message_maps_to_user_role() {
        let messages = vec![user_msg("Hello")];
        let req = format_request(&messages, &[], &default_config(), None);
        assert_eq!(req.contents.len(), 1);
        assert_eq!(req.contents[0].role, "user");
        assert_eq!(req.contents[0].parts[0].text.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_assistant_message_maps_to_model_role() {
        let messages = vec![user_msg("Hi"), assistant_text_msg("Hello!")];
        let req = format_request(&messages, &[], &default_config(), None);
        assert_eq!(req.contents[1].role, "model");
    }

    #[test]
    fn test_system_message_becomes_system_instruction() {
        let messages = vec![system_msg("You are helpful."), user_msg("Hi")];
        let req = format_request(&messages, &[], &default_config(), None);
        // System message is NOT in contents
        assert_eq!(req.contents.len(), 1);
        assert!(req.system_instruction.is_some());
        let si = req.system_instruction.unwrap();
        assert_eq!(si.parts[0].text.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn test_tool_result_becomes_function_response_with_name_lookup() {
        let messages = vec![
            user_msg("What time is it?"),
            ConversationMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "get_time".to_string(),
                    input: json!({}),
                    provider_metadata: None,
                }],
            },
            ConversationMessage {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: "12:00 PM".to_string(),
                    is_error: false,
                }],
            },
        ];
        let req = format_request(&messages, &[], &default_config(), None);
        // Contents: user, model (tool call), user (function response)
        assert_eq!(req.contents.len(), 3);
        let fr_part = &req.contents[2].parts[0];
        let fr = fr_part.function_response.as_ref().unwrap();
        assert_eq!(fr.name, "get_time");
        assert_eq!(fr.id.as_deref(), Some("call-1"));
        assert_eq!(fr.response["output"], "12:00 PM");
    }

    #[test]
    fn test_error_tool_result_wraps_in_error_key() {
        let messages = vec![
            user_msg("Do something"),
            ConversationMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call-2".to_string(),
                    name: "do_thing".to_string(),
                    input: json!({}),
                    provider_metadata: None,
                }],
            },
            ConversationMessage {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-2".to_string(),
                    content: "boom".to_string(),
                    is_error: true,
                }],
            },
        ];
        let req = format_request(&messages, &[], &default_config(), None);
        let fr = req.contents[2].parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.response["error"], "boom");
    }

    #[test]
    fn test_tools_become_function_declarations() {
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }];
        let messages = vec![user_msg("Search for cats")];
        let req = format_request(&messages, &tools, &default_config(), None);
        let gemini_tools = req.tools.unwrap();
        assert_eq!(gemini_tools.len(), 1);
        assert_eq!(gemini_tools[0].function_declarations[0].name, "search");
    }

    #[test]
    fn test_tool_schema_strips_unsupported_fields() {
        let tools = vec![ToolDefinition {
            name: "calc".to_string(),
            description: "Calculate".to_string(),
            parameters: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "n": {
                        "type": "integer",
                        "exclusiveMinimum": 0,
                        "exclusiveMaximum": 100,
                        "description": "a bounded number"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string", "pattern": "^x" }
                    }
                },
                "required": ["n"]
            }),
        }];
        let messages = vec![user_msg("go")];
        let req = format_request(&messages, &tools, &default_config(), None);
        let params = &req.tools.unwrap()[0].function_declarations[0].parameters;

        assert!(params.get("$schema").is_none());
        assert!(params.get("additionalProperties").is_none());
        let n = &params["properties"]["n"];
        assert!(n.get("exclusiveMinimum").is_none());
        assert!(n.get("exclusiveMaximum").is_none());
        assert_eq!(n["type"], "integer");
        assert_eq!(n["description"], "a bounded number");
        assert_eq!(params["properties"]["tags"]["items"]["type"], "string");
        assert_eq!(params["required"][0], "n");
    }

    #[test]
    fn test_empty_tools_omits_tools_field() {
        let messages = vec![user_msg("Hi")];
        let req = format_request(&messages, &[], &default_config(), None);
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_generation_config_from_inference_config() {
        let config = InferenceConfig {
            model: Some("gemini-2.0-flash".to_string()),
            max_tokens: Some(512),
            temperature: Some(0.5),
            ..Default::default()
        };
        let req = format_request(&[user_msg("Hi")], &[], &config, None);
        let gc = req.generation_config.unwrap();
        assert_eq!(gc.max_output_tokens, Some(512));
        assert_eq!(gc.temperature, Some(0.5));
    }

    #[test]
    fn test_no_generation_config_when_defaults() {
        let req = format_request(&[user_msg("Hi")], &[], &default_config(), None);
        assert!(req.generation_config.is_none());
    }

    // ── format_response ────────────────────────────────────────────

    fn make_response(finish_reason: Option<&str>, parts: Vec<GeminiPart>) -> GeminiResponse {
        GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts,
                }),
                finish_reason: finish_reason.map(|s| s.to_string()),
            }],
            usage_metadata: None,
        }
    }

    #[test]
    fn test_text_response_maps_to_content_block_text() {
        let resp = make_response(
            Some("STOP"),
            vec![GeminiPart {
                text: Some("Hello world".to_string()),
                thought: None,
                function_call: None,
                function_response: None,
                thought_signature: None,
            }],
        );
        let model_resp = format_response(resp).unwrap();
        assert_eq!(model_resp.stop_reason, StopReason::EndTurn);
        assert!(
            matches!(&model_resp.content[0], ContentBlock::Text { text } if text == "Hello world")
        );
    }

    #[test]
    fn test_function_call_maps_to_tool_use_with_stop_reason() {
        let resp = make_response(
            Some("STOP"),
            vec![GeminiPart {
                text: None,
                thought: None,
                function_call: Some(GeminiFunctionCall {
                    id: Some("fc-123".to_string()),
                    name: "get_weather".to_string(),
                    args: json!({ "city": "London" }),
                }),
                function_response: None,
                thought_signature: None,
            }],
        );
        let model_resp = format_response(resp).unwrap();
        assert_eq!(model_resp.stop_reason, StopReason::ToolUse);
        match &model_resp.content[0] {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "fc-123");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "London");
            }
            other => panic!("Expected ToolUse, got {:?}", other),
        }
    }

    #[test]
    fn test_function_call_missing_id_generates_unique_id() {
        let resp = make_response(
            Some("STOP"),
            vec![GeminiPart {
                text: None,
                thought: None,
                function_call: Some(GeminiFunctionCall {
                    id: None,
                    name: "my_tool".to_string(),
                    args: json!({}),
                }),
                function_response: None,
                thought_signature: None,
            }],
        );
        let model_resp = format_response(resp).unwrap();
        match &model_resp.content[0] {
            ContentBlock::ToolUse { id, .. } => assert!(id.starts_with("tooluse_")),
            other => panic!("Expected ToolUse, got {:?}", other),
        }
    }

    #[test]
    fn test_max_tokens_finish_reason() {
        let resp = make_response(
            Some("MAX_TOKENS"),
            vec![GeminiPart {
                text: Some("truncated".to_string()),
                thought: None,
                function_call: None,
                function_response: None,
                thought_signature: None,
            }],
        );
        let model_resp = format_response(resp).unwrap();
        assert_eq!(model_resp.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_empty_candidates_returns_error() {
        let resp = GeminiResponse {
            candidates: vec![],
            usage_metadata: None,
        };
        let err = format_response(resp).unwrap_err();
        match err {
            KovaError::Provider { message, .. } => {
                assert!(message.contains("No candidates"));
            }
            other => panic!("Expected Provider error, got {:?}", other),
        }
    }

    #[test]
    fn test_usage_metadata_maps_to_usage_stats() {
        use super::super::types::GeminiUsageMetadata;
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart {
                        text: Some("Hi".to_string()),
                        thought: None,
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 5,
                total_token_count: 15,
                thoughts_token_count: Some(4),
            }),
        };
        let model_resp = format_response(resp).unwrap();
        let usage = model_resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.thinking_tokens, Some(4));
    }

    // ── Thinking / thought part tests ──────────────────────────────

    fn thought_part(text: &str) -> GeminiPart {
        GeminiPart {
            text: Some(text.to_string()),
            thought: Some(true),
            function_call: None,
            function_response: None,
            thought_signature: None,
        }
    }

    fn text_part(text: &str) -> GeminiPart {
        GeminiPart {
            text: Some(text.to_string()),
            thought: None,
            function_call: None,
            function_response: None,
            thought_signature: None,
        }
    }

    #[test]
    fn test_thinking_part_populates_thinking_field() {
        let resp = make_response(
            Some("STOP"),
            vec![
                thought_part("I need to consider this carefully."),
                text_part("Here's the answer."),
            ],
        );
        let model_resp = format_response(resp).unwrap();
        assert_eq!(
            model_resp.thinking,
            Some("I need to consider this carefully.".to_string())
        );
    }

    #[test]
    fn test_thinking_part_excluded_from_content_blocks() {
        let resp = make_response(
            Some("STOP"),
            vec![
                thought_part("internal reasoning"),
                text_part("visible answer"),
            ],
        );
        let model_resp = format_response(resp).unwrap();
        assert_eq!(model_resp.content.len(), 1);
        assert!(
            matches!(&model_resp.content[0], ContentBlock::Text { text } if text == "visible answer")
        );
    }

    #[test]
    fn test_multiple_thinking_parts_joined() {
        let resp = make_response(
            Some("STOP"),
            vec![
                thought_part("Step 1: "),
                thought_part("Step 2."),
                text_part("Done."),
            ],
        );
        let model_resp = format_response(resp).unwrap();
        assert_eq!(model_resp.thinking, Some("Step 1: Step 2.".to_string()));
    }

    #[test]
    fn test_empty_thinking_text_not_included_in_thinking_field() {
        let resp = make_response(
            Some("STOP"),
            vec![
                GeminiPart {
                    text: Some(String::new()),
                    thought: Some(true),
                    function_call: None,
                    function_response: None,
                    thought_signature: None,
                },
                text_part("response"),
            ],
        );
        let model_resp = format_response(resp).unwrap();
        assert!(
            model_resp.thinking.is_none(),
            "Empty thought text should not populate thinking field"
        );
    }

    #[test]
    fn test_thinking_only_response_has_no_content_blocks() {
        let resp = make_response(Some("STOP"), vec![thought_part("pure thinking, no output")]);
        let model_resp = format_response(resp).unwrap();
        assert!(model_resp.content.is_empty());
        assert!(model_resp.thinking.is_some());
    }

    // ── format_stream_events thinking tests ────────────────────────

    #[test]
    fn test_format_stream_events_thought_part_emits_thinking_delta() {
        let chunk = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![thought_part("thinking aloud")],
                }),
                finish_reason: None,
            }],
            usage_metadata: None,
        };
        let events = format_stream_events(chunk);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::ThinkingDelta {
                text: "thinking aloud".to_string()
            }
        );
    }

    #[test]
    fn test_format_stream_events_empty_thought_not_emitted() {
        let chunk = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart {
                        text: Some(String::new()),
                        thought: Some(true),
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                    }],
                }),
                finish_reason: None,
            }],
            usage_metadata: None,
        };
        let events = format_stream_events(chunk);
        assert!(
            events.is_empty(),
            "empty thought text should not produce ThinkingDelta"
        );
    }

    #[test]
    fn test_format_stream_events_mixed_thinking_and_content() {
        let chunk = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![thought_part("my reasoning"), text_part("my answer")],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };
        let events = format_stream_events(chunk);
        // StopEvent, ThinkingDelta, ContentDelta (stop is emitted first, then parts)
        assert!(
            events.iter().any(
                |e| matches!(e, StreamEvent::ThinkingDelta { text } if text == "my reasoning")
            )
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ContentDelta { text } if text == "my answer"))
        );
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::StopEvent {
                stop_reason: StopReason::EndTurn
            }
        )));
    }

    #[test]
    fn test_format_stream_events_thought_not_counted_as_function_call_for_stop() {
        let chunk = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![thought_part("thinking only")],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };
        let events = format_stream_events(chunk);
        // finish_reason=STOP with no real function_calls => StopReason::EndTurn
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::StopEvent {
                stop_reason: StopReason::EndTurn
            }
        )));
    }

    // ── format_request thinking_budget tests ───────────────────────

    #[test]
    fn test_format_request_thinking_budget_sets_generation_config() {
        let req = format_request(&[user_msg("Hi")], &[], &default_config(), Some(1024));
        let gc = req
            .generation_config
            .expect("generation_config should be set");
        let tc = gc.thinking_config.expect("thinking_config should be set");
        assert_eq!(tc.thinking_budget, 1024);
        assert_eq!(tc.include_thoughts, Some(true));
    }

    #[test]
    fn test_format_request_zero_thinking_budget_disables_thinking() {
        let req = format_request(&[user_msg("Hi")], &[], &default_config(), Some(0));
        let gc = req
            .generation_config
            .expect("generation_config should be set");
        let tc = gc.thinking_config.expect("thinking_config should be set");
        assert_eq!(tc.thinking_budget, 0);
        assert!(
            tc.include_thoughts.is_none(),
            "budget=0 should not set include_thoughts"
        );
    }

    #[test]
    fn test_format_request_negative_thinking_budget_enables_dynamic_thinking() {
        let req = format_request(&[user_msg("Hi")], &[], &default_config(), Some(-1));
        let gc = req
            .generation_config
            .expect("generation_config should be set");
        let tc = gc.thinking_config.expect("thinking_config should be set");
        assert_eq!(tc.thinking_budget, -1);
        assert_eq!(tc.include_thoughts, Some(true));
    }

    #[test]
    fn test_format_request_no_thinking_budget_no_generation_config_when_defaults() {
        let req = format_request(&[user_msg("Hi")], &[], &default_config(), None);
        assert!(req.generation_config.is_none());
    }

    // ── thoughtSignature round-trip tests ──────────────────────────

    #[test]
    fn test_format_request_thought_signature_in_tool_use_provider_metadata() {
        let messages = vec![
            user_msg("call the tool"),
            ConversationMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "fc-1".to_string(),
                    name: "my_tool".to_string(),
                    input: json!({"x": 1}),
                    provider_metadata: Some(json!({ "thoughtSignature": "sig-xyz" })),
                }],
            },
        ];
        let req = format_request(&messages, &[], &default_config(), None);
        // The model content has the tool use part
        let model_content = req
            .contents
            .iter()
            .find(|c| c.role == "model")
            .expect("model content");
        let part = &model_content.parts[0];
        assert_eq!(part.thought_signature.as_deref(), Some("sig-xyz"));
    }

    #[test]
    fn test_format_request_tool_use_without_thought_signature_has_no_signature() {
        let messages = vec![
            user_msg("call the tool"),
            ConversationMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "fc-2".to_string(),
                    name: "my_tool".to_string(),
                    input: json!({}),
                    provider_metadata: None,
                }],
            },
        ];
        let req = format_request(&messages, &[], &default_config(), None);
        let model_content = req
            .contents
            .iter()
            .find(|c| c.role == "model")
            .expect("model content");
        let part = &model_content.parts[0];
        assert!(part.thought_signature.is_none());
    }

    #[test]
    fn test_format_stream_events_function_call_with_thought_signature() {
        let chunk = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart {
                        text: None,
                        thought: None,
                        function_call: Some(GeminiFunctionCall {
                            id: Some("fc-10".to_string()),
                            name: "search".to_string(),
                            args: json!({"q": "rust"}),
                        }),
                        function_response: None,
                        thought_signature: Some("thought-sig-abc".to_string()),
                    }],
                }),
                finish_reason: None,
            }],
            usage_metadata: None,
        };
        let events = format_stream_events(chunk);
        let tool_delta = events
            .iter()
            .find(|e| matches!(e, StreamEvent::ToolUseDelta { .. }))
            .expect("should have ToolUseDelta");
        match tool_delta {
            StreamEvent::ToolUseDelta {
                provider_metadata, ..
            } => {
                let meta = provider_metadata
                    .as_ref()
                    .expect("should have provider_metadata");
                assert_eq!(meta["thoughtSignature"], "thought-sig-abc");
            }
            _ => panic!("expected ToolUseDelta"),
        }
    }

    #[test]
    fn test_format_response_function_call_with_thought_signature_stores_provider_metadata() {
        let resp = make_response(
            Some("STOP"),
            vec![GeminiPart {
                text: None,
                thought: None,
                function_call: Some(GeminiFunctionCall {
                    id: Some("fc-20".to_string()),
                    name: "get_data".to_string(),
                    args: json!({}),
                }),
                function_response: None,
                thought_signature: Some("sig-round-trip".to_string()),
            }],
        );
        let model_resp = format_response(resp).unwrap();
        match &model_resp.content[0] {
            ContentBlock::ToolUse {
                provider_metadata, ..
            } => {
                let meta = provider_metadata
                    .as_ref()
                    .expect("should have provider_metadata");
                assert_eq!(meta["thoughtSignature"], "sig-round-trip");
            }
            other => panic!("Expected ToolUse, got {:?}", other),
        }
    }
}
