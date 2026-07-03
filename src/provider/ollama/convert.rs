use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::Stream;

use super::config::OllamaProviderConfig;
use super::types::{
    OllamaMessage, OllamaModelListResponse, OllamaRequest, OllamaResponse, OllamaTool,
    OllamaToolCall, OllamaToolCallFunction, OllamaToolFunction,
};
use crate::error::KovaError;
use crate::models::{
    ContentBlock, ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, Role, StopReason,
    StreamEvent, ToolDefinition, UsageStats,
};
use crate::streaming::line_stream::{LineOutcome, line_stream_to_events};

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_tool_id() -> String {
    let n = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tooluse_{n:016x}")
}

// ── Request formatting ─────────────────────────────────────────────

/// Build the `options` map from `InferenceConfig` and provider-level extras.
fn build_options(
    provider_config: &OllamaProviderConfig,
    config: &InferenceConfig,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut opts = provider_config.extra_options.clone().unwrap_or_default();

    if let Some(temp) = config.temperature {
        opts.insert("temperature".to_string(), serde_json::json!(temp));
    }
    if let Some(max) = config.max_tokens {
        opts.insert("num_predict".to_string(), serde_json::json!(max));
    }
    if let Some(top_p) = config.top_p {
        opts.insert("top_p".to_string(), serde_json::json!(top_p));
    }
    if let Some(stop) = &config.stop_sequences {
        opts.insert("stop".to_string(), serde_json::json!(stop));
    }

    if opts.is_empty() { None } else { Some(opts) }
}

/// Convert kova's canonical message list to Ollama's flat message array.
///
/// Ollama messages carry a single string `content` field; structured content
/// (tool calls, tool results) maps to dedicated fields or separate messages.
fn build_messages(messages: &[ConversationMessage]) -> Vec<OllamaMessage> {
    let mut out: Vec<OllamaMessage> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    out.push(OllamaMessage {
                        role: "system".to_string(),
                        content: text,
                        thinking: None,
                        tool_calls: None,
                        images: None,
                    });
                }
            }

            Role::User => {
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    out.push(OllamaMessage {
                        role: "user".to_string(),
                        content: text,
                        thinking: None,
                        tool_calls: None,
                        images: None,
                    });
                }
            }

            Role::Assistant => {
                // Text and tool-call blocks can both appear in an assistant turn.
                // Emit text first (as a separate message), then tool calls (grouped).
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");

                if !text.is_empty() {
                    out.push(OllamaMessage {
                        role: "assistant".to_string(),
                        content: text,
                        thinking: None,
                        tool_calls: None,
                        images: None,
                    });
                }

                let tool_calls: Vec<OllamaToolCall> = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolUse { name, input, .. } = b {
                            Some(OllamaToolCall {
                                function: OllamaToolCallFunction {
                                    name: name.clone(),
                                    arguments: input.clone(),
                                },
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                if !tool_calls.is_empty() {
                    out.push(OllamaMessage {
                        role: "assistant".to_string(),
                        content: String::new(),
                        thinking: None,
                        tool_calls: Some(tool_calls),
                        images: None,
                    });
                }
            }

            // Each ToolResult becomes its own `tool`-role message.
            Role::Tool => {
                for block in &msg.content {
                    if let ContentBlock::ToolResult { content, .. } = block {
                        out.push(OllamaMessage {
                            role: "tool".to_string(),
                            content: content.clone(),
                            thinking: None,
                            tool_calls: None,
                            images: None,
                        });
                    }
                }
            }
        }
    }

    out
}

pub(crate) fn format_request(
    messages: &[ConversationMessage],
    tools: &[ToolDefinition],
    config: &InferenceConfig,
    provider_config: &OllamaProviderConfig,
    streaming: bool,
) -> OllamaRequest {
    let model = config
        .model
        .as_deref()
        .unwrap_or(&provider_config.model)
        .to_string();

    let ollama_tools = if tools.is_empty() {
        None
    } else {
        Some(
            tools
                .iter()
                .map(|t| OllamaTool {
                    kind: "function".to_string(),
                    function: OllamaToolFunction {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect(),
        )
    };

    OllamaRequest {
        model,
        messages: build_messages(messages),
        stream: streaming,
        tools: ollama_tools,
        think: provider_config.think.clone(),
        format: config.response_format.as_ref().map(|f| f.schema.clone()),
        options: build_options(provider_config, config),
        keep_alive: provider_config.keep_alive.clone(),
    }
}

// ── Stop reason ────────────────────────────────────────────────────

fn stop_reason_from(done_reason: Option<&str>, has_tool_calls: bool) -> StopReason {
    if has_tool_calls {
        return StopReason::ToolUse;
    }
    match done_reason {
        Some("length") => StopReason::MaxTokens,
        Some("tool_use") => StopReason::ToolUse,
        Some(other) if other != "stop" => StopReason::Unknown(other.to_string()),
        _ => StopReason::EndTurn,
    }
}

// ── Non-streaming response ─────────────────────────────────────────

pub(crate) fn format_response(resp: OllamaResponse) -> Result<ModelResponse, KovaError> {
    let tool_calls = resp.message.tool_calls.unwrap_or_default();
    let has_tool_calls = !tool_calls.is_empty();
    let mut content: Vec<ContentBlock> = Vec::new();

    if !resp.message.content.is_empty() {
        content.push(ContentBlock::Text {
            text: resp.message.content,
        });
    }

    for tc in tool_calls {
        content.push(ContentBlock::ToolUse {
            id: generate_tool_id(),
            name: tc.function.name,
            input: tc.function.arguments,
            provider_metadata: None,
        });
    }

    let stop_reason = stop_reason_from(resp.done_reason.as_deref(), has_tool_calls);

    let usage = if resp.prompt_eval_count > 0 || resp.eval_count > 0 {
        Some(UsageStats {
            input_tokens: resp.prompt_eval_count,
            output_tokens: resp.eval_count,
            total_tokens: resp.prompt_eval_count + resp.eval_count,
            thinking_tokens: None,
            cache_read_tokens: None,
            cache_creation_tokens: None,
        })
    } else {
        None
    };

    Ok(ModelResponse {
        content,
        stop_reason,
        usage,
        thinking: None,
    })
}

// ── Streaming chunk ────────────────────────────────────────────────

pub(crate) fn format_stream_chunk(resp: OllamaResponse) -> Vec<StreamEvent> {
    let mut events: Vec<StreamEvent> = Vec::new();

    // Thinking delta (if thinking mode is active on the model)
    if let Some(thinking) = &resp.message.thinking
        && !thinking.is_empty()
    {
        events.push(StreamEvent::ThinkingDelta {
            text: thinking.clone(),
        });
    }

    // Text content delta
    if !resp.message.content.is_empty() {
        events.push(StreamEvent::ContentDelta {
            text: resp.message.content.clone(),
        });
    }

    // Tool calls (Ollama delivers them whole, not as incremental deltas)
    let has_tool_calls = resp
        .message
        .tool_calls
        .as_deref()
        .is_some_and(|tc| !tc.is_empty());
    for tc in resp.message.tool_calls.as_deref().unwrap_or_default() {
        let input_delta = serde_json::to_string(&tc.function.arguments).ok();
        events.push(StreamEvent::ToolUseDelta {
            id: generate_tool_id(),
            name: Some(tc.function.name.clone()),
            input_delta,
            provider_metadata: None,
            // Ollama delivers each tool call complete in one chunk.
            index: None,
        });
    }

    if resp.done {
        events.push(StreamEvent::StopEvent {
            stop_reason: stop_reason_from(resp.done_reason.as_deref(), has_tool_calls),
        });

        if resp.prompt_eval_count > 0 || resp.eval_count > 0 {
            events.push(StreamEvent::UsageEvent {
                input_tokens: resp.prompt_eval_count,
                output_tokens: resp.eval_count,
                thinking_tokens: None,
                cache_read_tokens: None,
                cache_creation_tokens: None,
            });
        }
    }

    events
}

// ── NDJSON byte-stream adapter ─────────────────────────────────────
//
// Ollama streaming uses newline-delimited JSON: each line is a full
// OllamaResponse object. The stream closes after the chunk with `done: true`.

pub(crate) fn ndjson_byte_stream_to_events(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>> {
    line_stream_to_events(byte_stream, |line| {
        let line = line.trim();
        if line.is_empty() {
            return LineOutcome::Events(Vec::new());
        }
        match serde_json::from_str::<OllamaResponse>(line) {
            Ok(resp) => LineOutcome::Events(format_stream_chunk(resp)),
            Err(e) => LineOutcome::Fail(KovaError::provider_invalid(format!(
                "Failed to parse streaming chunk: {e}"
            ))),
        }
    })
}

// ── Model list ─────────────────────────────────────────────────────

pub(crate) fn format_model_list(resp: OllamaModelListResponse) -> Vec<ModelInfo> {
    resp.models
        .into_iter()
        .map(|m| ModelInfo {
            id: m.name,
            object: "model".to_string(),
            created: 0,
            owned_by: "ollama".to_string(),
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContentBlock, ConversationMessage, Role};
    use serde_json::json;

    fn default_config() -> InferenceConfig {
        InferenceConfig {
            model: Some("llama3.2".to_string()),
            max_tokens: None,
            temperature: None,
            ..Default::default()
        }
    }

    fn provider_config() -> OllamaProviderConfig {
        OllamaProviderConfig::new("llama3.2")
    }

    fn user_msg(text: &str) -> ConversationMessage {
        ConversationMessage {
            role: Role::User,
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

    // ── format_request ──────────────────────────────────────────────

    #[test]
    fn test_user_message_maps_correctly() {
        let req = format_request(
            &[user_msg("Hello")],
            &[],
            &default_config(),
            &provider_config(),
            false,
        );
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "Hello");
    }

    #[test]
    fn test_system_message_maps_correctly() {
        let msgs = vec![system_msg("Be helpful."), user_msg("Hi")];
        let req = format_request(&msgs, &[], &default_config(), &provider_config(), false);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[0].content, "Be helpful.");
        assert_eq!(req.messages[1].role, "user");
    }

    #[test]
    fn test_tool_calls_in_assistant_message() {
        let msgs = vec![
            user_msg("What's the weather?"),
            ConversationMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({"city": "Paris"}),
                    provider_metadata: None,
                }],
            },
        ];
        let req = format_request(&msgs, &[], &default_config(), &provider_config(), false);
        let asst = &req.messages[1];
        assert_eq!(asst.role, "assistant");
        let tcs = asst.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_weather");
        assert_eq!(tcs[0].function.arguments["city"], "Paris");
    }

    #[test]
    fn test_tool_result_becomes_tool_role_message() {
        let msgs = vec![
            user_msg("Time?"),
            ConversationMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "get_time".to_string(),
                    input: json!({}),
                    provider_metadata: None,
                }],
            },
            ConversationMessage {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: "12:00 PM".to_string(),
                    is_error: false,
                }],
            },
        ];
        let req = format_request(&msgs, &[], &default_config(), &provider_config(), false);
        let tool_msg = req.messages.last().unwrap();
        assert_eq!(tool_msg.role, "tool");
        assert_eq!(tool_msg.content, "12:00 PM");
    }

    #[test]
    fn test_tools_become_ollama_tool_format() {
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }];
        let req = format_request(
            &[user_msg("Search cats")],
            &tools,
            &default_config(),
            &provider_config(),
            false,
        );
        let ollama_tools = req.tools.unwrap();
        assert_eq!(ollama_tools.len(), 1);
        assert_eq!(ollama_tools[0].kind, "function");
        assert_eq!(ollama_tools[0].function.name, "search");
    }

    #[test]
    fn test_empty_tools_omits_tools_field() {
        let req = format_request(
            &[user_msg("Hi")],
            &[],
            &default_config(),
            &provider_config(),
            false,
        );
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_options_include_temperature_and_max_tokens() {
        let config = InferenceConfig {
            model: Some("llama3.2".to_string()),
            max_tokens: Some(512),
            temperature: Some(0.7),
            ..Default::default()
        };
        let req = format_request(&[user_msg("Hi")], &[], &config, &provider_config(), false);
        let opts = req.options.unwrap();
        assert_eq!(opts["temperature"], json!(0.7_f32));
        assert_eq!(opts["num_predict"], json!(512_u32));
    }

    #[test]
    fn test_no_options_when_defaults() {
        let req = format_request(
            &[user_msg("Hi")],
            &[],
            &default_config(),
            &provider_config(),
            false,
        );
        assert!(req.options.is_none());
    }

    #[test]
    fn test_stream_flag_is_set() {
        let req = format_request(
            &[user_msg("Hi")],
            &[],
            &default_config(),
            &provider_config(),
            true,
        );
        assert!(req.stream);
    }

    #[test]
    fn test_think_forwarded_from_provider_config() {
        use super::super::types::OllamaThink;
        let mut pc = provider_config();
        pc.think = Some(OllamaThink::High);
        let req = format_request(&[user_msg("Hi")], &[], &default_config(), &pc, false);
        assert!(req.think.is_some());
        // Serialise and verify the value
        let serialised = serde_json::to_value(&req).unwrap();
        assert_eq!(serialised["think"], json!("high"));
    }

    #[test]
    fn test_keep_alive_forwarded() {
        let mut pc = provider_config();
        pc.keep_alive = Some("10m".to_string());
        let req = format_request(&[user_msg("Hi")], &[], &default_config(), &pc, false);
        assert_eq!(req.keep_alive.as_deref(), Some("10m"));
    }

    // ── format_response ─────────────────────────────────────────────

    fn make_response(content: &str, done_reason: &str) -> OllamaResponse {
        use super::super::types::OllamaMessage;
        OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: content.to_string(),
                thinking: None,
                tool_calls: None,
                images: None,
            },
            done: true,
            done_reason: Some(done_reason.to_string()),
            prompt_eval_count: 10,
            eval_count: 20,
            total_duration: 1_000_000_000,
        }
    }

    #[test]
    fn test_text_response() {
        let resp = format_response(make_response("Hello!", "stop")).unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(matches!(&resp.content[0], ContentBlock::Text { text } if text == "Hello!"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
    }

    #[test]
    fn test_max_tokens_stop_reason() {
        let resp = format_response(make_response("...", "length")).unwrap();
        assert_eq!(resp.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_tool_call_response() {
        use super::super::types::{OllamaMessage, OllamaToolCall, OllamaToolCallFunction};
        let ollama_resp = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                thinking: None,
                tool_calls: Some(vec![OllamaToolCall {
                    function: OllamaToolCallFunction {
                        name: "search".to_string(),
                        arguments: json!({"q": "cats"}),
                    },
                }]),
                images: None,
            },
            done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: 5,
            eval_count: 15,
            total_duration: 0,
        };
        let resp = format_response(ollama_resp).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        match &resp.content[0] {
            ContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "search");
                assert_eq!(input["q"], "cats");
            }
            other => panic!("expected ToolUse, got {:?}", other),
        }
    }

    // ── format_stream_chunk ─────────────────────────────────────────

    #[test]
    fn test_stream_chunk_text_delta() {
        use super::super::types::OllamaMessage;
        let chunk = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: "Hello".to_string(),
                thinking: None,
                tool_calls: None,
                images: None,
            },
            done: false,
            done_reason: None,
            prompt_eval_count: 0,
            eval_count: 0,
            total_duration: 0,
        };
        let events = format_stream_chunk(chunk);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::ContentDelta { text } if text == "Hello"));
    }

    #[test]
    fn test_stream_chunk_thinking_delta() {
        use super::super::types::OllamaMessage;
        let chunk = OllamaResponse {
            model: "qwen3".to_string(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                thinking: Some("Let me think...".to_string()),
                tool_calls: None,
                images: None,
            },
            done: false,
            done_reason: None,
            prompt_eval_count: 0,
            eval_count: 0,
            total_duration: 0,
        };
        let events = format_stream_chunk(chunk);
        assert!(
            matches!(&events[0], StreamEvent::ThinkingDelta { text } if text == "Let me think...")
        );
    }

    #[test]
    fn test_stream_done_chunk_emits_stop_and_usage() {
        use super::super::types::OllamaMessage;
        let chunk = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                thinking: None,
                tool_calls: None,
                images: None,
            },
            done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: 8,
            eval_count: 12,
            total_duration: 500_000_000,
        };
        let events = format_stream_chunk(chunk);
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::StopEvent {
                stop_reason: StopReason::EndTurn
            }
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::UsageEvent {
                input_tokens: 8,
                output_tokens: 12,
                ..
            }
        )));
    }

    #[test]
    fn test_stream_tool_call_chunk() {
        use super::super::types::{OllamaMessage, OllamaToolCall, OllamaToolCallFunction};
        let chunk = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                thinking: None,
                tool_calls: Some(vec![OllamaToolCall {
                    function: OllamaToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: json!({"city": "London"}),
                    },
                }]),
                images: None,
            },
            done: false,
            done_reason: None,
            prompt_eval_count: 0,
            eval_count: 0,
            total_duration: 0,
        };
        let events = format_stream_chunk(chunk);
        assert!(events.iter().any(
            |e| matches!(e, StreamEvent::ToolUseDelta { name: Some(n), .. } if n == "get_weather")
        ));
    }

    #[test]
    fn response_format_maps_to_format_field() {
        let provider_config = OllamaProviderConfig::new("llama3.2");
        let schema = serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}}});
        let config = InferenceConfig {
            response_format: Some(crate::models::ResponseFormat::new(schema.clone())),
            ..Default::default()
        };
        let req = format_request(&[], &[], &config, &provider_config, false);
        assert_eq!(req.format, Some(schema));

        let req = format_request(&[], &[], &InferenceConfig::default(), &provider_config, false);
        assert!(req.format.is_none());
    }
}
