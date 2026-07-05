//! Conversion between kova's canonical types and the Messages API wire format.

use std::pin::Pin;

use futures::Stream;
use serde_json::json;

use super::config::AnthropicProviderConfig;
use super::types::{
    AnthropicRequest, AnthropicResponse, RespBlock, SseBlockStart, SseDelta, SseEvent, WireBlock,
    WireMessage, WireTool, WireUsage,
};
use crate::error::KovaError;
use crate::models::{
    ContentBlock, ConversationMessage, InferenceConfig, ModelResponse, Role, StopReason,
    StreamEvent, ToolDefinition, UsageStats,
};
use crate::streaming::line_stream::{LineOutcome, line_stream_to_events};
use crate::streaming::sse::{SseLine, parse_sse_data, parse_sse_line};

// ── Request ─────────────────────────────────────────────────────────

/// Build a Messages API request from canonical history + tools + config.
///
/// - `Role::System` messages become the top-level `system` string (joined).
/// - `Role::Tool` messages become `user` messages of `tool_result` blocks.
/// - `Thinking` blocks round-trip on assistant messages when they carry a
///   signature (required to continue a tool-use turn); unsigned ones are
///   display-only and skipped.
pub(super) fn format_request(
    messages: &[ConversationMessage],
    tools: &[ToolDefinition],
    config: &InferenceConfig,
    provider: &AnthropicProviderConfig,
    stream: bool,
) -> AnthropicRequest {
    let mut system_parts: Vec<String> = Vec::new();
    let mut wire_messages: Vec<WireMessage> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                for block in &msg.content {
                    if let ContentBlock::Text { text } = block {
                        system_parts.push(text.clone());
                    }
                }
            }
            Role::User | Role::Tool => {
                let blocks: Vec<WireBlock> = msg.content.iter().filter_map(to_wire_block).collect();
                if !blocks.is_empty() {
                    wire_messages.push(WireMessage {
                        role: "user",
                        content: blocks,
                    });
                }
            }
            Role::Assistant => {
                let blocks: Vec<WireBlock> = msg.content.iter().filter_map(to_wire_block).collect();
                if !blocks.is_empty() {
                    wire_messages.push(WireMessage {
                        role: "assistant",
                        content: blocks,
                    });
                }
            }
        }
    }

    let wire_tools: Option<Vec<WireTool>> = if tools.is_empty() {
        None
    } else {
        Some(
            tools
                .iter()
                .map(|t| WireTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.parameters.clone(),
                })
                .collect(),
        )
    };

    AnthropicRequest {
        model: config
            .model
            .clone()
            .unwrap_or_else(|| provider.model.clone()),
        max_tokens: config.max_tokens.unwrap_or(provider.default_max_tokens),
        messages: wire_messages,
        system: if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        },
        tools: wire_tools,
        temperature: config.temperature,
        top_p: config.top_p,
        stop_sequences: config.stop_sequences.clone(),
        thinking: provider
            .adaptive_thinking
            .then(|| json!({"type": "adaptive"})),
        output_config: {
            let mut oc = serde_json::Map::new();
            if let Some(e) = provider.effort.as_ref() {
                oc.insert("effort".into(), json!(e));
            }
            if let Some(f) = config.response_format.as_ref() {
                oc.insert(
                    "format".into(),
                    json!({"type": "json_schema", "schema": f.schema}),
                );
            }
            (!oc.is_empty()).then_some(serde_json::Value::Object(oc))
        },
        cache_control: provider.cache.then(|| json!({"type": "ephemeral"})),
        stream: stream.then_some(true),
    }
}

fn to_wire_block(block: &ContentBlock) -> Option<WireBlock> {
    match block {
        ContentBlock::Text { text } => Some(WireBlock::Text { text: text.clone() }),
        ContentBlock::ToolUse {
            id, name, input, ..
        } => Some(WireBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        }),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => Some(WireBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        }),
        // Only signed thinking blocks are valid to send back; unsigned ones
        // came from another provider (or display-only) and are dropped.
        ContentBlock::Thinking {
            thinking,
            signature: Some(signature),
        } => Some(WireBlock::Thinking {
            thinking: thinking.clone(),
            signature: signature.clone(),
        }),
        ContentBlock::Thinking {
            signature: None, ..
        } => None,
    }
}

// ── Response (non-streaming) ────────────────────────────────────────

pub(super) fn format_response(resp: AnthropicResponse) -> Result<ModelResponse, KovaError> {
    let mut content: Vec<ContentBlock> = Vec::new();
    let mut thinking_text = String::new();

    for block in resp.content {
        match block {
            RespBlock::Text { text } => content.push(ContentBlock::Text { text }),
            RespBlock::ToolUse { id, name, input } => content.push(ContentBlock::ToolUse {
                id,
                name,
                input,
                provider_metadata: None,
            }),
            RespBlock::Thinking {
                thinking,
                signature,
            } => {
                if !thinking.is_empty() {
                    if !thinking_text.is_empty() {
                        thinking_text.push('\n');
                    }
                    thinking_text.push_str(&thinking);
                }
                content.push(ContentBlock::Thinking {
                    thinking,
                    signature,
                });
            }
            RespBlock::Other => {}
        }
    }

    Ok(ModelResponse {
        content,
        stop_reason: map_stop_reason(resp.stop_reason.as_deref()),
        usage: Some(usage_from_wire(&resp.usage, resp.usage.output_tokens)),
        thinking: (!thinking_text.is_empty()).then_some(thinking_text),
    })
}

pub(super) fn map_stop_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("tool_use") => StopReason::ToolUse,
        Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some(other) => StopReason::Unknown(other.to_string()),
        None => StopReason::EndTurn,
    }
}

fn usage_from_wire(usage: &WireUsage, output_tokens: u32) -> UsageStats {
    UsageStats {
        input_tokens: usage.input_tokens,
        output_tokens,
        total_tokens: usage.input_tokens + output_tokens,
        thinking_tokens: None,
        cache_read_tokens: usage.cache_read_input_tokens,
        cache_creation_tokens: usage.cache_creation_input_tokens,
    }
}

// ── Response (streaming) ────────────────────────────────────────────

/// What kind of block is open at each content index, with any state being
/// accumulated for it.
enum OpenBlock {
    Text,
    ToolUse,
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    Other,
}

/// Per-stream parser state, owned by the `FnMut` line closure.
struct StreamState {
    blocks: Vec<Option<OpenBlock>>,
    input_tokens: u32,
    cache_read: Option<u32>,
    cache_creation: Option<u32>,
    output_tokens: u32,
}

impl StreamState {
    fn slot(&mut self, index: usize) -> &mut Option<OpenBlock> {
        if self.blocks.len() <= index {
            self.blocks.resize_with(index + 1, || None);
        }
        &mut self.blocks[index]
    }
}

/// Convert the Messages API SSE byte stream into canonical [`StreamEvent`]s.
///
/// Emits `ContentDelta`/`ThinkingDelta`/`ToolUseDelta` as they arrive, a
/// complete `ThinkingBlock` at each thinking block's end (so the agent loop
/// can round-trip it with its signature), and a final `UsageEvent` +
/// `StopEvent` from `message_delta`.
pub(super) fn sse_byte_stream_to_events(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>> {
    let mut state = StreamState {
        blocks: Vec::new(),
        input_tokens: 0,
        cache_read: None,
        cache_creation: None,
        output_tokens: 0,
    };

    line_stream_to_events(byte_stream, move |line| match parse_sse_line(line) {
        SseLine::Data(data) => match parse_sse_data::<SseEvent>(&data) {
            Ok(event) => LineOutcome::Events(handle_sse_event(&mut state, event)),
            Err(e) => LineOutcome::Fail(e),
        },
        // The Messages API signals completion via `message_stop`, not [DONE].
        SseLine::Done | SseLine::Empty | SseLine::Comment => LineOutcome::Events(Vec::new()),
    })
}

fn handle_sse_event(state: &mut StreamState, event: SseEvent) -> Vec<StreamEvent> {
    match event {
        SseEvent::MessageStart { message } => {
            state.input_tokens = message.usage.input_tokens;
            state.cache_read = message.usage.cache_read_input_tokens;
            state.cache_creation = message.usage.cache_creation_input_tokens;
            Vec::new()
        }
        SseEvent::ContentBlockStart {
            index,
            content_block,
        } => match content_block {
            SseBlockStart::Text {} => {
                *state.slot(index) = Some(OpenBlock::Text);
                Vec::new()
            }
            SseBlockStart::ToolUse { id, name } => {
                *state.slot(index) = Some(OpenBlock::ToolUse);
                vec![StreamEvent::ToolUseDelta {
                    id,
                    name: Some(name),
                    input_delta: None,
                    provider_metadata: None,
                    index: Some(index as u32),
                }]
            }
            SseBlockStart::Thinking {} => {
                *state.slot(index) = Some(OpenBlock::Thinking {
                    thinking: String::new(),
                    signature: None,
                });
                Vec::new()
            }
            SseBlockStart::Other => {
                *state.slot(index) = Some(OpenBlock::Other);
                Vec::new()
            }
        },
        SseEvent::ContentBlockDelta { index, delta } => match delta {
            SseDelta::TextDelta { text } => vec![StreamEvent::ContentDelta { text }],
            SseDelta::InputJsonDelta { partial_json } => vec![StreamEvent::ToolUseDelta {
                id: String::new(),
                name: None,
                input_delta: Some(partial_json),
                provider_metadata: None,
                index: Some(index as u32),
            }],
            SseDelta::ThinkingDelta { thinking } => {
                if let Some(OpenBlock::Thinking {
                    thinking: acc_text, ..
                }) = state.slot(index).as_mut()
                {
                    acc_text.push_str(&thinking);
                }
                vec![StreamEvent::ThinkingDelta { text: thinking }]
            }
            SseDelta::SignatureDelta { signature } => {
                if let Some(OpenBlock::Thinking {
                    signature: acc_sig, ..
                }) = state.slot(index).as_mut()
                {
                    match acc_sig {
                        Some(existing) => existing.push_str(&signature),
                        None => *acc_sig = Some(signature),
                    }
                }
                Vec::new()
            }
            SseDelta::Other => Vec::new(),
        },
        SseEvent::ContentBlockStop { index } => match state.slot(index).take() {
            Some(OpenBlock::Thinking {
                thinking,
                signature,
            }) => vec![StreamEvent::ThinkingBlock {
                thinking,
                signature,
            }],
            _ => Vec::new(),
        },
        SseEvent::MessageDelta { delta, usage } => {
            if let Some(u) = usage {
                state.output_tokens = u.output_tokens;
            }
            let mut events = vec![StreamEvent::UsageEvent {
                input_tokens: state.input_tokens,
                output_tokens: state.output_tokens,
                thinking_tokens: None,
                cache_read_tokens: state.cache_read,
                cache_creation_tokens: state.cache_creation,
            }];
            if let Some(reason) = delta.stop_reason.as_deref() {
                events.push(StreamEvent::StopEvent {
                    stop_reason: map_stop_reason(Some(reason)),
                });
            }
            events
        }
        SseEvent::Error { error } => vec![StreamEvent::Error {
            message: error.message,
        }],
        SseEvent::MessageStop | SseEvent::Ping | SseEvent::Other => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::types::{SseMessageStart, SseOutputUsage};
    use super::*;

    fn provider_cfg() -> AnthropicProviderConfig {
        AnthropicProviderConfig::new("claude-opus-4-8")
    }

    fn msg(role: Role, block: ContentBlock) -> ConversationMessage {
        ConversationMessage {
            role,
            content: vec![block],
        }
    }

    fn text(t: &str) -> ContentBlock {
        ContentBlock::Text {
            text: t.to_string(),
        }
    }

    #[test]
    fn system_messages_become_top_level_system() {
        let messages = vec![
            msg(Role::System, text("You are helpful.")),
            msg(Role::User, text("hi")),
        ];
        let req = format_request(
            &messages,
            &[],
            &InferenceConfig::default(),
            &provider_cfg(),
            false,
        );
        assert_eq!(req.system.as_deref(), Some("You are helpful."));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn tool_role_messages_become_user_tool_results() {
        let messages = vec![msg(
            Role::Tool,
            ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "42".into(),
                is_error: false,
            },
        )];
        let req = format_request(
            &messages,
            &[],
            &InferenceConfig::default(),
            &provider_cfg(),
            false,
        );
        assert_eq!(req.messages[0].role, "user");
        let body = serde_json::to_value(&req.messages[0]).unwrap();
        assert_eq!(body["content"][0]["type"], "tool_result");
        assert_eq!(body["content"][0]["tool_use_id"], "tu_1");
    }

    #[test]
    fn signed_thinking_round_trips_unsigned_dropped() {
        let messages = vec![ConversationMessage {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".into(),
                    signature: Some("sig".into()),
                },
                ContentBlock::Thinking {
                    thinking: "display only".into(),
                    signature: None,
                },
                text("answer"),
            ],
        }];
        let req = format_request(
            &messages,
            &[],
            &InferenceConfig::default(),
            &provider_cfg(),
            false,
        );
        let body = serde_json::to_value(&req.messages[0]).unwrap();
        let blocks = body["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["signature"], "sig");
        assert_eq!(blocks[1]["type"], "text");
    }

    #[test]
    fn config_knobs_map_to_wire_fields() {
        let cfg = provider_cfg().with_effort("high");
        let req = format_request(
            &[msg(Role::User, text("hi"))],
            &[],
            &InferenceConfig::default(),
            &cfg,
            true,
        );
        assert_eq!(req.thinking, Some(json!({"type": "adaptive"})));
        assert_eq!(req.output_config, Some(json!({"effort": "high"})));
        assert_eq!(req.cache_control, Some(json!({"type": "ephemeral"})));
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.max_tokens, 32_000);

        let cfg = provider_cfg()
            .with_adaptive_thinking(false)
            .with_cache(false);
        let req = format_request(
            &[msg(Role::User, text("hi"))],
            &[],
            &InferenceConfig {
                max_tokens: Some(512),
                ..Default::default()
            },
            &cfg,
            false,
        );
        assert!(req.thinking.is_none());
        assert!(req.cache_control.is_none());
        assert!(req.stream.is_none());
        assert_eq!(req.max_tokens, 512);
    }

    #[test]
    fn tools_map_to_input_schema() {
        let tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "weather".into(),
            parameters: json!({"type": "object", "properties": {}}),
        }];
        let req = format_request(
            &[msg(Role::User, text("hi"))],
            &tools,
            &InferenceConfig::default(),
            &provider_cfg(),
            false,
        );
        let body = serde_json::to_value(req.tools.unwrap()).unwrap();
        assert_eq!(body[0]["name"], "get_weather");
        assert!(body[0]["input_schema"].is_object());
    }

    #[test]
    fn response_maps_blocks_stop_reason_and_cache_usage() {
        let resp: AnthropicResponse = serde_json::from_value(json!({
            "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "s1"},
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "tu_1", "name": "shell", "input": {"cmd": "ls"}},
                {"type": "redacted_thinking", "data": "opaque"}
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 100
            }
        }))
        .unwrap();
        let out = format_response(resp).unwrap();
        assert_eq!(out.stop_reason, StopReason::ToolUse);
        assert_eq!(out.thinking.as_deref(), Some("let me think"));
        // redacted_thinking is skipped; thinking/text/tool_use survive
        assert_eq!(out.content.len(), 3);
        assert!(matches!(
            out.content[0],
            ContentBlock::Thinking { ref signature, .. } if signature.as_deref() == Some("s1")
        ));
        let usage = out.usage.unwrap();
        assert_eq!(usage.cache_read_tokens, Some(900));
        assert_eq!(usage.cache_creation_tokens, Some(100));
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn sse_sequence_produces_expected_events() {
        let mut state = StreamState {
            blocks: Vec::new(),
            input_tokens: 0,
            cache_read: None,
            cache_creation: None,
            output_tokens: 0,
        };

        // message_start with cache usage
        let ev = handle_sse_event(
            &mut state,
            SseEvent::MessageStart {
                message: SseMessageStart {
                    usage: WireUsage {
                        input_tokens: 12,
                        output_tokens: 0,
                        cache_read_input_tokens: Some(800),
                        cache_creation_input_tokens: None,
                    },
                },
            },
        );
        assert!(ev.is_empty());

        // thinking block: deltas accumulate, signature attaches, stop emits ThinkingBlock
        handle_sse_event(
            &mut state,
            SseEvent::ContentBlockStart {
                index: 0,
                content_block: SseBlockStart::Thinking {},
            },
        );
        let ev = handle_sse_event(
            &mut state,
            SseEvent::ContentBlockDelta {
                index: 0,
                delta: SseDelta::ThinkingDelta {
                    thinking: "reason".into(),
                },
            },
        );
        assert_eq!(
            ev,
            vec![StreamEvent::ThinkingDelta {
                text: "reason".into()
            }]
        );
        handle_sse_event(
            &mut state,
            SseEvent::ContentBlockDelta {
                index: 0,
                delta: SseDelta::SignatureDelta {
                    signature: "sig".into(),
                },
            },
        );
        let ev = handle_sse_event(&mut state, SseEvent::ContentBlockStop { index: 0 });
        assert_eq!(
            ev,
            vec![StreamEvent::ThinkingBlock {
                thinking: "reason".into(),
                signature: Some("sig".into()),
            }]
        );

        // tool_use block correlated by index
        let ev = handle_sse_event(
            &mut state,
            SseEvent::ContentBlockStart {
                index: 1,
                content_block: SseBlockStart::ToolUse {
                    id: "tu_1".into(),
                    name: "shell".into(),
                },
            },
        );
        assert!(matches!(
            &ev[0],
            StreamEvent::ToolUseDelta { id, name: Some(n), index: Some(1), .. }
                if id == "tu_1" && n == "shell"
        ));
        let ev = handle_sse_event(
            &mut state,
            SseEvent::ContentBlockDelta {
                index: 1,
                delta: SseDelta::InputJsonDelta {
                    partial_json: "{\"cmd\":".into(),
                },
            },
        );
        assert!(matches!(
            &ev[0],
            StreamEvent::ToolUseDelta { input_delta: Some(d), index: Some(1), .. }
                if d == "{\"cmd\":"
        ));

        // message_delta: usage (with cache) + stop
        let ev = handle_sse_event(
            &mut state,
            SseEvent::MessageDelta {
                delta: super::super::types::SseMessageDelta {
                    stop_reason: Some("tool_use".into()),
                },
                usage: Some(SseOutputUsage { output_tokens: 7 }),
            },
        );
        assert_eq!(
            ev,
            vec![
                StreamEvent::UsageEvent {
                    input_tokens: 12,
                    output_tokens: 7,
                    thinking_tokens: None,
                    cache_read_tokens: Some(800),
                    cache_creation_tokens: None,
                },
                StreamEvent::StopEvent {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
    }

    #[tokio::test]
    async fn sse_byte_stream_end_to_end() {
        use futures::StreamExt;

        let body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let chunks: Vec<Result<bytes::Bytes, reqwest::Error>> =
            vec![Ok(bytes::Bytes::from_static(body.as_bytes()))];
        let stream = sse_byte_stream_to_events(futures::stream::iter(chunks));
        let events: Vec<_> = stream.map(|e| e.unwrap()).collect().await;
        assert_eq!(
            events,
            vec![
                StreamEvent::ContentDelta { text: "Hi".into() },
                StreamEvent::UsageEvent {
                    input_tokens: 3,
                    output_tokens: 2,
                    thinking_tokens: None,
                    cache_read_tokens: None,
                    cache_creation_tokens: None,
                },
                StreamEvent::StopEvent {
                    stop_reason: StopReason::EndTurn
                },
            ]
        );
    }

    #[test]
    fn response_format_merges_into_output_config() {
        let cfg = provider_cfg().with_effort("high");
        let inference = InferenceConfig {
            response_format: Some(crate::models::ResponseFormat::new(
                json!({"type": "object", "properties": {"route": {"type": "string"}}}),
            )),
            ..Default::default()
        };
        let req = format_request(&[msg(Role::User, text("hi"))], &[], &inference, &cfg, false);
        let oc = req.output_config.unwrap();
        assert_eq!(oc["effort"], "high");
        assert_eq!(oc["format"]["type"], "json_schema");
        assert_eq!(oc["format"]["schema"]["type"], "object");
    }
}
