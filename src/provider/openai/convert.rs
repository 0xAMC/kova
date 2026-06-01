use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde_json;

use super::types::{
    OaiChatCompletionRequest, OaiChatCompletionResponse, OaiFunctionCall, OaiFunctionDefinition,
    OaiMessage, OaiResponseChunk, OaiToolCall, OaiToolDefinition,
};
use crate::error::KovaError;
use crate::models::{
    ContentBlock, ConversationMessage, InferenceConfig, ModelResponse, Role, StopReason,
    StreamEvent, ToolDefinition, UsageStats,
};
use crate::streaming::sse::{SseLine, parse_sse_data, parse_sse_line};

// ── Request formatting ─────────────────────────────────────────────

pub(crate) fn format_request(
    messages: &[ConversationMessage],
    tools: &[ToolDefinition],
    config: &InferenceConfig,
) -> OaiChatCompletionRequest {
    let mut oai_messages = Vec::new();

    for msg in messages {
        // Tool-role messages: each ToolResult becomes a separate OaiMessage.
        if msg.role == Role::Tool {
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = block
                {
                    oai_messages.push(OaiMessage {
                        role: "tool".to_string(),
                        content: Some(content.clone()),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: Some(tool_use_id.clone()),
                        name: None,
                    });
                }
            }
            continue;
        }

        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => unreachable!(),
        };

        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => text_parts.push(text.clone()),
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => {
                    tool_calls.push(OaiToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: OaiFunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                    });
                }
                ContentBlock::ToolResult { .. } => {}
            }
        }

        oai_messages.push(OaiMessage {
            role: role_str.to_string(),
            content: if text_parts.is_empty() {
                None
            } else {
                Some(text_parts.join(""))
            },
            reasoning_content: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            name: None,
        });
    }

    let oai_tools = if tools.is_empty() {
        None
    } else {
        Some(
            tools
                .iter()
                .map(|t| OaiToolDefinition {
                    tool_type: "function".to_string(),
                    function: OaiFunctionDefinition {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect(),
        )
    };

    OaiChatCompletionRequest {
        model: config.model.clone().unwrap_or_default(),
        messages: oai_messages,
        tools: oai_tools,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        stream: None,
        stream_options: None,
        reasoning_effort: None,
    }
}

// ── Response formatting ────────────────────────────────────────────

pub(crate) fn format_response(
    oai_resp: OaiChatCompletionResponse,
) -> Result<ModelResponse, KovaError> {
    let choice = oai_resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| KovaError::Provider {
            message: "No choices in response".to_string(),
            status_code: None,
        })?;

    let role = choice.message.role.as_str();
    if role != "system" && role != "user" && role != "assistant" && role != "tool" {
        return Err(KovaError::Provider {
            message: format!("Unrecognized role: {}", role),
            status_code: None,
        });
    }

    let stop_reason = map_finish_reason(choice.finish_reason.as_deref());

    let mut content = Vec::new();
    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text { text });
    }
    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
            content.push(ContentBlock::ToolUse {
                id: tc.id,
                name: tc.function.name,
                input,
                provider_metadata: None,
            });
        }
    }

    let usage = oai_resp.usage.map(|u| UsageStats {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(ModelResponse {
        content,
        stop_reason,
        usage,
        thinking: None,
    })
}

fn map_finish_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some(other) => StopReason::Unknown(other.to_string()),
        None => StopReason::Unknown("none".to_string()),
    }
}

// ── Stream event formatting ────────────────────────────────────────

pub(crate) fn format_stream_event(chunk: OaiResponseChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if let Some(usage) = chunk.usage {
        events.push(StreamEvent::UsageEvent {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        });
    }

    for choice in chunk.choices {
        if let Some(reason) = &choice.finish_reason {
            events.push(StreamEvent::StopEvent {
                stop_reason: map_finish_reason(Some(reason)),
            });
        }
        if let Some(text) = &choice.delta.reasoning_content
            && !text.is_empty()
        {
            events.push(StreamEvent::ThinkingDelta { text: text.clone() });
        }
        if let Some(text) = &choice.delta.content
            && !text.is_empty()
        {
            events.push(StreamEvent::ContentDelta { text: text.clone() });
        }
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc_delta in tool_calls {
                events.push(StreamEvent::ToolUseDelta {
                    id: tc_delta.id.clone().unwrap_or_default(),
                    name: tc_delta.function.as_ref().and_then(|f| f.name.clone()),
                    input_delta: tc_delta.function.as_ref().and_then(|f| f.arguments.clone()),
                    provider_metadata: None,
                });
            }
        }
    }

    events
}

// ── SSE byte stream adapter ────────────────────────────────────────

/// Convert a raw byte stream into a stream of [`StreamEvent`]s.
///
/// Buffers incoming bytes, splits on newlines, parses each SSE line,
/// deserializes `data:` payloads as `OaiResponseChunk`, and maps them
/// to `StreamEvent`. Terminates on `[DONE]`.
pub(crate) fn sse_byte_stream_to_events(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>> {
    let mapped = byte_stream
        .map(|chunk| chunk.map_err(|e| KovaError::Stream(format!("Connection error: {e}"))));

    Box::pin(futures::stream::unfold(
        (
            Box::pin(mapped) as Pin<Box<dyn Stream<Item = Result<bytes::Bytes, KovaError>> + Send>>,
            String::new(),
            std::collections::VecDeque::<StreamEvent>::new(),
        ),
        |(mut stream, mut buffer, mut pending)| async move {
            if let Some(event) = pending.pop_front() {
                return Some((Ok(event), (stream, buffer, pending)));
            }

            loop {
                if let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    match parse_sse_line(&line) {
                        SseLine::Done => return None,
                        SseLine::Data(data) => match parse_sse_data::<OaiResponseChunk>(&data) {
                            Ok(chunk) => {
                                let mut events: std::collections::VecDeque<_> =
                                    format_stream_event(chunk).into();
                                if let Some(first) = events.pop_front() {
                                    return Some((Ok(first), (stream, buffer, events)));
                                }
                                continue;
                            }
                            Err(e) => return Some((Err(e), (stream, buffer, pending))),
                        },
                        SseLine::Empty | SseLine::Comment => continue,
                    }
                }

                match stream.next().await {
                    Some(Ok(bytes)) => buffer.push_str(&String::from_utf8_lossy(&bytes)),
                    Some(Err(e)) => return Some((Err(e), (stream, buffer, pending))),
                    None => return None,
                }
            }
        },
    ))
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
