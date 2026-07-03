use std::pin::Pin;

use futures::Stream;
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
use crate::streaming::line_stream::{LineOutcome, line_stream_to_events};
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
                // Anthropic-signed reasoning blocks don't exist in the
                // Chat Completions shape; drop them across providers.
                ContentBlock::ToolResult { .. } | ContentBlock::Thinking { .. } => {}
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
        top_p: config.top_p,
        stop: config.stop_sequences.clone(),
        stream: None,
        stream_options: None,
        reasoning_effort: None,
        response_format: config.response_format.as_ref().map(|f| {
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": f.name.clone().unwrap_or_else(|| "output".to_string()),
                    "schema": f.schema,
                    "strict": true,
                },
            })
        }),
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
        .ok_or_else(|| KovaError::provider_invalid("No choices in response".to_string()))?;

    let role = choice.message.role.as_str();
    if role != "system" && role != "user" && role != "assistant" && role != "tool" {
        return Err(KovaError::provider_invalid(format!(
            "Unrecognized role: {}",
            role
        )));
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
        thinking_tokens: u
            .completion_tokens_details
            .as_ref()
            .map(|d| d.reasoning_tokens),
        cache_read_tokens: u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens),
        // OpenAI's cache is automatic; writes aren't reported separately.
        cache_creation_tokens: None,
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
            thinking_tokens: usage
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens),
            cache_read_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens),
            cache_creation_tokens: None,
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
                    index: Some(tc_delta.index),
                });
            }
        }
    }

    events
}

// ── SSE byte stream adapter ────────────────────────────────────────

/// Convert a raw byte stream into a stream of [`StreamEvent`]s.
///
/// Parses each SSE line, deserializes `data:` payloads as
/// `OaiResponseChunk`, and maps them to `StreamEvent`. Terminates on
/// `[DONE]`.
pub(crate) fn sse_byte_stream_to_events(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>> {
    line_stream_to_events(byte_stream, |line| match parse_sse_line(line) {
        SseLine::Done => LineOutcome::Done,
        SseLine::Data(data) => match parse_sse_data::<OaiResponseChunk>(&data) {
            Ok(chunk) => LineOutcome::Events(format_stream_event(chunk)),
            Err(e) => LineOutcome::Fail(e),
        },
        SseLine::Empty | SseLine::Comment => LineOutcome::Events(Vec::new()),
    })
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
