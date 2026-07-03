use super::types::{
    BedrockCachePoint, BedrockContentBlock, BedrockContentBlockDelta, BedrockContentBlockStart,
    BedrockConverseRequest, BedrockConverseResponse, BedrockInferenceConfig, BedrockInputSchema,
    BedrockMessage, BedrockStreamEvent, BedrockSystemBlock, BedrockToolConfig,
    BedrockToolResultContent, BedrockToolSpec, BedrockToolSpecInner,
};
use crate::error::KovaError;
use crate::models::{
    ContentBlock, ConversationMessage, InferenceConfig, ModelResponse, Role, StopReason,
    StreamEvent, ToolDefinition, UsageStats,
};

pub(super) fn format_request(
    messages: &[ConversationMessage],
    tools: &[ToolDefinition],
    config: &InferenceConfig,
    additional_model_request_fields: Option<serde_json::Value>,
    cache: bool,
) -> BedrockConverseRequest {
    let mut system_blocks: Vec<BedrockSystemBlock> = Vec::new();
    let mut bedrock_messages: Vec<BedrockMessage> = Vec::new();

    for msg in messages {
        if msg.role == Role::System {
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    system_blocks.push(BedrockSystemBlock::Text { text: text.clone() });
                }
            }
            continue;
        }

        let role_str = match msg.role {
            Role::User | Role::Tool => "user",
            Role::Assistant => "assistant",
            Role::System => unreachable!(),
        };

        let content = msg
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(BedrockContentBlock::Text(text.clone())),
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => Some(BedrockContentBlock::ToolUse {
                    tool_use_id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => Some(BedrockContentBlock::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    content: vec![BedrockToolResultContent {
                        text: content.clone(),
                    }],
                    status: Some(if *is_error {
                        "error".to_string()
                    } else {
                        "success".to_string()
                    }),
                }),
                // Anthropic-signed reasoning blocks don't translate to the
                // Converse API; drop them when history crosses providers.
                ContentBlock::Thinking { .. } => None,
            })
            .collect();

        bedrock_messages.push(BedrockMessage {
            role: role_str.to_string(),
            content,
        });
    }

    // Prompt caching: a cachePoint after the system prompt caches it, and one
    // after the last message caches the whole conversation prefix, mirroring
    // the Anthropic provider's automatic placement. Opt-in — only cachePoint-
    // capable Bedrock models (Anthropic Claude, Amazon Nova) accept these.
    if cache {
        if !system_blocks.is_empty() {
            system_blocks.push(BedrockSystemBlock::CachePoint {
                cache_point: BedrockCachePoint::default_point(),
            });
        }
        if let Some(last) = bedrock_messages.last_mut() {
            last.content.push(BedrockContentBlock::CachePoint(
                BedrockCachePoint::default_point(),
            ));
        }
    }

    let system = if system_blocks.is_empty() {
        None
    } else {
        Some(system_blocks)
    };

    let inference_config = Some(BedrockInferenceConfig {
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_p: config.top_p,
        stop_sequences: config.stop_sequences.clone(),
    });

    let tool_config = if tools.is_empty() {
        None
    } else {
        Some(BedrockToolConfig {
            tools: tools
                .iter()
                .map(|t| BedrockToolSpec {
                    tool_spec: BedrockToolSpecInner {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: BedrockInputSchema {
                            json: t.parameters.clone(),
                        },
                    },
                })
                .collect(),
        })
    };

    BedrockConverseRequest {
        messages: bedrock_messages,
        system,
        inference_config,
        tool_config,
        additional_model_request_fields,
    }
}

pub(super) fn format_response(resp: BedrockConverseResponse) -> Result<ModelResponse, KovaError> {
    let mut thinking_parts: Vec<String> = Vec::new();
    let content = resp
        .output
        .message
        .content
        .into_iter()
        .filter_map(|block| match block {
            BedrockContentBlock::Text(text) => Some(ContentBlock::Text { text }),
            BedrockContentBlock::ToolUse {
                tool_use_id,
                name,
                input,
            } => Some(ContentBlock::ToolUse {
                id: tool_use_id,
                name,
                input,
                provider_metadata: None,
            }),
            BedrockContentBlock::ToolResult {
                tool_use_id,
                content,
                status,
            } => {
                let text = content
                    .into_iter()
                    .map(|c| c.text)
                    .collect::<Vec<_>>()
                    .join("");
                let is_error = status.as_deref() == Some("error");
                Some(ContentBlock::ToolResult {
                    tool_use_id,
                    content: text,
                    is_error,
                })
            }
            // Collect thinking text but exclude from conversation history.
            BedrockContentBlock::CachePoint(_) => None,
            BedrockContentBlock::ReasoningContent { reasoning_text } => {
                if !reasoning_text.text.is_empty() {
                    thinking_parts.push(reasoning_text.text);
                }
                None
            }
        })
        .collect();

    let thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join(""))
    };

    let stop_reason = map_stop_reason(&resp.stop_reason);

    let usage = UsageStats {
        input_tokens: resp.usage.input_tokens,
        output_tokens: resp.usage.output_tokens,
        total_tokens: resp.usage.total_tokens,
        // Bedrock folds reasoning into output_tokens; no separate count.
        thinking_tokens: None,
        cache_read_tokens: resp.usage.cache_read_input_tokens,
        cache_creation_tokens: resp.usage.cache_write_input_tokens,
    };

    Ok(ModelResponse {
        content,
        stop_reason,
        usage: Some(usage),
        thinking,
    })
}

pub(super) fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        other => StopReason::Unknown(other.to_string()),
    }
}

pub(super) fn format_stream_event(event: BedrockStreamEvent) -> Option<StreamEvent> {
    match event {
        BedrockStreamEvent::ContentBlockDelta {
            delta,
            content_block_index,
        } => match delta {
            BedrockContentBlockDelta::Text(text) => Some(StreamEvent::ContentDelta { text }),
            BedrockContentBlockDelta::ToolUse { input } => Some(StreamEvent::ToolUseDelta {
                id: String::new(),
                name: None,
                input_delta: Some(input),
                provider_metadata: None,
                index: Some(content_block_index),
            }),
            BedrockContentBlockDelta::ReasoningContent {
                text: Some(ref text),
                ..
            } if !text.is_empty() => {
                tracing::debug!(
                    len = text.len(),
                    "Bedrock ReasoningContent delta → ThinkingDelta"
                );
                Some(StreamEvent::ThinkingDelta { text: text.clone() })
            }
            BedrockContentBlockDelta::ReasoningContent {
                ref text,
                ref signature,
            } => {
                tracing::debug!(text = ?text, signature = ?signature, "Bedrock ReasoningContent delta (empty/signature-only, skipped)");
                None
            }
        },
        BedrockStreamEvent::ContentBlockStart {
            start,
            content_block_index,
        } => match start {
            BedrockContentBlockStart::ToolUse { tool_use_id, name } => {
                Some(StreamEvent::ToolUseDelta {
                    id: tool_use_id,
                    name: Some(name),
                    input_delta: None,
                    provider_metadata: None,
                    index: Some(content_block_index),
                })
            }
            BedrockContentBlockStart::ReasoningContent { .. } => None,
        },
        BedrockStreamEvent::MessageStop { stop_reason } => Some(StreamEvent::StopEvent {
            stop_reason: map_stop_reason(&stop_reason),
        }),
        BedrockStreamEvent::ContentBlockStop { .. } => None,
        BedrockStreamEvent::Metadata { usage } => Some(StreamEvent::UsageEvent {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            thinking_tokens: None,
            cache_read_tokens: usage.cache_read_input_tokens,
            cache_creation_tokens: usage.cache_write_input_tokens,
        }),
    }
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
