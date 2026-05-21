use super::types::{
    BedrockContentBlock, BedrockContentBlockDelta, BedrockContentBlockStart,
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
) -> BedrockConverseRequest {
    let mut system_blocks: Vec<BedrockSystemBlock> = Vec::new();
    let mut bedrock_messages: Vec<BedrockMessage> = Vec::new();

    for msg in messages {
        if msg.role == Role::System {
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    system_blocks.push(BedrockSystemBlock { text: text.clone() });
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
            .map(|block| match block {
                ContentBlock::Text { text } => BedrockContentBlock::Text(text.clone()),
                ContentBlock::ToolUse { id, name, input } => BedrockContentBlock::ToolUse {
                    tool_use_id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                },
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => BedrockContentBlock::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    content: vec![BedrockToolResultContent {
                        text: content.clone(),
                    }],
                    status: Some(if *is_error {
                        "error".to_string()
                    } else {
                        "success".to_string()
                    }),
                },
            })
            .collect();

        bedrock_messages.push(BedrockMessage {
            role: role_str.to_string(),
            content,
        });
    }

    let system = if system_blocks.is_empty() {
        None
    } else {
        Some(system_blocks)
    };

    let inference_config = Some(BedrockInferenceConfig {
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_p: None,
        stop_sequences: None,
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
    }
}

pub(super) fn format_response(resp: BedrockConverseResponse) -> Result<ModelResponse, KovaError> {
    let content = resp
        .output
        .message
        .content
        .into_iter()
        .map(|block| match block {
            BedrockContentBlock::Text(text) => ContentBlock::Text { text },
            BedrockContentBlock::ToolUse {
                tool_use_id,
                name,
                input,
            } => ContentBlock::ToolUse {
                id: tool_use_id,
                name,
                input,
            },
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
                ContentBlock::ToolResult {
                    tool_use_id,
                    content: text,
                    is_error,
                }
            }
        })
        .collect();

    let stop_reason = map_stop_reason(&resp.stop_reason);

    let usage = UsageStats {
        input_tokens: resp.usage.input_tokens,
        output_tokens: resp.usage.output_tokens,
        total_tokens: resp.usage.total_tokens,
    };

    Ok(ModelResponse {
        content,
        stop_reason,
        usage: Some(usage),
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
        BedrockStreamEvent::ContentBlockDelta { delta, .. } => match delta {
            BedrockContentBlockDelta::Text(text) => Some(StreamEvent::ContentDelta { text }),
            BedrockContentBlockDelta::ToolUse { input } => Some(StreamEvent::ToolUseDelta {
                id: String::new(),
                name: None,
                input_delta: Some(input),
            }),
        },
        BedrockStreamEvent::ContentBlockStart { start, .. } => match start {
            BedrockContentBlockStart::ToolUse { tool_use_id, name } => {
                Some(StreamEvent::ToolUseDelta {
                    id: tool_use_id,
                    name: Some(name),
                    input_delta: None,
                })
            }
        },
        BedrockStreamEvent::MessageStop { stop_reason } => Some(StreamEvent::StopEvent {
            stop_reason: map_stop_reason(&stop_reason),
        }),
        BedrockStreamEvent::ContentBlockStop { .. } | BedrockStreamEvent::Metadata { .. } => None,
    }
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
