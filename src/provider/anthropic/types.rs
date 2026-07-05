//! Wire types for the Anthropic Messages API (`POST /v1/messages`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct AnthropicRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Value>,
    /// Top-level automatic prompt caching: marks the last cacheable block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(super) struct WireMessage {
    pub role: &'static str,
    pub content: Vec<WireBlock>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum WireBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
}

#[derive(Debug, Serialize)]
pub(super) struct WireTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// ── Response (non-streaming) ────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicResponse {
    #[serde(default)]
    pub content: Vec<RespBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    pub usage: WireUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum RespBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    /// Unknown block types (redacted_thinking, server tool results, …) are
    /// tolerated and skipped rather than failing the whole response.
    #[serde(other)]
    Other,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct WireUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
}

// ── Response (streaming SSE) ────────────────────────────────────────

/// One decoded SSE `data:` payload. Discriminated by `type`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum SseEvent {
    MessageStart {
        message: SseMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: SseBlockStart,
    },
    ContentBlockDelta {
        index: usize,
        delta: SseDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        #[serde(default)]
        delta: SseMessageDelta,
        #[serde(default)]
        usage: Option<SseOutputUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: SseError,
    },
    /// Forward-compatible: skip event types we don't know.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub(super) struct SseMessageStart {
    pub usage: WireUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum SseBlockStart {
    Text {},
    ToolUse {
        id: String,
        name: String,
    },
    Thinking {},
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum SseDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct SseMessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct SseOutputUsage {
    #[serde(default)]
    pub output_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub(super) struct SseError {
    #[serde(default)]
    pub message: String,
}

// ── Model listing ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(super) struct ModelListResponse {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub display_name: Option<String>,
}
