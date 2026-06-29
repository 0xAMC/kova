use serde::{Deserialize, Serialize};

// ── Roles ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

// ── Content Blocks ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    #[serde(alias = "Text")]
    Text { text: String },
    #[serde(alias = "ToolUse")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        // Provider-specific data that must be round-tripped verbatim in conversation history.
        // e.g. Gemini thinking models store {"thoughtSignature": "<blob>"} here.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        provider_metadata: Option<serde_json::Value>,
    },
    #[serde(alias = "ToolResult")]
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

// ── Messages ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversationMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

// ── Stop Reason ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Unknown(String),
}

impl StopReason {
    pub fn as_str(&self) -> &str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::ToolUse => "tool_use",
            StopReason::MaxTokens => "max_tokens",
            StopReason::Unknown(s) => s.as_str(),
        }
    }
}

// ── Usage Stats ────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageStats {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    /// Reasoning/thinking tokens, when the provider reports them separately
    /// (OpenAI o-series, Gemini). These are a *subset* of `output_tokens`, not
    /// additive. `None` means the provider gives no separate count (Anthropic,
    /// Bedrock, Ollama) — surfaced as "unknown" rather than a misleading `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u32>,
}

// ── Model Response ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Option<UsageStats>,
    /// Thinking/reasoning text from the model (not stored in conversation history).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

// ── Inference Config ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct InferenceConfig {
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Sequences that cause the model to stop generating.
    pub stop_sequences: Option<Vec<String>>,
}

// ── Tool Definition (canonical) ────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ── Streaming Types ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamEvent {
    ContentDelta {
        text: String,
    },
    ToolUseDelta {
        id: String,
        name: Option<String>,
        input_delta: Option<String>,
        provider_metadata: Option<serde_json::Value>,
        /// Provider-assigned position of this tool call in the response
        /// (OpenAI `index`, Bedrock `contentBlockIndex`). Deltas for the same
        /// call share an index; it is the only reliable correlation key when
        /// providers omit or repeat `id` across delta chunks.
        #[serde(default)]
        index: Option<u32>,
    },
    StopEvent {
        stop_reason: StopReason,
    },
    ThinkingDelta {
        text: String,
    },
    UsageEvent {
        input_tokens: u32,
        output_tokens: u32,
        /// Reasoning tokens when the provider reports them; `None` otherwise.
        /// A subset of `output_tokens`, mirrored onto [`UsageStats::thinking_tokens`].
        #[serde(default)]
        thinking_tokens: Option<u32>,
    },
    Error {
        message: String,
    },
}

// ── Tool Result (used by Tool trait) ───────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

// ── Model Info ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}
