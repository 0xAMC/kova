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
    /// A model reasoning block that must round-trip verbatim in conversation
    /// history. Anthropic requires the thinking blocks (with their opaque
    /// `signature`) to be passed back unchanged when continuing a tool-use
    /// turn; other providers ignore these blocks when building requests.
    #[serde(alias = "Thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        signature: Option<String>,
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
    /// Input tokens served from the provider's prompt cache (~10% price), when
    /// reported. Never included in `input_tokens` — kova normalizes providers
    /// that fold cached tokens into their prompt count (OpenAI), so
    /// `input_tokens + cache_read_tokens` is always the full prompt.
    /// `None` = unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    /// Input tokens written to the provider's prompt cache this call (billed
    /// at a premium), when reported. `None` = unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u32>,
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
    /// Constrain the model's final text to a JSON schema. Mapped natively per
    /// provider (OpenAI `response_format`, Anthropic `output_config.format`,
    /// Gemini `responseSchema`, Ollama `format`); Bedrock has no native
    /// equivalent and rejects requests that set this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// A JSON-schema constraint on the model's output.
///
/// Keep schemas simple — the common subset all providers accept: `object`
/// types with `properties`, `required`, `enum`, and `additionalProperties:
/// false`. Provider-specific constructs are stripped where required
/// (e.g. Gemini's schema sanitizer).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseFormat {
    /// Schema name, required by some providers (OpenAI); defaults to
    /// `"output"` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The JSON schema the final text must validate against.
    pub schema: serde_json::Value,
}

impl ResponseFormat {
    pub fn new(schema: serde_json::Value) -> Self {
        Self { name: None, schema }
    }

    pub fn named(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: Some(name.into()),
            schema,
        }
    }
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
        /// Prompt tokens served from cache, mirrored onto
        /// [`UsageStats::cache_read_tokens`].
        #[serde(default)]
        cache_read_tokens: Option<u32>,
        /// Prompt tokens written to cache, mirrored onto
        /// [`UsageStats::cache_creation_tokens`].
        #[serde(default)]
        cache_creation_tokens: Option<u32>,
    },
    /// A complete reasoning block with its provider signature, for verbatim
    /// round-trip in conversation history (Anthropic tool loops require it).
    /// Display consumers use [`ThinkingDelta`](Self::ThinkingDelta); this
    /// variant is consumed by the agent loop's accumulator and never doubles
    /// the visible text.
    ThinkingBlock {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
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
