use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Think ──────────────────────────────────────────────────────────
// Ollama accepts `true` (bool) or `"high"` / `"medium"` / `"low"` (string).

#[derive(Debug, Clone, PartialEq)]
pub enum OllamaThink {
    /// Basic thinking — serialises as `true`.
    Enabled,
    High,
    Medium,
    Low,
}

impl Serialize for OllamaThink {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Enabled => s.serialize_bool(true),
            Self::High => s.serialize_str("high"),
            Self::Medium => s.serialize_str("medium"),
            Self::Low => s.serialize_str("low"),
        }
    }
}

// ── Tools (request) ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OllamaTool {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: OllamaToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OllamaToolFunction {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

// ── Tool calls (response) ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OllamaToolCall {
    pub(crate) function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OllamaToolCallFunction {
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

// ── Messages ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OllamaMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: String,
    /// Reasoning trace; present when `think` is enabled.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) tool_calls: Option<Vec<OllamaToolCall>>,
    /// Base64-encoded image bytes for multimodal models.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) images: Option<Vec<String>>,
}

// ── Request ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct OllamaRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<OllamaMessage>,
    pub(crate) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<OllamaTool>>,
    /// Enables thinking/reasoning. Serialises as `true` or `"high"` / `"medium"` / `"low"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) think: Option<OllamaThink>,
    /// Force structured JSON output. Pass a JSON Schema `Value` or `"json"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) format: Option<Value>,
    /// Generation options: temperature, top_k, num_predict, seed, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) options: Option<serde_json::Map<String, Value>>,
    /// How long to keep the model loaded, e.g. `"5m"` or `"0"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) keep_alive: Option<String>,
}

// ── Response ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(crate) struct OllamaResponse {
    #[serde(default)]
    pub(crate) model: String,
    pub(crate) message: OllamaMessage,
    pub(crate) done: bool,
    /// `"stop"` | `"length"` | `"tool_use"`. Absent in intermediate stream chunks.
    #[serde(default)]
    pub(crate) done_reason: Option<String>,
    /// Input token count; only populated in the final done chunk.
    #[serde(default)]
    pub(crate) prompt_eval_count: u32,
    /// Output token count; only populated in the final done chunk.
    #[serde(default)]
    pub(crate) eval_count: u32,
    /// Wall-clock latency in nanoseconds.
    #[serde(default)]
    pub(crate) total_duration: u64,
}

// ── Model list ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(crate) struct OllamaModelListResponse {
    #[serde(default)]
    pub(crate) models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(crate) struct OllamaModelInfo {
    /// Model name including tag, e.g. `"llama3.2:latest"`.
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) size: u64,
    #[serde(default)]
    pub(crate) modified_at: String,
}
