use serde::{Deserialize, Serialize};

// ── Parts ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    // Thinking models set this to true on chain-of-thought parts.
    // These parts must not be shown to users or treated as tool calls.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function_response: Option<GeminiFunctionResponse>,
    // Required by Gemini thinking models: must be round-tripped as-is
    // when replaying function calls in subsequent turns.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct GeminiFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    pub(crate) name: String,
    pub(crate) args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct GeminiFunctionResponse {
    pub(crate) name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    pub(crate) response: serde_json::Value,
}

// ── Content ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct GeminiContent {
    pub(crate) role: String,
    pub(crate) parts: Vec<GeminiPart>,
}

// ── Tools ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiTool {
    pub(crate) function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct GeminiFunctionDeclaration {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: serde_json::Value,
}

// ── GenerationConfig ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_config: Option<GeminiThinkingConfig>,
}

/// Controls chain-of-thought budget for thinking-capable Gemini models.
/// Set `thinking_budget` to -1 for dynamic (unlimited), 0 to disable,
/// or a positive value to cap tokens spent on reasoning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiThinkingConfig {
    pub(crate) thinking_budget: i32,
    // Must be true for thought parts to appear in the response; false = model
    // thinks silently. Omitted (None) when thinking is disabled (budget == 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) include_thoughts: Option<bool>,
}

// ── Request ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiRequest {
    pub(crate) contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) generation_config: Option<GeminiGenerationConfig>,
}

// ── Response ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiResponse {
    #[serde(default)]
    pub(crate) candidates: Vec<GeminiCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiCandidate {
    // Optional: Gemini's final usage/stop chunks often omit the content field entirely.
    #[serde(default)]
    pub(crate) content: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiUsageMetadata {
    #[serde(default)]
    pub(crate) prompt_token_count: u32,
    #[serde(default)]
    pub(crate) candidates_token_count: u32,
    #[serde(default)]
    pub(crate) total_token_count: u32,
    /// Reasoning tokens for thinking-capable models; absent otherwise.
    #[serde(default)]
    pub(crate) thoughts_token_count: Option<u32>,
}

// ── Model list ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiModelListResponse {
    #[serde(default)]
    pub(crate) models: Vec<GeminiModelInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) next_page_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiModelInfo {
    /// Resource name, e.g. `"models/gemini-2.0-flash"`.
    pub(crate) name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) version: Option<String>,
}
