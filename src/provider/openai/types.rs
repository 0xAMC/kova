use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiMessage {
    pub(crate) role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    // DeepSeek uses "reasoning_content"; LM Studio / OpenAI o-series uses "reasoning"
    #[serde(alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<OaiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) call_type: String,
    pub(crate) function: OaiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiFunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiStreamOptions {
    pub(crate) include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiChatCompletionRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<OaiToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream_options: Option<OaiStreamOptions>,
    // OpenAI o-series: "low" | "medium" | "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiChatCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<OaiChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<OaiUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiChoice {
    pub(crate) index: u32,
    pub(crate) message: OaiMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiUsage {
    pub(crate) prompt_tokens: u32,
    pub(crate) completion_tokens: u32,
    pub(crate) total_tokens: u32,
    /// Present for o-series reasoning models; carries `reasoning_tokens`.
    #[serde(default)]
    pub(crate) completion_tokens_details: Option<OaiCompletionTokensDetails>,
    /// Carries `cached_tokens` (prompt tokens served from OpenAI's automatic
    /// prompt cache) when the API reports it.
    #[serde(default)]
    pub(crate) prompt_tokens_details: Option<OaiPromptTokensDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiCompletionTokensDetails {
    #[serde(default)]
    pub(crate) reasoning_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiPromptTokensDetails {
    #[serde(default)]
    pub(crate) cached_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiToolDefinition {
    #[serde(rename = "type")]
    pub(crate) tool_type: String,
    pub(crate) function: OaiFunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiFunctionDefinition {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiResponseChunk {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<OaiChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<OaiUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiChunkChoice {
    pub(crate) index: u32,
    pub(crate) delta: OaiDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    // DeepSeek uses "reasoning_content"; LM Studio / OpenAI o-series uses "reasoning"
    #[serde(alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<OaiToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiToolCallDelta {
    pub(crate) index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function: Option<OaiFunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiFunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiModelListResponse {
    pub(crate) data: Vec<OaiModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OaiModelInfo {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: u64,
    pub(crate) owned_by: String,
}
