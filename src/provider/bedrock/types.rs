use serde::{Deserialize, Serialize};

// ── Request types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockConverseRequest {
    pub(crate) messages: Vec<BedrockMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system: Option<Vec<BedrockSystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) inference_config: Option<BedrockInferenceConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_config: Option<BedrockToolConfig>,
    /// Model-specific parameters passed through verbatim as `additionalModelRequestFields`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) additional_model_request_fields: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BedrockMessage {
    pub(crate) role: String,
    pub(crate) content: Vec<BedrockContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) enum BedrockContentBlock {
    Text(String),
    #[serde(rename_all = "camelCase")]
    ToolUse {
        tool_use_id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename_all = "camelCase")]
    ToolResult {
        tool_use_id: String,
        content: Vec<BedrockToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    ReasoningContent {
        reasoning_text: BedrockReasoningText,
    },
    /// Prompt-cache breakpoint (`{"cachePoint": {"type": "default"}}`).
    /// Request-only; never appears in responses.
    CachePoint(BedrockCachePoint),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BedrockCachePoint {
    #[serde(rename = "type")]
    pub(crate) point_type: String,
}

impl BedrockCachePoint {
    pub(crate) fn default_point() -> Self {
        Self {
            point_type: "default".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BedrockReasoningText {
    pub(crate) text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BedrockToolResultContent {
    pub(crate) text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum BedrockSystemBlock {
    Text {
        text: String,
    },
    CachePoint {
        #[serde(rename = "cachePoint")]
        cache_point: BedrockCachePoint,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockInferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BedrockToolConfig {
    pub(crate) tools: Vec<BedrockToolSpec>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockToolSpec {
    pub(crate) tool_spec: BedrockToolSpecInner,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockToolSpecInner {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: BedrockInputSchema,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BedrockInputSchema {
    pub(crate) json: serde_json::Value,
}

// ── Response types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockConverseResponse {
    pub(crate) output: BedrockOutput,
    pub(crate) stop_reason: String,
    pub(crate) usage: BedrockUsage,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct BedrockOutput {
    pub(crate) message: BedrockMessage,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockUsage {
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) total_tokens: u32,
    /// Prompt tokens served from the Bedrock prompt cache (cachePoint models).
    #[serde(default)]
    pub(crate) cache_read_input_tokens: Option<u32>,
    /// Prompt tokens written to the Bedrock prompt cache.
    #[serde(default)]
    pub(crate) cache_write_input_tokens: Option<u32>,
}

// ── Stream event types ─────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum BedrockStreamEvent {
    ContentBlockStart {
        #[allow(dead_code)]
        content_block_index: u32,
        start: BedrockContentBlockStart,
    },
    ContentBlockDelta {
        #[allow(dead_code)]
        content_block_index: u32,
        delta: BedrockContentBlockDelta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        content_block_index: u32,
    },
    MessageStop {
        stop_reason: String,
    },
    Metadata {
        #[allow(dead_code)]
        usage: BedrockUsage,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum BedrockContentBlockStart {
    ToolUse {
        #[serde(rename = "toolUseId")]
        tool_use_id: String,
        name: String,
    },
    ReasoningContent {
        #[serde(rename = "reasoningType", default)]
        #[allow(dead_code)]
        reasoning_type: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum BedrockContentBlockDelta {
    Text(String),
    ToolUse {
        input: String,
    },
    ReasoningContent {
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        #[allow(dead_code)]
        signature: Option<String>,
    },
}

// ── Model listing types ────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockModelListResponse {
    pub(crate) model_summaries: Vec<BedrockModelSummary>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockModelSummary {
    pub(crate) model_id: String,
    #[allow(dead_code)]
    pub(crate) model_name: String,
    pub(crate) provider_name: String,
}
