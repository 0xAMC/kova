use std::time::Duration;

/// Configuration for the native Anthropic (Messages API) provider.
#[derive(Debug, Clone)]
pub struct AnthropicProviderConfig {
    /// Base URL of the API (default `https://api.anthropic.com`).
    pub base_url: String,
    /// Default model identifier (e.g. `claude-opus-4-8`).
    pub model: String,
    /// API key sent as `x-api-key`. `None` only makes sense against proxies
    /// that inject their own auth.
    pub api_key: Option<String>,
    /// Per-request timeout (default 300s — thinking models can run long).
    pub timeout: Duration,
    /// `max_tokens` used when the request's `InferenceConfig` doesn't set one.
    /// The Messages API requires it on every request (default 32_000).
    pub default_max_tokens: u32,
    /// Send `thinking: {"type": "adaptive"}` (recommended on Claude 4.6+;
    /// default true). Off = the model runs without extended thinking.
    pub adaptive_thinking: bool,
    /// `output_config.effort`: `low` | `medium` | `high` | `xhigh` | `max`.
    /// `None` uses the API default.
    pub effort: Option<String>,
    /// Automatic prompt caching (default true): a top-level
    /// `cache_control: {"type": "ephemeral"}` marks the last cacheable block,
    /// so repeated prefixes (system prompt, tools, conversation history) are
    /// served from cache. Cache reads/writes surface in `UsageStats`.
    pub cache: bool,
    /// `anthropic-version` header (default `2023-06-01`).
    pub api_version: String,
}

impl AnthropicProviderConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.anthropic.com".to_string(),
            model: model.into(),
            api_key: None,
            timeout: Duration::from_secs(300),
            default_max_tokens: 32_000,
            adaptive_thinking: true,
            effort: None,
            cache: true,
            api_version: "2023-06-01".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_default_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    pub fn with_adaptive_thinking(mut self, enabled: bool) -> Self {
        self.adaptive_thinking = enabled;
        self
    }

    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = Some(effort.into());
        self
    }

    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache = enabled;
        self
    }

    /// URL of the Messages endpoint.
    pub(crate) fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url.trim_end_matches('/'))
    }

    /// URL of the model-listing endpoint.
    pub(crate) fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url.trim_end_matches('/'))
    }
}
