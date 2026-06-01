use std::time::Duration;

use crate::provider::config::ProviderConfig;

use super::types::OllamaThink;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
pub(super) const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Configuration for the Ollama provider.
///
/// Connects to a locally-running (or remote) Ollama instance via its REST API.
/// Use [`new`](OllamaProviderConfig::new) for required fields and `with_*`
/// builders for optional settings.
///
/// # Example
/// ```rust,no_run
/// use kova_sdk::provider::ollama::{OllamaProvider, OllamaProviderConfig, OllamaThink};
///
/// let config = OllamaProviderConfig::new("qwen3")
///     .with_think(OllamaThink::High)
///     .with_keep_alive("10m");
///
/// let provider = OllamaProvider::new(config).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OllamaProviderConfig {
    /// Base URL of the Ollama server. Defaults to `http://localhost:11434`.
    pub base_url: String,
    /// Ollama model identifier, e.g. `"llama3.2"`, `"qwen3"`, `"deepseek-r1"`.
    pub model: String,
    /// Request timeout. Defaults to 120 s (local inference can be slow).
    pub timeout: Duration,
    /// How long to keep the model loaded after inference, e.g. `"5m"` or `"0"`.
    pub keep_alive: Option<String>,
    /// Reasoning mode. Requires a model that supports thinking (e.g. `qwen3`, `deepseek-r1`).
    pub think: Option<OllamaThink>,
    /// Extra generation options merged into the Ollama `options` object.
    /// Common keys: `"top_k"`, `"seed"`, `"stop"`, `"repeat_penalty"`, `"num_ctx"`.
    pub extra_options: Option<serde_json::Map<String, serde_json::Value>>,
}

impl OllamaProviderConfig {
    /// Create a config with the given model and sensible defaults.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            keep_alive: None,
            think: None,
            extra_options: None,
        }
    }

    /// Override the Ollama server URL (useful for remote instances or testing).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set model keep-alive duration. `"0"` unloads immediately after inference.
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Enable thinking/reasoning mode.
    pub fn with_think(mut self, think: OllamaThink) -> Self {
        self.think = Some(think);
        self
    }

    /// Merge additional generation options into every request's `options` object.
    pub fn with_extra_options(
        mut self,
        options: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.extra_options = Some(options);
        self
    }

    pub(super) fn chat_url(&self) -> String {
        format!("{}/api/chat", self.base_url.trim_end_matches('/'))
    }

    pub(super) fn tags_url(&self) -> String {
        format!("{}/api/tags", self.base_url.trim_end_matches('/'))
    }
}

impl ProviderConfig for OllamaProviderConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn api_key(&self) -> Option<&str> {
        None
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}
