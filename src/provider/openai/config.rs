use std::collections::HashMap;
use std::time::Duration;

use crate::provider::config::ProviderConfig;

const DEFAULT_CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
const DEFAULT_MODELS_PATH: &str = "/v1/models";
pub(super) const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Configuration for the OpenAI-compatible provider.
///
/// Covers any OpenAI-compatible REST API: OpenAI, Azure OpenAI, Ollama, vLLM,
/// LM Studio, etc. Use `new()` for required fields and `with_*` methods for
/// optional settings.
#[derive(Debug, Clone)]
pub struct OpenAiProviderConfig {
    /// Base URL of the provider API (e.g., `"https://api.openai.com"`).
    pub base_url: String,
    /// Model identifier to use for requests.
    pub model: String,
    /// Optional API key for `Authorization: Bearer` authentication.
    pub api_key: Option<String>,
    /// Request timeout duration.
    pub timeout: Duration,
    /// Maximum tokens for completions.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Endpoint path for chat completions. Default: `/v1/chat/completions`.
    pub chat_completions_path: String,
    /// Endpoint path for listing models. Default: `/v1/models`.
    pub models_path: String,
    /// Optional `api-version` query parameter (Azure OpenAI).
    pub api_version: Option<String>,
    /// Additional provider-specific key-value settings.
    pub extra: HashMap<String, String>,
}

impl OpenAiProviderConfig {
    /// Create a config with required fields and sensible defaults.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_tokens: None,
            temperature: None,
            chat_completions_path: DEFAULT_CHAT_COMPLETIONS_PATH.to_string(),
            models_path: DEFAULT_MODELS_PATH.to_string(),
            api_version: None,
            extra: HashMap::new(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_chat_completions_path(mut self, path: impl Into<String>) -> Self {
        self.chat_completions_path = path.into();
        self
    }

    pub fn with_models_path(mut self, path: impl Into<String>) -> Self {
        self.models_path = path.into();
        self
    }

    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Build a full URL for the given endpoint path, appending `api-version` if set.
    pub(super) fn build_url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let mut url = format!("{}{}", base, path);
        if let Some(ref v) = self.api_version {
            let sep = if url.contains('?') { '&' } else { '?' };
            url.push_str(&format!("{}api-version={}", sep, v));
        }
        url
    }

    pub(super) fn chat_completions_url(&self) -> String {
        self.build_url(&self.chat_completions_path)
    }

    pub(super) fn models_url(&self) -> String {
        self.build_url(&self.models_path)
    }
}

impl ProviderConfig for OpenAiProviderConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
    fn timeout(&self) -> Duration {
        self.timeout
    }
}
