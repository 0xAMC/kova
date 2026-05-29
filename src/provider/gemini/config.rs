use std::time::Duration;

use crate::provider::config::ProviderConfig;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_API_VERSION: &str = "v1beta";
pub(super) const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Configuration for the Google Gemini provider.
///
/// Authenticates via `x-goog-api-key` header. Use `new()` for required
/// fields and `with_*` methods for optional settings.
#[derive(Debug, Clone)]
pub struct GeminiProviderConfig {
    /// Base URL of the Gemini API. Defaults to `https://generativelanguage.googleapis.com`.
    pub base_url: String,
    /// Gemini model identifier, e.g. `"gemini-2.0-flash"`.
    pub model: String,
    /// API key for `x-goog-api-key` authentication.
    pub api_key: Option<String>,
    /// Request timeout duration. Defaults to 60 seconds.
    pub timeout: Duration,
    /// API version path segment. Defaults to `"v1beta"`.
    pub api_version: String,
}

impl GeminiProviderConfig {
    /// Create a config with the given model and sensible defaults.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            api_key: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            api_version: DEFAULT_API_VERSION.to_string(),
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

    /// Override the base URL (useful for testing with a mock server).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    pub(super) fn generate_content_url(&self, model: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{}/{}/models/{}:generateContent", base, self.api_version, model)
    }

    pub(super) fn stream_generate_content_url(&self, model: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        // alt=sse requests SSE format so we can reuse the shared SSE parser.
        format!(
            "{}/{}/models/{}:streamGenerateContent?alt=sse",
            base, self.api_version, model
        )
    }

    pub(super) fn models_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{}/{}/models", base, self.api_version)
    }
}

impl ProviderConfig for GeminiProviderConfig {
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
