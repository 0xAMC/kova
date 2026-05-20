use std::time::Duration;

use crate::provider::config::ProviderConfig;

pub(super) const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Percent-encode a string for use in a URL path segment.
///
/// Bedrock inference profile IDs (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`)
/// and ARNs contain characters like `:` and `/` that must be encoded in the URL path.
pub(super) fn encode_path_segment(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            // Unreserved characters (RFC 3986 §2.3) — safe in path segments
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            // Everything else gets percent-encoded
            _ => {
                encoded.push('%');
                encoded.push_str(&format!("{:02X}", byte));
            }
        }
    }
    encoded
}

pub struct BedrockProviderConfig {
    pub region: String,
    pub model_id: String,
    pub profile: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
    pub timeout: Duration,
    pub endpoint_url: Option<String>,
    base_url: String,
}

impl BedrockProviderConfig {
    pub fn new(region: impl Into<String>, model_id: impl Into<String>) -> Self {
        let region = region.into();
        let model_id = model_id.into();
        let base_url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse",
            region, encode_path_segment(&model_id)
        );
        Self {
            region,
            model_id,
            profile: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            endpoint_url: None,
            base_url,
        }
    }

    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self.session_token = session_token;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_endpoint_url(mut self, url: impl Into<String>) -> Self {
        self.endpoint_url = Some(url.into());
        self
    }
}

impl ProviderConfig for BedrockProviderConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn model(&self) -> &str {
        &self.model_id
    }

    fn api_key(&self) -> Option<&str> {
        None
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}
