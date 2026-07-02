use std::time::Duration;

/// Classification of a provider (LLM API) failure.
///
/// Derived from the HTTP status of the provider response (or the provider's
/// own error envelope, e.g. Bedrock exception types normalized to statuses).
/// Lets applications react precisely — prompt for a new key on
/// [`AuthInvalid`](ProviderErrorClass::AuthInvalid), back off on
/// [`RateLimited`](ProviderErrorClass::RateLimited) — without sniffing
/// messages or status codes themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderErrorClass {
    /// Credentials missing, malformed, or revoked (401).
    AuthInvalid,
    /// Credentials valid but not permitted for this model/operation (403).
    AuthForbidden,
    /// Provider throttled the request (429). `retry_after` is taken from the
    /// `Retry-After` response header when the provider sends one.
    RateLimited { retry_after: Option<Duration> },
    /// Transient provider-side failure worth retrying (408, 500, 502, 503,
    /// 504, and Anthropic's 529 overloaded).
    Overloaded,
    /// The request itself was rejected (400/413/422 — includes
    /// context-length violations).
    InvalidRequest,
    /// Unknown model or endpoint (404).
    NotFound,
    /// Anything else, including malformed provider responses.
    Other,
}

impl ProviderErrorClass {
    /// Classify an HTTP status returned by a provider.
    pub fn from_status(status: u16, retry_after: Option<Duration>) -> Self {
        match status {
            401 => Self::AuthInvalid,
            403 => Self::AuthForbidden,
            429 => Self::RateLimited { retry_after },
            408 | 500 | 502 | 503 | 504 | 529 => Self::Overloaded,
            400 | 413 | 422 => Self::InvalidRequest,
            404 => Self::NotFound,
            _ => Self::Other,
        }
    }

    /// Whether retrying the request might succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimited { .. } | Self::Overloaded)
    }
}

/// Unified error type for the kova SDK.
#[derive(Debug, thiserror::Error)]
pub enum KovaError {
    #[error("Provider error: {message} (status: {status_code:?})")]
    Provider {
        message: String,
        status_code: Option<u16>,
        class: ProviderErrorClass,
    },

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Tool execution error: {tool_name}: {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Orchestration error: {0}")]
    Orchestration(String),

    #[error("Build error: {0}")]
    Build(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Max iterations ({0}) reached in tool-call loop")]
    MaxIterations(usize),

    #[error("Turn cancelled")]
    Cancelled,
}

impl KovaError {
    /// A provider failure from an HTTP error response, classified from the
    /// status code.
    pub fn provider_http(
        status: u16,
        retry_after: Option<Duration>,
        message: impl Into<String>,
    ) -> Self {
        KovaError::Provider {
            message: message.into(),
            status_code: Some(status),
            class: ProviderErrorClass::from_status(status, retry_after),
        }
    }

    /// A provider failure without an HTTP status — malformed or
    /// undeserializable provider responses. Classified as
    /// [`ProviderErrorClass::Other`].
    pub fn provider_invalid(message: impl Into<String>) -> Self {
        KovaError::Provider {
            message: message.into(),
            status_code: None,
            class: ProviderErrorClass::Other,
        }
    }

    /// A credential failure detected before any HTTP exchange (e.g. an AWS
    /// credential chain that resolves nothing). Classified as
    /// [`ProviderErrorClass::AuthInvalid`].
    pub fn provider_auth(message: impl Into<String>) -> Self {
        KovaError::Provider {
            message: message.into(),
            status_code: None,
            class: ProviderErrorClass::AuthInvalid,
        }
    }

    /// HTTP status code reported by the provider, when available.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            KovaError::Provider { status_code, .. } => *status_code,
            _ => None,
        }
    }

    /// Classification of a provider failure, when this is a provider error.
    pub fn provider_class(&self) -> Option<&ProviderErrorClass> {
        match self {
            KovaError::Provider { class, .. } => Some(class),
            _ => None,
        }
    }

    /// Whether retrying the request might succeed.
    ///
    /// True for connection failures, timeouts, and provider errors classified
    /// as [`RateLimited`](ProviderErrorClass::RateLimited) or
    /// [`Overloaded`](ProviderErrorClass::Overloaded).
    pub fn is_retryable(&self) -> bool {
        match self {
            KovaError::Connection(_) | KovaError::Timeout(_) => true,
            KovaError::Provider { class, .. } => class.is_retryable(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_auth_statuses() {
        assert_eq!(
            ProviderErrorClass::from_status(401, None),
            ProviderErrorClass::AuthInvalid
        );
        assert_eq!(
            ProviderErrorClass::from_status(403, None),
            ProviderErrorClass::AuthForbidden
        );
    }

    #[test]
    fn classify_rate_limit_carries_retry_after() {
        let class = ProviderErrorClass::from_status(429, Some(Duration::from_secs(7)));
        assert_eq!(
            class,
            ProviderErrorClass::RateLimited {
                retry_after: Some(Duration::from_secs(7))
            }
        );
        assert!(class.is_retryable());
    }

    #[test]
    fn classify_transient_and_terminal_statuses() {
        for status in [408, 500, 502, 503, 504, 529] {
            assert_eq!(
                ProviderErrorClass::from_status(status, None),
                ProviderErrorClass::Overloaded,
                "status {status}"
            );
        }
        for status in [400, 413, 422] {
            assert_eq!(
                ProviderErrorClass::from_status(status, None),
                ProviderErrorClass::InvalidRequest,
                "status {status}"
            );
        }
        assert_eq!(
            ProviderErrorClass::from_status(404, None),
            ProviderErrorClass::NotFound
        );
        assert_eq!(
            ProviderErrorClass::from_status(418, None),
            ProviderErrorClass::Other
        );
    }

    #[test]
    fn retryability_matches_class() {
        assert!(KovaError::provider_http(429, None, "slow down").is_retryable());
        assert!(KovaError::provider_http(529, None, "overloaded").is_retryable());
        assert!(!KovaError::provider_http(401, None, "bad key").is_retryable());
        assert!(!KovaError::provider_http(400, None, "too long").is_retryable());
        assert!(!KovaError::provider_invalid("garbled").is_retryable());
    }

    #[test]
    fn constructors_set_class() {
        assert_eq!(
            KovaError::provider_auth("no creds").provider_class(),
            Some(&ProviderErrorClass::AuthInvalid)
        );
        assert_eq!(
            KovaError::provider_invalid("bad json").provider_class(),
            Some(&ProviderErrorClass::Other)
        );
        assert_eq!(
            KovaError::provider_http(404, None, "no such model").provider_class(),
            Some(&ProviderErrorClass::NotFound)
        );
        assert_eq!(
            KovaError::Connection("refused".into()).provider_class(),
            None
        );
    }
}
