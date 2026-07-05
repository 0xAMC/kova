use std::time::Duration;

use crate::error::KovaError;

/// Map a `reqwest` send error to an [`KovaError`].
///
/// Timeout errors map to `KovaError::Timeout`; everything else (including
/// connection refused) maps to `KovaError::Connection`.
pub(crate) fn map_request_error(e: reqwest::Error, timeout: Duration) -> KovaError {
    if e.is_timeout() {
        KovaError::Timeout(timeout)
    } else {
        KovaError::Connection(e.to_string())
    }
}

/// Turn a non-success provider HTTP response into a classified
/// [`KovaError::Provider`], consuming the response.
///
/// Reads the `Retry-After` header (seconds form) before draining the body so
/// rate-limit errors carry the provider's requested backoff.
pub(crate) async fn error_from_response(response: reqwest::Response) -> KovaError {
    let status = response.status().as_u16();
    let retry_after = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    let body = response.text().await.unwrap_or_default();
    KovaError::provider_http(status, retry_after, body)
}
