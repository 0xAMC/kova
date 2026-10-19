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
    let message = parse_error_body(&body).map_or(body, |(message, _)| message);
    KovaError::provider_http(status, retry_after, message)
}

/// Message and HTTP-equivalent status from a provider error body.
///
/// Covers the shapes providers actually send: `{"error": {"message", "code"?}}`
/// (OpenAI, LM Studio, llama.cpp, Gemini — Gemini/llama.cpp put the status in a
/// numeric `code`; OpenAI's `code` is a string), `{"type": "error", "error":
/// {..}}` (Anthropic), and `{"error": "<text>", "status"?: <int>}` (Ollama).
pub(crate) fn parse_error_body(body: &str) -> Option<(String, Option<u16>)> {
    use serde_json::Value;
    let value: Value = serde_json::from_str(body).ok()?;
    let (message, code) = match value.get("error")? {
        Value::String(message) => (message.clone(), value.get("status")),
        Value::Object(obj) => (obj.get("message")?.as_str()?.to_string(), obj.get("code")),
        _ => return None,
    };
    let status = code
        .and_then(Value::as_u64)
        .and_then(|c| u16::try_from(c).ok())
        .filter(|c| (400..600).contains(c));
    Some((message, status))
}
