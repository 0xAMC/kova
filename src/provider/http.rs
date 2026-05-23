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
