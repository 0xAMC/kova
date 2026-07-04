use std::time::Duration;

/// Common interface that all provider configs must implement.
///
/// This lets generic code introspect
/// provider configuration without knowing the concrete type.
pub trait ProviderConfig {
    /// Base URL of the provider API.
    fn base_url(&self) -> &str;
    /// Model identifier to use for requests.
    fn model(&self) -> &str;
    /// Optional API key for authentication.
    fn api_key(&self) -> Option<&str>;
    /// Request timeout duration.
    fn timeout(&self) -> Duration;
}
