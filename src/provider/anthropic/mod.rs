//! Native Anthropic (Messages API) provider.
//!
//! Speaks `POST /v1/messages` directly: streaming SSE, tool use, extended
//! (adaptive) thinking with signed thinking-block round-trip, and automatic
//! prompt caching (cache read/write tokens surface in `UsageStats`).

mod config;
mod convert;
mod provider;
mod types;

pub use config::AnthropicProviderConfig;
pub use provider::AnthropicProvider;
