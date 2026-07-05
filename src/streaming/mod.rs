#[cfg(any(
    feature = "openai",
    feature = "gemini",
    feature = "ollama",
    feature = "anthropic"
))]
pub(crate) mod line_stream;
#[cfg(any(feature = "openai", feature = "gemini", feature = "anthropic"))]
pub(crate) mod sse;
