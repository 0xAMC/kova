use std::time::Duration;

/// Unified error type for the AgentX SDK.
#[derive(Debug, thiserror::Error)]
pub enum KovaError {
    #[error("Provider error: {message} (status: {status_code:?})")]
    Provider {
        message: String,
        status_code: Option<u16>,
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

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Max iterations ({0}) reached in tool-call loop")]
    MaxIterations(usize),
}
