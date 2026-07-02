pub mod agent;
pub mod error;
pub mod mcp;
pub mod memory;
pub mod models;
pub mod orchestrator;
pub mod provider;
pub mod streaming;
pub mod telemetry;
pub mod tool;
#[cfg(feature = "tools")]
pub mod tools;

pub use tool::ToolLifecycleHook;
pub use tool::approval::{ApprovalDecision, ToolApprovalHandler};

/// Convenience re-exports for the most commonly used types.
///
/// ```
/// use kova_sdk::prelude::*;
/// ```
pub mod prelude {
    pub use crate::agent::{Agent, AgentBuilder, AgentEvent, AgentResponse};
    pub use crate::error::{KovaError, ProviderErrorClass};
    pub use crate::memory::MemoryStore;
    pub use crate::memory::in_memory::InMemoryStore;
    pub use crate::models::{
        ContentBlock, ConversationMessage, InferenceConfig, ModelResponse, Role, StopReason,
        StreamEvent, ToolDefinition, ToolResult, UsageStats,
    };
    pub use crate::provider::{LlmProvider, RetryConfig};
    pub use crate::streaming::StreamingHandler;
    pub use crate::tool::approval::{ApprovalDecision, ToolApprovalHandler};
    pub use crate::tool::registry::ToolRegistry;
    pub use crate::tool::{Tool, ToolLifecycleHook};
}
