use async_trait::async_trait;
use serde_json::Value;

/// Decision returned by a [`ToolApprovalHandler`] for a pending tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Allow this invocation only.
    Approved,
    /// Allow this invocation and remember the approval for the session.
    ApprovedForSession,
    /// Block this invocation only.
    Denied,
    /// Block this invocation and persist the denial as a permanent rule.
    DeniedAlways,
}

/// Gate that is consulted before every tool execution.
///
/// Implement this trait to add interactive prompts, pattern-based allow/deny
/// rules, or any other approval policy. The agent calls [`approve`] with the
/// tool name and its JSON arguments; execution only proceeds on `Approved` or
/// `ApprovedForSession`.
///
/// Implementations must be `Send + Sync` because the agent executes tools
/// concurrently.
#[async_trait]
pub trait ToolApprovalHandler: Send + Sync {
    async fn approve(&self, tool_name: &str, args: &Value) -> ApprovalDecision;
}
