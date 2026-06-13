use async_trait::async_trait;
use serde_json::Value;

/// Decision returned by a [`ToolApprovalHandler`] for a pending tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Allow this invocation only; the handler is asked again next time.
    Approved,
    /// Allow this invocation and all future invocations of this tool for the
    /// lifetime of the agent. The handler is not consulted again for it.
    ApprovedForSession,
    /// Block this invocation only; the handler is asked again next time.
    Denied,
    /// Block this invocation only, returning `reason` to the model as part of
    /// the error tool result so it can adapt (e.g. "user said: use the staging
    /// database instead").
    DeniedWithReason(String),
    /// Block this invocation and all future invocations of this tool for the
    /// lifetime of the agent. The handler is not consulted again for it.
    /// Persisting the rule beyond the agent's lifetime is the handler's
    /// responsibility.
    DeniedAlways,
}

/// Gate that is consulted before every tool execution.
///
/// Implement this trait to add interactive prompts, pattern-based allow/deny
/// rules, or any other approval policy. The agent calls
/// [`approve`](Self::approve) with the tool name and its JSON arguments;
/// execution only proceeds on `Approved` or `ApprovedForSession`.
///
/// Implementations must be `Send + Sync` because the agent executes tools
/// concurrently.
#[async_trait]
pub trait ToolApprovalHandler: Send + Sync {
    async fn approve(&self, tool_name: &str, args: &Value) -> ApprovalDecision;
}
