use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use crate::error::KovaError;
use crate::mcp::McpClient;
use crate::mcp::tool::McpTool;
use crate::models::InferenceConfig;
use crate::provider::{LlmProvider, RetryConfig};
use crate::telemetry::MetricsCollector;
use crate::tool::Tool;
use crate::tool::ToolLifecycleHook;
use crate::tool::approval::ToolApprovalHandler;
use crate::tool::registry::ToolRegistry;

use super::Agent;

/// Builder for constructing an [`Agent`] with a consuming (chained) API.
///
/// The only required field is the LLM provider — calling `build()` without
/// one returns `KovaError::Build`.
///
/// `.tool()` and `.tool_registry()` compose: tools registered with `.tool()`
/// are merged into the provided registry (overwriting same-named entries).
pub struct AgentBuilder {
    provider: Option<Arc<dyn LlmProvider>>,
    tools: Vec<Arc<dyn Tool>>,
    tool_registry: ToolRegistry,
    system_prompt: Option<String>,
    max_iterations: usize,
    context_budget: Option<u32>,
    max_concurrent_tools: usize,
    inference_config: InferenceConfig,
    approval_handler: Option<Arc<dyn ToolApprovalHandler>>,
    lifecycle_hook: Option<Arc<dyn ToolLifecycleHook>>,
    metrics: Option<Arc<MetricsCollector>>,
    retry_config: RetryConfig,
}

impl AgentBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            provider: None,
            tools: Vec::new(),
            tool_registry: ToolRegistry::new(),
            system_prompt: None,
            max_iterations: 10,
            context_budget: None,
            max_concurrent_tools: 10,
            inference_config: InferenceConfig::default(),
            approval_handler: None,
            lifecycle_hook: None,
            metrics: None,
            retry_config: RetryConfig::default(),
        }
    }

    /// Set the LLM provider (required).
    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the system prompt prepended to every conversation.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum tool-call loop iterations (default: 10).
    /// Cap the assembled prompt size for every provider call in a turn.
    ///
    /// Checked with the cheap offline heuristic (`heuristic_count_tokens`)
    /// before each call — exceeding it fails the turn with
    /// [`KovaError::ContextBudgetExceeded`] instead of sending a doomed
    /// (or expensive) request. kova imposes no default; hosts supply the
    /// number (e.g. per pipeline step).
    pub fn context_budget(mut self, max_prompt_tokens: u32) -> Self {
        self.context_budget = Some(max_prompt_tokens);
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the maximum number of concurrent tool executions (default: 10).
    pub fn max_concurrent_tools(mut self, n: usize) -> Self {
        self.max_concurrent_tools = n;
        self
    }

    /// Set inference parameters (max_tokens, temperature) forwarded to the provider on every call.
    pub fn inference_config(mut self, config: InferenceConfig) -> Self {
        self.inference_config = config;
        self
    }

    /// Register a single tool.
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set a pre-built tool registry. Tools registered via `.tool()` are
    /// merged into it on `build()`.
    pub fn tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = registry;
        self
    }

    /// Connect an MCP client, discover its tools, and register them.
    ///
    /// The discovered MCP tools are wrapped as [`McpTool`] instances and
    /// added to the agent's tool list alongside any `.tool()` registrations.
    /// Each tool receives a qualified name in the format `{server_name}__{tool_name}`.
    pub async fn mcp_client(
        mut self,
        client: Arc<McpClient>,
        server_name: &str,
    ) -> Result<Self, KovaError> {
        let tool_defs = client.tools_list().await?;
        for def in tool_defs {
            self.tools.push(Arc::new(McpTool::new(
                def,
                Arc::clone(&client),
                server_name,
            )));
        }
        Ok(self)
    }

    /// Set a tool approval handler that gates every tool execution.
    ///
    /// When set, the agent calls [`ToolApprovalHandler::approve`] before
    /// executing any tool. Execution proceeds only on `Approved` or
    /// `ApprovedForSession`; denied calls return an error result to the LLM.
    pub fn with_approval_handler(mut self, handler: Arc<dyn ToolApprovalHandler>) -> Self {
        self.approval_handler = Some(handler);
        self
    }

    /// Set the retry policy for provider calls (default: 2 retries with
    /// exponential backoff on transient errors). Use
    /// [`RetryConfig::disabled`] to turn retries off.
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Register a metrics collector.
    ///
    /// When set, the agent records LLM request latency, token usage, errors,
    /// and tool execution durations into it automatically.
    pub fn metrics(mut self, collector: Arc<MetricsCollector>) -> Self {
        self.metrics = Some(collector);
        self
    }

    /// Set a lifecycle hook to observe tool execution start and end.
    ///
    /// The hook is called around every tool execution regardless of
    /// approval decisions — the `on_tool_end` callback is skipped for
    /// denied calls.
    pub fn with_lifecycle_hook(mut self, hook: Arc<dyn ToolLifecycleHook>) -> Self {
        self.lifecycle_hook = Some(hook);
        self
    }

    /// Build the agent.
    ///
    /// # Errors
    ///
    /// - `KovaError::Build` if no provider is set.
    pub fn build(self) -> Result<Agent, KovaError> {
        let provider = self
            .provider
            .ok_or_else(|| KovaError::Build("LlmProvider is required".into()))?;

        let tool_registry = self.tool_registry;
        for tool in self.tools {
            tool_registry.register(tool);
        }

        Ok(Agent {
            provider,
            tool_registry,
            system_prompt: self.system_prompt,
            max_iterations: self.max_iterations,
            context_budget: self.context_budget,
            max_concurrent_tools: self.max_concurrent_tools,
            inference_config: self.inference_config,
            approval_handler: self.approval_handler,
            lifecycle_hook: self.lifecycle_hook,
            approval_cache: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
            metrics: self.metrics,
            retry_config: self.retry_config,
            last_turn_input_tokens: Arc::new(AtomicU32::new(0)),
        })
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
