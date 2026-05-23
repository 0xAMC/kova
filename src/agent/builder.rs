use std::sync::Arc;

use crate::error::KovaError;
use crate::mcp::McpClient;
use crate::mcp::tool::McpTool;
use crate::memory::MemoryStore;
use crate::memory::in_memory::InMemoryStore;
use crate::models::InferenceConfig;
use crate::provider::LlmProvider;
use crate::streaming::StreamingHandler;
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
/// Use either `.tool()` **or** `.tool_registry()`, not both — mixing them
/// returns `KovaError::Build` from `build()`.
pub struct AgentBuilder {
    provider: Option<Arc<dyn LlmProvider>>,
    tools: Vec<Arc<dyn Tool>>,
    tool_registry: ToolRegistry,
    registry_explicitly_set: bool,
    memory: Option<Arc<dyn MemoryStore>>,
    system_prompt: Option<String>,
    max_iterations: usize,
    max_concurrent_tools: usize,
    inference_config: InferenceConfig,
    streaming_handler: Option<Arc<dyn StreamingHandler>>,
    approval_handler: Option<Arc<dyn ToolApprovalHandler>>,
    lifecycle_hook: Option<Arc<dyn ToolLifecycleHook>>,
}

impl AgentBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            provider: None,
            tools: Vec::new(),
            tool_registry: ToolRegistry::new(),
            registry_explicitly_set: false,
            memory: None,
            system_prompt: None,
            max_iterations: 10,
            max_concurrent_tools: 10,
            inference_config: InferenceConfig::default(),
            streaming_handler: None,
            approval_handler: None,
            lifecycle_hook: None,
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

    /// Register a single tool. Cannot be combined with `.tool_registry()`.
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set a pre-built tool registry. Cannot be combined with `.tool()`.
    pub fn tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = registry;
        self.registry_explicitly_set = true;
        self
    }

    /// Set the memory store for conversation persistence.
    ///
    /// If not set, defaults to an [`InMemoryStore`] with no message limit.
    pub fn memory(mut self, memory: Arc<dyn MemoryStore>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the streaming handler for receiving response chunks in real time.
    pub fn streaming_handler(mut self, handler: Arc<dyn StreamingHandler>) -> Self {
        self.streaming_handler = Some(handler);
        self
    }

    /// Connect an MCP client, discover its tools, and register them.
    ///
    /// The discovered MCP tools are wrapped as [`McpTool`] instances and
    /// added to the agent's tool list alongside any `.tool()` registrations.
    /// Each tool receives a qualified name in the format `@{server_name}/{tool_name}`.
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
    /// - `KovaError::Build` if both `.tool()` and `.tool_registry()` are used.
    pub fn build(self) -> Result<Agent, KovaError> {
        let provider = self
            .provider
            .ok_or_else(|| KovaError::Build("LlmProvider is required".into()))?;

        if self.registry_explicitly_set && !self.tools.is_empty() {
            return Err(KovaError::Build(
                "Use either .tool() or .tool_registry(), not both".into(),
            ));
        }

        let tool_registry = if !self.tools.is_empty() {
            ToolRegistry::from_tools(self.tools)
        } else {
            self.tool_registry
        };

        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(InMemoryStore::new()));

        Ok(Agent {
            provider,
            tool_registry,
            memory,
            system_prompt: self.system_prompt,
            max_iterations: self.max_iterations,
            max_concurrent_tools: self.max_concurrent_tools,
            inference_config: self.inference_config,
            streaming_handler: self.streaming_handler,
            approval_handler: self.approval_handler,
            lifecycle_hook: self.lifecycle_hook,
        })
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
