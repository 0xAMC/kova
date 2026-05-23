mod builder;
pub use builder::AgentBuilder;

use std::sync::Arc;

use futures::StreamExt;
use futures::future::join_all;
use tokio::sync::Semaphore;
use tracing::Instrument;

use crate::error::KovaError;
use crate::memory::MemoryStore;
use crate::models::{
    ContentBlock, ConversationMessage, InferenceConfig, Role, StopReason, StreamEvent,
};
use crate::provider::LlmProvider;
use crate::streaming::StreamingHandler;
use crate::tool::ToolLifecycleHook;
use crate::tool::approval::{ApprovalDecision, ToolApprovalHandler};
use crate::tool::registry::ToolRegistry;

/// An AI agent that uses an LLM provider to generate responses.
///
/// Constructed via [`AgentBuilder`]. The agent is `Send + Sync` and can be
/// safely shared across async tasks.
pub struct Agent {
    provider: Arc<dyn LlmProvider>,
    tool_registry: ToolRegistry,
    memory: Arc<dyn MemoryStore>,
    system_prompt: Option<String>,
    max_iterations: usize,
    max_concurrent_tools: usize,
    inference_config: InferenceConfig,
    streaming_handler: Option<Arc<dyn StreamingHandler>>,
    approval_handler: Option<Arc<dyn ToolApprovalHandler>>,
    lifecycle_hook: Option<Arc<dyn ToolLifecycleHook>>,
}

impl Agent {
    // ── Public API ────────────────────────────────────────────────────

    /// Send a user message and return the assistant's response text.
    ///
    /// Messages are persisted in the memory store. The full conversation
    /// history is included in every LLM request. If the LLM responds with
    /// tool calls, the agent resolves each tool, executes it, and loops
    /// until no tool calls remain or `max_iterations` is reached.
    pub async fn chat(
        &self,
        conversation_id: &str,
        user_message: &str,
    ) -> Result<String, KovaError> {
        let span = tracing::info_span!(
            "agent.chat",
            conversation_id = conversation_id,
            otel.status_code = tracing::field::Empty,
        );
        self.chat_inner(conversation_id, user_message)
            .instrument(span)
            .await
    }

    /// Send a user message and stream the assistant's response via the
    /// configured [`StreamingHandler`].
    ///
    /// Behaves like [`chat`](Self::chat) but uses the provider's streaming
    /// endpoint. Each chunk is delivered to the handler's `on_chunk` in
    /// order. Tool-call loops are handled identically to `chat`. Returns
    /// the accumulated assistant text.
    ///
    /// # Errors
    ///
    /// Returns `KovaError::Build` if no streaming handler is configured.
    pub async fn chat_stream(
        &self,
        conversation_id: &str,
        user_message: &str,
    ) -> Result<String, KovaError> {
        let span = tracing::info_span!(
            "agent.chat_stream",
            conversation_id = conversation_id,
            otel.status_code = tracing::field::Empty,
        );
        self.chat_stream_inner(conversation_id, user_message)
            .instrument(span)
            .await
    }

    // ── Non-streaming agentic loop ────────────────────────────────────

    // The first provider call is intentionally outside the loop so that
    // max_iterations bounds the number of tool-execution rounds, not LLM calls.
    // Total provider calls = max_iterations + 1 (the initial call is "free").
    async fn chat_inner(
        &self,
        conversation_id: &str,
        user_message: &str,
    ) -> Result<String, KovaError> {
        self.store_user_message(conversation_id, user_message)
            .await?;

        let tool_defs = self.tool_registry.tool_definitions().await;
        let config = self.inference_config.clone();

        let messages = self.build_messages(conversation_id).await?;
        let mut response = self
            .provider
            .chat_completion(&messages, &tool_defs, &config)
            .await?;

        for _iteration in 0..self.max_iterations {
            match response.stop_reason {
                StopReason::ToolUse => {
                    self.store_assistant_message(conversation_id, response.content.clone())
                        .await?;
                    let tool_uses = Self::extract_tool_uses(&response.content);
                    self.execute_and_store_tool_results(conversation_id, tool_uses)
                        .await?;

                    let messages = self.build_messages(conversation_id).await?;
                    response = self
                        .provider
                        .chat_completion(&messages, &tool_defs, &config)
                        .await?;
                }
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                    self.store_assistant_message(conversation_id, response.content.clone())
                        .await?;
                    return Ok(Self::collect_text(&response.content));
                }
            }
        }

        Err(KovaError::MaxIterations(self.max_iterations))
    }

    // ── Streaming agentic loop ────────────────────────────────────────

    async fn chat_stream_inner(
        &self,
        conversation_id: &str,
        user_message: &str,
    ) -> Result<String, KovaError> {
        let handler = self
            .streaming_handler
            .as_ref()
            .ok_or_else(|| KovaError::Build("StreamingHandler is required for chat_stream".into()))?
            .clone();

        self.store_user_message(conversation_id, user_message)
            .await?;

        let tool_defs = self.tool_registry.tool_definitions().await;
        let config = self.inference_config.clone();

        for _iteration in 0..=self.max_iterations {
            let messages = self.build_messages(conversation_id).await?;

            let mut stream = match self
                .provider
                .chat_completion_stream(&messages, &tool_defs, &config)
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    handler.on_error(&e).await;
                    return Err(e);
                }
            };

            let accumulated = self.consume_stream(&mut stream, &handler).await?;

            match accumulated.stop_reason {
                StopReason::ToolUse => {
                    let content_blocks =
                        Self::build_tool_use_content(&accumulated.text, &accumulated.tool_calls);
                    self.store_assistant_message(conversation_id, content_blocks)
                        .await?;

                    let tool_uses = Self::parse_streamed_tool_calls(accumulated.tool_calls);
                    self.execute_and_store_tool_results(conversation_id, tool_uses)
                        .await?;
                }
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                    self.store_assistant_message(
                        conversation_id,
                        vec![ContentBlock::Text {
                            text: accumulated.text.clone(),
                        }],
                    )
                    .await?;
                    handler.on_complete().await?;
                    return Ok(accumulated.text);
                }
            }
        }

        Err(KovaError::MaxIterations(self.max_iterations))
    }

    // ── Memory helpers ────────────────────────────────────────────────

    async fn store_message(
        &self,
        conversation_id: &str,
        role: Role,
        content: Vec<ContentBlock>,
    ) -> Result<(), KovaError> {
        self.memory
            .add_message(conversation_id, ConversationMessage { role, content })
            .await
    }

    async fn store_user_message(&self, conversation_id: &str, text: &str) -> Result<(), KovaError> {
        self.store_message(
            conversation_id,
            Role::User,
            vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        )
        .await
    }

    async fn store_assistant_message(
        &self,
        conversation_id: &str,
        content: Vec<ContentBlock>,
    ) -> Result<(), KovaError> {
        self.store_message(conversation_id, Role::Assistant, content)
            .await
    }

    /// Build the full message list: system prompt (if set) followed by memory history.
    async fn build_messages(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>, KovaError> {
        let history = self.memory.get_history(conversation_id).await?;
        let system_msgs = self.system_prompt.iter().map(|p| ConversationMessage {
            role: Role::System,
            content: vec![ContentBlock::Text { text: p.clone() }],
        });
        Ok(system_msgs.chain(history).collect())
    }

    // ── Content helpers ───────────────────────────────────────────────

    fn extract_tool_uses(content: &[ContentBlock]) -> Vec<(String, String, serde_json::Value)> {
        content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    Some((id.clone(), name.clone(), input.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn collect_text(content: &[ContentBlock]) -> String {
        content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Parse a JSON string into a tool input `Value`, falling back to `{}` on failure.
    fn parse_tool_input(json_str: &str) -> serde_json::Value {
        serde_json::from_str(json_str)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
    }

    fn build_tool_use_content(text: &str, tool_calls: &[StreamedToolCall]) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();
        if !text.is_empty() {
            blocks.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }
        for call in tool_calls {
            blocks.push(ContentBlock::ToolUse {
                id: call.id.clone(),
                name: call.name.clone(),
                input: Self::parse_tool_input(&call.input_json),
            });
        }
        blocks
    }

    fn parse_streamed_tool_calls(
        calls: Vec<StreamedToolCall>,
    ) -> Vec<(String, String, serde_json::Value)> {
        calls
            .into_iter()
            .map(|c| (c.id, c.name, Self::parse_tool_input(&c.input_json)))
            .collect()
    }

    // ── Tool execution ────────────────────────────────────────────────

    async fn execute_and_store_tool_results(
        &self,
        conversation_id: &str,
        tool_uses: Vec<(String, String, serde_json::Value)>,
    ) -> Result<(), KovaError> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_tools));
        let registry = self.tool_registry.clone();
        let approval_handler = self.approval_handler.clone();
        let lifecycle_hook = self.lifecycle_hook.clone();

        let futures: Vec<_> = tool_uses
            .into_iter()
            .map(|(tool_use_id, tool_name, input)| {
                let sem = Arc::clone(&semaphore);
                let reg = registry.clone();
                let approval = approval_handler.clone();
                let hook = lifecycle_hook.clone();
                let span = tracing::info_span!(
                    "tool.execute",
                    tool.name = %tool_name,
                    otel.status_code = tracing::field::Empty,
                );
                async move {
                    let _permit = sem.acquire().await.expect("semaphore closed");
                    let start = std::time::Instant::now();

                    let (result_content, is_error) = match reg.get(&tool_name).await {
                        None => {
                            tracing::Span::current().record("otel.status_code", "ERROR");
                            tracing::warn!(tool.name = %tool_name, "Tool not found");
                            (format!("Tool not found: {}", tool_name), true)
                        }
                        Some(tool) => {
                            let denied = if let Some(handler) = &approval {
                                matches!(
                                    handler.approve(&tool_name, &input).await,
                                    ApprovalDecision::Denied | ApprovalDecision::DeniedAlways
                                )
                            } else {
                                false
                            };

                            if denied {
                                tracing::Span::current().record("otel.status_code", "ERROR");
                                tracing::info!(tool.name = %tool_name, "Tool execution denied");
                                (format!("Tool execution denied: {}", tool_name), true)
                            } else {
                                if let Some(ref h) = hook {
                                    h.on_tool_start(&tool_name, &input).await;
                                }
                                let (content, err_flag) = match tool.execute(input).await {
                                    Ok(result) => (result.content, result.is_error),
                                    Err(err) => {
                                        tracing::Span::current().record("otel.status_code", "ERROR");
                                        tracing::warn!(tool.name = %tool_name, error = %err, "Tool execution failed");
                                        (err.to_string(), true)
                                    }
                                };
                                if let Some(ref h) = hook {
                                    h.on_tool_end(&tool_name, &content, err_flag).await;
                                }
                                (content, err_flag)
                            }
                        }
                    };

                    let duration_ms = start.elapsed().as_millis() as u64;
                    tracing::info!(tool.name = %tool_name, duration_ms, success = !is_error, "Tool execution complete");
                    (tool_use_id, result_content, is_error)
                }
                .instrument(span)
            })
            .collect();

        let results = join_all(futures).await;

        for (tool_use_id, result_content, is_error) in results {
            self.store_message(
                conversation_id,
                Role::Tool,
                vec![ContentBlock::ToolResult {
                    tool_use_id,
                    content: result_content,
                    is_error,
                }],
            )
            .await?;
        }

        Ok(())
    }

    // ── Streaming helpers ─────────────────────────────────────────────

    async fn consume_stream(
        &self,
        stream: &mut (impl futures::Stream<Item = Result<StreamEvent, KovaError>> + Unpin),
        handler: &Arc<dyn StreamingHandler>,
    ) -> Result<StreamAccumulator, KovaError> {
        let mut acc = StreamAccumulator::default();

        while let Some(event_result) = stream.next().await {
            match event_result {
                Ok(event) => {
                    if let Err(e) = handler.on_chunk(&event).await {
                        handler.on_error(&e).await;
                        return Err(e);
                    }

                    match &event {
                        StreamEvent::ContentDelta { text } => acc.text.push_str(text),
                        StreamEvent::ToolUseDelta {
                            id,
                            name,
                            input_delta,
                        } => {
                            if !id.is_empty() {
                                acc.tool_calls.push(StreamedToolCall {
                                    id: id.clone(),
                                    name: name.clone().unwrap_or_default(),
                                    input_json: String::new(),
                                });
                            } else if let (Some(last), Some(delta)) =
                                (acc.tool_calls.last_mut(), input_delta)
                            {
                                last.input_json.push_str(delta);
                            }
                        }
                        StreamEvent::StopEvent { stop_reason } => {
                            acc.stop_reason = stop_reason.clone();
                        }
                        StreamEvent::Error { message } => {
                            let err = KovaError::Stream(message.clone());
                            handler.on_error(&err).await;
                            return Err(err);
                        }
                    }
                }
                Err(e) => {
                    handler.on_error(&e).await;
                    return Err(e);
                }
            }
        }

        Ok(acc)
    }
}

// ── Private streaming types ───────────────────────────────────────────

/// A single tool call accumulated from streaming deltas.
struct StreamedToolCall {
    id: String,
    name: String,
    input_json: String,
}

/// State accumulated while consuming a streaming response.
struct StreamAccumulator {
    text: String,
    tool_calls: Vec<StreamedToolCall>,
    stop_reason: StopReason,
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self {
            text: String::new(),
            tool_calls: Vec::new(),
            stop_reason: StopReason::EndTurn,
        }
    }
}
