mod builder;
pub use builder::AgentBuilder;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use futures::StreamExt;
use futures::future::join_all;
use tokio::sync::Semaphore;
use tracing::Instrument;

use crate::error::KovaError;
use crate::memory::MemoryStore;
use crate::models::{
    ContentBlock, ConversationMessage, InferenceConfig, ModelResponse, Role, StopReason,
    StreamEvent, ToolDefinition, UsageStats,
};
use crate::provider::{LlmProvider, RetryConfig};
use crate::streaming::StreamingHandler;
use crate::telemetry::MetricsCollector;
use crate::tool::ToolLifecycleHook;
use crate::tool::approval::{ApprovalDecision, ToolApprovalHandler};
use crate::tool::registry::ToolRegistry;

/// The result of one complete agentic turn.
///
/// Produced by [`Agent::run`] (stateless) and [`Agent::chat_response`]
/// (session-backed). `new_messages` carries everything the turn appended to
/// the conversation — assistant messages (including tool-use blocks) and
/// tool results — so stateless callers can persist it to their own history.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentResponse {
    /// Final assistant text.
    pub text: String,
    /// All messages produced during the turn, in order. Does not include
    /// the caller's input messages.
    pub new_messages: Vec<ConversationMessage>,
    /// Stop reason of the final provider response.
    pub stop_reason: StopReason,
    /// Token usage summed across every provider call in the turn.
    /// All zeros when the provider does not report usage.
    pub usage: UsageStats,
    /// Number of provider calls made during the turn.
    pub llm_calls: u64,
    /// Chain-of-thought text from the final response, if the model produced
    /// any. Never part of `new_messages`.
    pub thinking: Option<String>,
}

/// Events yielded by [`Agent::run_stream`] while a turn executes.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    /// Incremental assistant text.
    TextDelta { text: String },
    /// Incremental chain-of-thought text from thinking models.
    ThinkingDelta { text: String },
    /// The model requested a tool invocation. Emitted once the call's
    /// arguments are fully accumulated, before execution begins.
    ToolCallStarted {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool invocation finished (successfully or with an error result).
    ToolCallFinished {
        id: String,
        name: String,
        result: String,
        is_error: bool,
    },
    /// The turn finished; carries the complete [`AgentResponse`].
    TurnCompleted { response: AgentResponse },
}

/// An AI agent that uses an LLM provider to generate responses.
///
/// Constructed via [`AgentBuilder`]. The agent is `Send + Sync` and can be
/// safely shared across async tasks.
///
/// The core primitive is [`run`](Self::run): caller-supplied history in,
/// [`AgentResponse`] out, no hidden state. [`chat`](Self::chat) and
/// [`chat_response`](Self::chat_response) layer conversation persistence on
/// top via the configured [`MemoryStore`].
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
    /// Remembered `ApprovedForSession` / `DeniedAlways` decisions, keyed by
    /// tool name. `true` = approved for the rest of this agent's lifetime,
    /// `false` = denied without re-asking the handler.
    approval_cache: Arc<std::sync::RwLock<std::collections::HashMap<String, bool>>>,
    metrics: Option<Arc<MetricsCollector>>,
    retry_config: RetryConfig,
    /// Input tokens reported by the provider for the most recent completed turn.
    last_turn_input_tokens: Arc<AtomicU32>,
}

impl Agent {
    // ── Public API ────────────────────────────────────────────────────

    /// Input tokens reported by the provider for the most recently completed turn.
    ///
    /// Returns 0 until the first successful turn completes or if the provider
    /// does not report usage statistics. Updated atomically after every turn.
    pub fn last_turn_input_tokens(&self) -> u32 {
        self.last_turn_input_tokens.load(Ordering::Relaxed)
    }

    /// Run one agentic turn over caller-supplied conversation history.
    ///
    /// This is the stateless core primitive: nothing is read from or written
    /// to the memory store. `messages` is the conversation so far (ending
    /// with the latest user message); the returned
    /// [`AgentResponse::new_messages`] contains everything the turn produced,
    /// for the caller to append to their own history.
    ///
    /// The configured system prompt is prepended automatically. If the LLM
    /// responds with tool calls, the agent resolves each tool, executes it
    /// (concurrently, bounded by `max_concurrent_tools`), and loops until no
    /// tool calls remain or `max_iterations` tool rounds are exhausted.
    pub async fn run(&self, messages: &[ConversationMessage]) -> Result<AgentResponse, KovaError> {
        let span = tracing::info_span!(
            "agent.run",
            otel.status_code = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
            llm.iterations = tracing::field::Empty,
        );
        self.run_inner(messages, self.inference_config.clone())
            .instrument(span)
            .await
    }

    /// Like [`run`](Self::run), with per-call inference overrides.
    ///
    /// Fields set in `overrides` replace the agent's configured
    /// [`InferenceConfig`] for this turn only; unset fields fall back to the
    /// agent's defaults.
    pub async fn run_with_config(
        &self,
        messages: &[ConversationMessage],
        overrides: InferenceConfig,
    ) -> Result<AgentResponse, KovaError> {
        let span = tracing::info_span!(
            "agent.run",
            otel.status_code = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
            llm.iterations = tracing::field::Empty,
        );
        let config = InferenceConfig {
            model: overrides
                .model
                .or_else(|| self.inference_config.model.clone()),
            max_tokens: overrides.max_tokens.or(self.inference_config.max_tokens),
            temperature: overrides.temperature.or(self.inference_config.temperature),
            top_p: overrides.top_p.or(self.inference_config.top_p),
            stop_sequences: overrides
                .stop_sequences
                .or_else(|| self.inference_config.stop_sequences.clone()),
        };
        self.run_inner(messages, config).instrument(span).await
    }

    /// Run one agentic turn over caller-supplied history, yielding
    /// [`AgentEvent`]s as the turn progresses.
    ///
    /// The pull-based sibling of [`run`](Self::run): text and thinking
    /// arrive as deltas, tool execution is announced via
    /// `ToolCallStarted`/`ToolCallFinished`, and the final event is
    /// `TurnCompleted` carrying the full [`AgentResponse`] (including
    /// `new_messages` for the caller to persist). No `StreamingHandler`
    /// needs to be configured and nothing touches the memory store.
    ///
    /// ```ignore
    /// let mut events = std::pin::pin!(agent.run_stream(&history));
    /// while let Some(event) = events.next().await {
    ///     match event? {
    ///         AgentEvent::TextDelta { text } => print!("{text}"),
    ///         AgentEvent::TurnCompleted { response } => history.extend(response.new_messages),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn run_stream<'a>(
        &'a self,
        messages: &'a [ConversationMessage],
    ) -> impl futures::Stream<Item = Result<AgentEvent, KovaError>> + Send + 'a {
        async_stream::try_stream! {
            let tool_defs = self.tool_registry.tool_definitions();
            let config = self.inference_config.clone();

            let mut working = self.seed_messages(messages);
            let new_start = working.len();
            let mut usage = UsageStats {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
            thinking_tokens: None,
            };
            for iteration in 0..=self.max_iterations {
                let llm_calls = iteration as u64 + 1;

                let start = std::time::Instant::now();
                let stream_result = self
                    .open_stream_with_retry(&working, &tool_defs, &config)
                    .await;
                if stream_result.is_err()
                    && let Some(m) = &self.metrics
                {
                    m.record_llm_error();
                }
                let mut stream = stream_result?;

                let mut acc = StreamAccumulator::default();
                while let Some(event_result) = stream.next().await {
                    match event_result {
                        Ok(event) => match &event {
                            StreamEvent::ContentDelta { text } => {
                                acc.text.push_str(text);
                                if !text.is_empty() {
                                    yield AgentEvent::TextDelta { text: text.clone() };
                                }
                            }
                            StreamEvent::ThinkingDelta { text } => {
                                if !text.is_empty() {
                                    yield AgentEvent::ThinkingDelta { text: text.clone() };
                                }
                            }
                            StreamEvent::ToolUseDelta {
                                id,
                                name,
                                input_delta,
                                provider_metadata,
                                index,
                            } => {
                                acc.merge_tool_use_delta(
                                    id,
                                    name.as_deref(),
                                    input_delta.as_deref(),
                                    provider_metadata,
                                    *index,
                                );
                            }
                            StreamEvent::StopEvent { stop_reason } => {
                                acc.stop_reason = stop_reason.clone();
                            }
                            StreamEvent::UsageEvent {
                                input_tokens,
                                output_tokens,
                                thinking_tokens,
                            } => {
                                acc.input_tokens = Some(*input_tokens);
                                acc.output_tokens = Some(*output_tokens);
                                acc.thinking_tokens = *thinking_tokens;
                            }
                            StreamEvent::Error { message } => {
                                if let Some(m) = &self.metrics {
                                    m.record_llm_error();
                                }
                                Err(KovaError::Stream(message.clone()))?;
                            }
                        },
                        Err(e) => {
                            if let Some(m) = &self.metrics {
                                m.record_llm_error();
                            }
                            Err(e)?;
                        }
                    }
                }
                if !acc.tool_calls.is_empty() {
                    acc.stop_reason = StopReason::ToolUse;
                }

                if let Some(m) = &self.metrics {
                    m.record_llm_request(
                        start.elapsed().as_secs_f64() * 1000.0,
                        acc.input_tokens.unwrap_or(0) as u64,
                        acc.output_tokens.unwrap_or(0) as u64,
                    );
                }
                Self::accumulate_usage(
                    &mut usage,
                    Some(&UsageStats {
                        input_tokens: acc.input_tokens.unwrap_or(0),
                        output_tokens: acc.output_tokens.unwrap_or(0),
                        total_tokens: acc.input_tokens.unwrap_or(0)
                            + acc.output_tokens.unwrap_or(0),
                        thinking_tokens: acc.thinking_tokens,
                    }),
                );

                match acc.stop_reason {
                    StopReason::ToolUse => {
                        let content_blocks =
                            Self::build_tool_use_content(&acc.text, &acc.tool_calls);
                        working.push(ConversationMessage {
                            role: Role::Assistant,
                            content: content_blocks,
                        });

                        let tool_uses = Self::parse_streamed_tool_calls(acc.tool_calls);
                        let names: std::collections::HashMap<String, String> = tool_uses
                            .iter()
                            .map(|(id, name, _)| (id.clone(), name.clone()))
                            .collect();
                        for (id, name, input) in &tool_uses {
                            yield AgentEvent::ToolCallStarted {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            };
                        }
                        let tool_messages = self.execute_tools(tool_uses).await;
                        for msg in &tool_messages {
                            for block in &msg.content {
                                if let ContentBlock::ToolResult {
                                    tool_use_id,
                                    content,
                                    is_error,
                                } = block
                                {
                                    yield AgentEvent::ToolCallFinished {
                                        id: tool_use_id.clone(),
                                        name: names
                                            .get(tool_use_id)
                                            .cloned()
                                            .unwrap_or_default(),
                                        result: content.clone(),
                                        is_error: *is_error,
                                    };
                                }
                            }
                        }
                        working.extend(tool_messages);
                    }
                    StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                        if let Some(tokens) = acc.input_tokens {
                            self.last_turn_input_tokens.store(tokens, Ordering::Relaxed);
                        }
                        working.push(ConversationMessage {
                            role: Role::Assistant,
                            content: vec![ContentBlock::Text {
                                text: acc.text.clone(),
                            }],
                        });
                        yield AgentEvent::TurnCompleted {
                            response: AgentResponse {
                                text: acc.text,
                                new_messages: working.split_off(new_start),
                                stop_reason: acc.stop_reason,
                                usage,
                                llm_calls,
                                thinking: None,
                            },
                        };
                        return;
                    }
                }
            }

            Err(KovaError::MaxIterations(self.max_iterations))?;
        }
    }

    /// Send a user message in a persisted conversation and return the
    /// assistant's response text.
    ///
    /// Convenience wrapper over [`run`](Self::run) backed by the configured
    /// [`MemoryStore`]: history is loaded, the turn is executed, and — only
    /// if the turn succeeds — the user message and all produced messages are
    /// persisted. A failed turn leaves the conversation unchanged.
    pub async fn chat(
        &self,
        conversation_id: &str,
        user_message: &str,
    ) -> Result<String, KovaError> {
        Ok(self
            .chat_response(conversation_id, user_message)
            .await?
            .text)
    }

    /// Like [`chat`](Self::chat), but returns the full [`AgentResponse`]
    /// (usage, stop reason, produced messages) instead of just the text.
    pub async fn chat_response(
        &self,
        conversation_id: &str,
        user_message: &str,
    ) -> Result<AgentResponse, KovaError> {
        let span = tracing::info_span!(
            "agent.chat",
            conversation_id = conversation_id,
            otel.status_code = tracing::field::Empty,
        );
        async {
            let mut history = self.memory.get_history(conversation_id).await?;
            let user_msg = Self::user_message(user_message);
            history.push(user_msg.clone());

            let response = self
                .run_inner(&history, self.inference_config.clone())
                .await?;

            self.persist_turn(conversation_id, user_msg, &response.new_messages)
                .await?;
            Ok(response)
        }
        .instrument(span)
        .await
    }

    /// Send a user message and stream the assistant's response via the
    /// configured [`StreamingHandler`].
    ///
    /// Behaves like [`chat`](Self::chat) but uses the provider's streaming
    /// endpoint. Each chunk is delivered to the handler's `on_chunk` in
    /// order. Tool-call loops are handled identically to `chat`. Returns
    /// the accumulated assistant text. Like `chat`, memory is only written
    /// when the whole turn succeeds.
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
            llm.stop_reason = tracing::field::Empty,
            llm.iterations = tracing::field::Empty,
        );
        async {
            let handler = self
                .streaming_handler
                .as_ref()
                .ok_or_else(|| {
                    KovaError::Build("StreamingHandler is required for chat_stream".into())
                })?
                .clone();

            let mut history = self.memory.get_history(conversation_id).await?;
            let user_msg = Self::user_message(user_message);
            history.push(user_msg.clone());

            let response = self.run_stream_inner(&history, &handler).await?;

            self.persist_turn(conversation_id, user_msg, &response.new_messages)
                .await?;
            Ok(response.text)
        }
        .instrument(span)
        .await
    }

    // ── Non-streaming agentic loop ────────────────────────────────────

    // The first provider call is intentionally outside the loop so that
    // max_iterations bounds the number of tool-execution rounds, not LLM calls.
    // Total provider calls = max_iterations + 1 (the initial call is "free").
    async fn run_inner(
        &self,
        messages: &[ConversationMessage],
        config: InferenceConfig,
    ) -> Result<AgentResponse, KovaError> {
        let tool_defs = self.tool_registry.tool_definitions();

        // Working set: system prompt + caller history + everything this turn
        // produces. Messages past `new_start` become `new_messages`.
        let mut working = self.seed_messages(messages);
        let new_start = working.len();
        let mut usage = UsageStats {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            thinking_tokens: None,
        };

        let mut response = self
            .timed_chat_completion(&working, &tool_defs, &config)
            .await?;
        Self::accumulate_usage(&mut usage, response.usage.as_ref());

        let mut llm_calls: u64 = 1;

        for _iteration in 0..self.max_iterations {
            match response.stop_reason {
                StopReason::ToolUse => {
                    working.push(ConversationMessage {
                        role: Role::Assistant,
                        content: response.content.clone(),
                    });
                    let tool_uses = Self::extract_tool_uses(&response.content);
                    let tool_messages = self.execute_tools(tool_uses).await;
                    working.extend(tool_messages);

                    response = self
                        .timed_chat_completion(&working, &tool_defs, &config)
                        .await?;
                    Self::accumulate_usage(&mut usage, response.usage.as_ref());
                    llm_calls += 1;
                }
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                    let span = tracing::Span::current();
                    span.record("llm.stop_reason", response.stop_reason.as_str());
                    span.record("llm.iterations", llm_calls);
                    if let Some(u) = &response.usage {
                        self.last_turn_input_tokens
                            .store(u.input_tokens, Ordering::Relaxed);
                    }
                    // Emit thinking content via the streaming handler so it displays
                    // in the terminal even in non-streaming mode.
                    if let (Some(thinking_text), Some(handler)) =
                        (&response.thinking, &self.streaming_handler)
                    {
                        let _ = handler
                            .on_chunk(&StreamEvent::ThinkingDelta {
                                text: thinking_text.clone(),
                            })
                            .await;
                        // Empty ContentDelta closes the thinking box without printing anything.
                        let _ = handler
                            .on_chunk(&StreamEvent::ContentDelta {
                                text: String::new(),
                            })
                            .await;
                    }
                    let text = Self::collect_text(&response.content);
                    working.push(ConversationMessage {
                        role: Role::Assistant,
                        content: response.content,
                    });
                    return Ok(AgentResponse {
                        text,
                        new_messages: working.split_off(new_start),
                        stop_reason: response.stop_reason,
                        usage,
                        llm_calls,
                        thinking: response.thinking,
                    });
                }
            }
        }

        Err(KovaError::MaxIterations(self.max_iterations))
    }

    // ── Streaming agentic loop ────────────────────────────────────────

    async fn run_stream_inner(
        &self,
        messages: &[ConversationMessage],
        handler: &Arc<dyn StreamingHandler>,
    ) -> Result<AgentResponse, KovaError> {
        let tool_defs = self.tool_registry.tool_definitions();
        let config = self.inference_config.clone();

        let mut working = self.seed_messages(messages);
        let new_start = working.len();
        let mut usage = UsageStats {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            thinking_tokens: None,
        };

        let mut llm_calls: u64 = 0;

        for _iteration in 0..=self.max_iterations {
            llm_calls += 1;

            let call_span = tracing::info_span!(
                "llm.chat_completion_stream",
                otel.status_code = tracing::field::Empty,
                llm.input_tokens = tracing::field::Empty,
                llm.output_tokens = tracing::field::Empty,
                llm.stop_reason = tracing::field::Empty,
            );
            let handler_ref = handler.clone();
            let tool_defs_ref = tool_defs.clone();
            let config_ref = config.clone();
            let accumulated = {
                let request_messages = &working;
                async move {
                    let start = std::time::Instant::now();
                    let mut stream = match self
                        .provider
                        .chat_completion_stream(request_messages, &tool_defs_ref, &config_ref)
                        .await
                    {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::Span::current().record("otel.status_code", "ERROR");
                            if let Some(m) = &self.metrics {
                                m.record_llm_error();
                            }
                            handler_ref.on_error(&e).await;
                            return Err(e);
                        }
                    };
                    let acc = match self.consume_stream(&mut stream, &handler_ref).await {
                        Ok(acc) => acc,
                        Err(e) => {
                            if let Some(m) = &self.metrics {
                                m.record_llm_error();
                            }
                            return Err(e);
                        }
                    };
                    if let Some(m) = &self.metrics {
                        m.record_llm_request(
                            start.elapsed().as_secs_f64() * 1000.0,
                            acc.input_tokens.unwrap_or(0) as u64,
                            acc.output_tokens.unwrap_or(0) as u64,
                        );
                    }
                    if let Some(t) = acc.input_tokens {
                        tracing::Span::current().record("llm.input_tokens", t);
                    }
                    if let Some(t) = acc.output_tokens {
                        tracing::Span::current().record("llm.output_tokens", t);
                    }
                    tracing::Span::current().record("llm.stop_reason", acc.stop_reason.as_str());
                    Ok(acc)
                }
                .instrument(call_span)
                .await?
            };

            if accumulated.input_tokens.is_some() || accumulated.output_tokens.is_some() {
                Self::accumulate_usage(
                    &mut usage,
                    Some(&UsageStats {
                        input_tokens: accumulated.input_tokens.unwrap_or(0),
                        output_tokens: accumulated.output_tokens.unwrap_or(0),
                        total_tokens: accumulated.input_tokens.unwrap_or(0)
                            + accumulated.output_tokens.unwrap_or(0),
                        thinking_tokens: accumulated.thinking_tokens,
                    }),
                );
            }

            match accumulated.stop_reason {
                StopReason::ToolUse => {
                    let content_blocks =
                        Self::build_tool_use_content(&accumulated.text, &accumulated.tool_calls);
                    working.push(ConversationMessage {
                        role: Role::Assistant,
                        content: content_blocks,
                    });

                    let tool_uses = Self::parse_streamed_tool_calls(accumulated.tool_calls);
                    let tool_messages = self.execute_tools(tool_uses).await;
                    working.extend(tool_messages);
                }
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                    let span = tracing::Span::current();
                    span.record("llm.stop_reason", accumulated.stop_reason.as_str());
                    span.record("llm.iterations", llm_calls);
                    if let Some(tokens) = accumulated.input_tokens {
                        self.last_turn_input_tokens.store(tokens, Ordering::Relaxed);
                    }
                    working.push(ConversationMessage {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: accumulated.text.clone(),
                        }],
                    });
                    handler.on_complete().await?;
                    return Ok(AgentResponse {
                        text: accumulated.text,
                        new_messages: working.split_off(new_start),
                        stop_reason: accumulated.stop_reason,
                        usage,
                        llm_calls,
                        thinking: None,
                    });
                }
            }
        }

        Err(KovaError::MaxIterations(self.max_iterations))
    }

    // ── Provider helpers ──────────────────────────────────────────────

    /// Call the provider, retrying transient failures per the configured
    /// [`RetryConfig`], and record latency/token/error metrics when a
    /// `MetricsCollector` is registered.
    async fn timed_chat_completion(
        &self,
        messages: &[ConversationMessage],
        tool_defs: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let mut attempt: u32 = 0;
        loop {
            let start = std::time::Instant::now();
            let result = self
                .provider
                .chat_completion(messages, tool_defs, config)
                .await;
            if let Some(metrics) = &self.metrics {
                match &result {
                    Ok(response) => {
                        let (input, output) = response
                            .usage
                            .as_ref()
                            .map(|u| (u.input_tokens as u64, u.output_tokens as u64))
                            .unwrap_or((0, 0));
                        metrics.record_llm_request(
                            start.elapsed().as_secs_f64() * 1000.0,
                            input,
                            output,
                        );
                    }
                    Err(_) => metrics.record_llm_error(),
                }
            }
            match result {
                Err(e) if e.is_retryable() && attempt < self.retry_config.max_retries => {
                    let delay = self.retry_config.backoff(attempt);
                    tracing::warn!(
                        error = %e,
                        attempt = attempt + 1,
                        delay_ms = delay.as_millis() as u64,
                        "Retrying provider call after transient failure"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                other => return other,
            }
        }
    }

    /// Open a streaming response, retrying transient failures to *establish*
    /// the stream. Mid-stream failures are never retried.
    async fn open_stream_with_retry(
        &self,
        messages: &[ConversationMessage],
        tool_defs: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, KovaError>> + Send>>,
        KovaError,
    > {
        let mut attempt: u32 = 0;
        loop {
            match self
                .provider
                .chat_completion_stream(messages, tool_defs, config)
                .await
            {
                Err(e) if e.is_retryable() && attempt < self.retry_config.max_retries => {
                    if let Some(m) = &self.metrics {
                        m.record_llm_error();
                    }
                    let delay = self.retry_config.backoff(attempt);
                    tracing::warn!(
                        error = %e,
                        attempt = attempt + 1,
                        delay_ms = delay.as_millis() as u64,
                        "Retrying provider stream after transient failure"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                other => return other,
            }
        }
    }

    // ── Session helpers ───────────────────────────────────────────────

    fn user_message(text: &str) -> ConversationMessage {
        ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    /// Persist a completed turn: the user message followed by everything the
    /// turn produced. Called only after the turn succeeds, so a failed turn
    /// never leaves partial state (dangling user messages, orphaned
    /// tool-use blocks) in the store.
    async fn persist_turn(
        &self,
        conversation_id: &str,
        user_msg: ConversationMessage,
        new_messages: &[ConversationMessage],
    ) -> Result<(), KovaError> {
        self.memory.add_message(conversation_id, user_msg).await?;
        for msg in new_messages {
            self.memory
                .add_message(conversation_id, msg.clone())
                .await?;
        }
        Ok(())
    }

    /// Build the request prefix: system prompt (if set) followed by the
    /// caller-supplied conversation.
    fn seed_messages(&self, messages: &[ConversationMessage]) -> Vec<ConversationMessage> {
        let system = self.system_prompt.iter().map(|p| ConversationMessage {
            role: Role::System,
            content: vec![ContentBlock::Text { text: p.clone() }],
        });
        system.chain(messages.iter().cloned()).collect()
    }

    fn accumulate_usage(acc: &mut UsageStats, usage: Option<&UsageStats>) {
        if let Some(u) = usage {
            acc.input_tokens += u.input_tokens;
            acc.output_tokens += u.output_tokens;
            acc.total_tokens += u.total_tokens;
            // `None` from both sides stays `None` ("unknown"); any reported
            // value makes the running total `Some` and adds in.
            if acc.thinking_tokens.is_some() || u.thinking_tokens.is_some() {
                acc.thinking_tokens =
                    Some(acc.thinking_tokens.unwrap_or(0) + u.thinking_tokens.unwrap_or(0));
            }
        }
    }

    // ── Content helpers ───────────────────────────────────────────────

    fn extract_tool_uses(content: &[ContentBlock]) -> Vec<(String, String, serde_json::Value)> {
        content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse {
                    id, name, input, ..
                } = block
                {
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
                provider_metadata: call.provider_metadata.clone(),
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

    /// Execute the requested tools concurrently (bounded by
    /// `max_concurrent_tools`) and return their results as `Role::Tool`
    /// messages in the same order as `tool_uses`. Tool failures become
    /// error-flagged results for the LLM, never `Err`.
    async fn execute_tools(
        &self,
        tool_uses: Vec<(String, String, serde_json::Value)>,
    ) -> Vec<ConversationMessage> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_tools));
        let registry = self.tool_registry.clone();
        let approval_handler = self.approval_handler.clone();
        let lifecycle_hook = self.lifecycle_hook.clone();
        let approval_cache = Arc::clone(&self.approval_cache);
        let metrics_collector = self.metrics.clone();

        let futures: Vec<_> = tool_uses
            .into_iter()
            .map(|(tool_use_id, tool_name, input)| {
                let sem = Arc::clone(&semaphore);
                let reg = registry.clone();
                let approval = approval_handler.clone();
                let hook = lifecycle_hook.clone();
                let cache = Arc::clone(&approval_cache);
                let metrics = metrics_collector.clone();
                let span = tracing::info_span!(
                    "tool.execute",
                    tool.name = %tool_name,
                    otel.status_code = tracing::field::Empty,
                );
                async move {
                    // The semaphore lives for the duration of this call and is
                    // never closed; if acquisition fails anyway, surface an
                    // error result to the LLM instead of panicking.
                    let permit = sem.acquire().await;
                    if permit.is_err() {
                        tracing::Span::current().record("otel.status_code", "ERROR");
                        return (
                            tool_use_id,
                            format!("Tool execution unavailable: {}", tool_name),
                            true,
                        );
                    }
                    let start = std::time::Instant::now();

                    let (result_content, is_error) = match reg.get(&tool_name) {
                        None => {
                            tracing::Span::current().record("otel.status_code", "ERROR");
                            tracing::warn!(tool.name = %tool_name, "Tool not found");
                            (format!("Tool not found: {}", tool_name), true)
                        }
                        Some(tool) => {
                            // None = approved; Some(reason) = denied with the
                            // message to surface to the model.
                            let denial: Option<String> = if let Some(handler) = &approval {
                                let cached = cache
                                    .read()
                                    .ok()
                                    .and_then(|c| c.get(&tool_name).copied());
                                match cached {
                                    Some(true) => None,
                                    Some(false) => {
                                        Some(format!("Tool execution denied: {}", tool_name))
                                    }
                                    None => match handler.approve(&tool_name, &input).await {
                                        ApprovalDecision::Approved => None,
                                        ApprovalDecision::ApprovedForSession => {
                                            if let Ok(mut c) = cache.write() {
                                                c.insert(tool_name.clone(), true);
                                            }
                                            None
                                        }
                                        ApprovalDecision::Denied => {
                                            Some(format!("Tool execution denied: {}", tool_name))
                                        }
                                        ApprovalDecision::DeniedWithReason(reason) => Some(
                                            format!(
                                                "Tool execution denied: {}. Reason: {}",
                                                tool_name, reason
                                            ),
                                        ),
                                        ApprovalDecision::DeniedAlways => {
                                            if let Ok(mut c) = cache.write() {
                                                c.insert(tool_name.clone(), false);
                                            }
                                            Some(format!("Tool execution denied: {}", tool_name))
                                        }
                                    },
                                }
                            } else {
                                None
                            };

                            if let Some(denial_msg) = denial {
                                tracing::Span::current().record("otel.status_code", "ERROR");
                                tracing::info!(tool.name = %tool_name, "Tool execution denied");
                                (denial_msg, true)
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

                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(m) = &metrics {
                        m.record_tool_invocation(duration_ms, !is_error);
                    }
                    let duration_ms = duration_ms as u64;
                    tracing::info!(tool.name = %tool_name, duration_ms, success = !is_error, "Tool execution complete");
                    (tool_use_id, result_content, is_error)
                }
                .instrument(span)
            })
            .collect();

        let results = join_all(futures).await;

        results
            .into_iter()
            .map(
                |(tool_use_id, result_content, is_error)| ConversationMessage {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id,
                        content: result_content,
                        is_error,
                    }],
                },
            )
            .collect()
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
                            provider_metadata,
                            index,
                        } => {
                            acc.merge_tool_use_delta(
                                id,
                                name.as_deref(),
                                input_delta.as_deref(),
                                provider_metadata,
                                *index,
                            );
                        }
                        StreamEvent::StopEvent { stop_reason } => {
                            acc.stop_reason = stop_reason.clone();
                        }
                        StreamEvent::UsageEvent {
                            input_tokens,
                            output_tokens,
                            thinking_tokens,
                        } => {
                            acc.input_tokens = Some(*input_tokens);
                            acc.output_tokens = Some(*output_tokens);
                            acc.thinking_tokens = *thinking_tokens;
                        }
                        StreamEvent::ThinkingDelta { .. } => {}
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

        // Gemini sends finishReason in the *last* chunk, which may not contain
        // function calls. Prefer content: if we accumulated any tool calls,
        // the stop reason must be ToolUse regardless of what StopEvent said.
        if !acc.tool_calls.is_empty() {
            acc.stop_reason = StopReason::ToolUse;
        }

        Ok(acc)
    }
}

// ── Private streaming types ───────────────────────────────────────────

/// A single tool call accumulated from streaming deltas.
struct StreamedToolCall {
    index: Option<u32>,
    id: String,
    name: String,
    input_json: String,
    provider_metadata: Option<serde_json::Value>,
}

/// State accumulated while consuming a streaming response.
struct StreamAccumulator {
    text: String,
    tool_calls: Vec<StreamedToolCall>,
    stop_reason: StopReason,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    thinking_tokens: Option<u32>,
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self {
            text: String::new(),
            tool_calls: Vec::new(),
            stop_reason: StopReason::EndTurn,
            input_tokens: None,
            output_tokens: None,
            thinking_tokens: None,
        }
    }
}

impl StreamAccumulator {
    /// Merge a `ToolUseDelta` into the accumulated tool calls.
    ///
    /// Correlation, in priority order:
    /// 1. provider `index` (OpenAI / Bedrock) — deltas for one call share it
    /// 2. `id` — for providers that repeat the id on every delta
    /// 3. fall back to the most recent call (continuation deltas with no key)
    fn merge_tool_use_delta(
        &mut self,
        id: &str,
        name: Option<&str>,
        input_delta: Option<&str>,
        provider_metadata: &Option<serde_json::Value>,
        index: Option<u32>,
    ) {
        let position = match index {
            Some(idx) => self.tool_calls.iter().position(|c| c.index == Some(idx)),
            None if !id.is_empty() => self.tool_calls.iter().position(|c| c.id == id),
            None if self.tool_calls.is_empty() => None,
            None => Some(self.tool_calls.len() - 1),
        };

        match position {
            Some(pos) => {
                let call = &mut self.tool_calls[pos];
                if call.id.is_empty() && !id.is_empty() {
                    call.id = id.to_string();
                }
                if call.name.is_empty()
                    && let Some(n) = name
                {
                    call.name = n.to_string();
                }
                if let Some(delta) = input_delta {
                    call.input_json.push_str(delta);
                }
                if call.provider_metadata.is_none() {
                    call.provider_metadata = provider_metadata.clone();
                }
            }
            None => {
                self.tool_calls.push(StreamedToolCall {
                    index,
                    id: id.to_string(),
                    name: name.unwrap_or_default().to_string(),
                    input_json: input_delta.unwrap_or_default().to_string(),
                    provider_metadata: provider_metadata.clone(),
                });
            }
        }
    }
}

#[cfg(test)]
mod accumulator_tests {
    use super::*;

    fn merge(
        acc: &mut StreamAccumulator,
        id: &str,
        name: Option<&str>,
        delta: Option<&str>,
        index: Option<u32>,
    ) {
        acc.merge_tool_use_delta(id, name, delta, &None, index);
    }

    #[test]
    fn openai_style_indexed_deltas_accumulate_per_call() {
        let mut acc = StreamAccumulator::default();
        merge(&mut acc, "call_a", Some("search"), None, Some(0));
        merge(&mut acc, "call_b", Some("fetch"), None, Some(1));
        // Interleaved continuation deltas carry only the index.
        merge(&mut acc, "", None, Some("{\"q\":"), Some(0));
        merge(&mut acc, "", None, Some("{\"url\":"), Some(1));
        merge(&mut acc, "", None, Some("\"cats\"}"), Some(0));
        merge(&mut acc, "", None, Some("\"x\"}"), Some(1));

        assert_eq!(acc.tool_calls.len(), 2);
        assert_eq!(acc.tool_calls[0].id, "call_a");
        assert_eq!(acc.tool_calls[0].input_json, "{\"q\":\"cats\"}");
        assert_eq!(acc.tool_calls[1].id, "call_b");
        assert_eq!(acc.tool_calls[1].input_json, "{\"url\":\"x\"}");
    }

    #[test]
    fn repeated_id_without_index_merges_instead_of_duplicating() {
        let mut acc = StreamAccumulator::default();
        merge(&mut acc, "call_a", Some("search"), Some("{\"q\":"), None);
        merge(&mut acc, "call_a", None, Some("\"cats\"}"), None);

        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].input_json, "{\"q\":\"cats\"}");
        assert_eq!(acc.tool_calls[0].name, "search");
    }

    #[test]
    fn empty_id_without_index_continues_most_recent_call() {
        let mut acc = StreamAccumulator::default();
        merge(&mut acc, "call_a", Some("search"), None, None);
        merge(&mut acc, "", None, Some("{}"), None);

        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].input_json, "{}");
    }

    #[test]
    fn distinct_ids_without_index_create_separate_calls() {
        let mut acc = StreamAccumulator::default();
        merge(&mut acc, "call_a", Some("search"), Some("{}"), None);
        merge(&mut acc, "call_b", Some("fetch"), Some("{}"), None);

        assert_eq!(acc.tool_calls.len(), 2);
    }

    #[test]
    fn late_id_and_name_fill_in_indexed_call() {
        let mut acc = StreamAccumulator::default();
        merge(&mut acc, "", None, Some("{"), Some(0));
        merge(&mut acc, "call_a", Some("search"), Some("}"), Some(0));

        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].id, "call_a");
        assert_eq!(acc.tool_calls[0].name, "search");
        assert_eq!(acc.tool_calls[0].input_json, "{}");
    }
}
