# Changelog

All notable changes to the `kova-sdk` library are documented here.

## 0.4.0 — Built-in tools

### Added
- **Built-in tools** (`kova_sdk::tools`), opt-in behind two feature flags so the core stays dependency-light:
  - `tools` — filesystem and shell tools: `read_file`, `list_dir`, `search` (regex + glob), `edit_file`, `write_file`, `patch_file` (unified diff), and `shell`. Light deps (`glob`, `regex`, `diffy`).
  - `web-tools` (implies `tools`) — `fetch_webpage` (readability extraction → Markdown/text, content-type dispatch) plus `tools::fetch_text`, a public SSRF-guarded HTTP-GET helper for building further network tools against trusted endpoints. Adds `scraper`/`dom_smoothie`/`htmd` and enables `tokio/net`.
- `ToolPolicy` / `WebPolicy` — injected, config-agnostic guardrails. `ToolPolicy` confines filesystem reads/writes to a `workspace_root`, blocks `protected_paths`, and bounds `shell_timeout`; `WebPolicy::default()` owns safe network defaults (HTTPS-only, no private hosts, response/content caps). Hosts map their own config onto these structs — the tools never read host configuration.
- `register_all_tools()` / `register_all_tools_with_policy(Arc<ToolPolicy>)` — build the full tool set (web tools included only under `web-tools`).
- `tools::tool_result` / `tools::tool_error` helpers, and `tools::normalize_path` / `tools::resolve_for_containment` (symlink-aware path containment).
- **SSRF defense** in the web tools: hostnames resolving to loopback/private/link-local/CGNAT addresses are rejected (unless `allow_private_hosts`), the HTTP client is pinned to the validated IP to defeat DNS rebinding, and redirects are followed manually with every guardrail re-checked per hop.

## 0.3.0 - Stateless agent loop

### Added
- **Stateless core loop**: `Agent::run(&[ConversationMessage]) -> AgentResponse` — caller-supplied history in, full result out, no memory-store involvement. `AgentResponse` carries `text`, `new_messages` (everything the turn produced, for the caller to persist), `stop_reason`, aggregated `usage` across all provider calls in the turn, `llm_calls`, and `thinking`.
- `Agent::run_with_config(messages, overrides)` — per-call `InferenceConfig` overrides; unset fields fall back to the agent's defaults.
- `Agent::run_stream(messages) -> impl Stream<Item = Result<AgentEvent, _>>` — pull-based streaming with `TextDelta`, `ThinkingDelta`, `ToolCallStarted`, `ToolCallFinished`, and `TurnCompleted` events. No `StreamingHandler` registration required.
- `Agent::chat_response(conversation_id, text) -> AgentResponse` — session-backed variant of `chat` returning the full response.
- **Retries**: `RetryConfig` (default: 2 retries, exponential backoff capped at 10 s) applied to every provider call and stream establishment; only errors where `KovaError::is_retryable()` is true are retried. Configure via `AgentBuilder::retry_config`, disable with `RetryConfig::disabled()`.
- **Provider feature flags**: `openai`, `gemini`, `ollama`, `bedrock` (all default). `default-features = false, features = ["openai"]` skips compiling the AWS dependency stack entirely.
- `KovaError::status_code()` and `KovaError::is_retryable()` helpers.
- `InferenceConfig::top_p` and `InferenceConfig::stop_sequences`, wired through all four providers.
- `StreamEvent::ToolUseDelta::index` — provider-assigned correlation key (OpenAI `index`, Bedrock `contentBlockIndex`) for matching argument deltas to tool calls.
- `kova_sdk::prelude` — one-line import of the common types.
- `AgentBuilder::metrics(Arc<MetricsCollector>)` — the agent records LLM latency/tokens/errors and tool durations automatically.
- `McpClient::connect_with_timeout` — per-request timeout (default 30 s) bounding the MCP handshake, `tools/list`, and every `tools/call`.
- Approval decisions `ApprovedForSession` / `DeniedAlways` are now enforced by the agent for its lifetime (handler consulted once per tool).
- `ApprovalDecision::DeniedWithReason(String)` — deny a single call and pass a free-text reason back to the model as part of the error tool result, so it can adapt (e.g. "use the staging database instead"). `UsageStats` now derives `Default`.
- `Orchestrator::execute_in_namespace` — explicit conversation namespace for deliberate continuation; `execute()` now isolates each run.

### Changed
- **Memory writes are transactional per turn**: `chat`/`chat_stream` persist the user message and produced messages only after the turn succeeds. A failed turn leaves the conversation unchanged (previously a mid-turn failure left dangling user messages or orphaned tool-use blocks in the store).
- `InMemoryStore` truncation is now tool-pair safe: history is cut at a user-message boundary so tool results are never orphaned (providers reject such histories).
- `ToolRegistry` methods (`register`, `get`, `list`, `tool_definitions`) are now synchronous; `tool_definitions` returns a cached `Arc<Vec<ToolDefinition>>` invalidated on registration.
- `MetricsCollector` histograms are fixed-bucket with constant memory (previously unbounded `Vec<f64>`); snapshot getters return `HistogramSnapshot` instead of raw samples.
- Streamed tool-call argument deltas are correlated by provider `index`, falling back to `id`, then to the most recent call — fixes duplicated/merged calls with providers that repeat ids or interleave parallel calls.
- Provider streaming decodes bytes line-wise (`BytesMut`), fixing UTF-8 corruption when multi-byte characters split across network chunks, and making buffering O(1) per line. The three per-provider SSE/NDJSON adapters were unified into one shared adapter.
- Provider methods attach tracing spans with `.instrument()` instead of holding span guards across `.await` (fixes corrupted OTEL traces).
- MCP stdio child processes are killed on drop and their stderr is logged via `tracing` instead of discarded.
- `AgentBuilder::tool()` and `tool_registry()` now compose instead of erroring when combined.
- SSE parsing follows the spec: at most one leading space stripped after `data:`, CRLF handled explicitly.

### Removed
- `KovaError::Http(reqwest::Error)` — unused; provider transport errors map to `Connection`/`Timeout`/`Provider`.

## 0.2.0 — New providers for Gemini and Ollama

### Added
- `OllamaProvider` — new first-party provider for locally-running Ollama instances. Implements `LlmProvider` with `chat_completion`, `chat_completion_stream`, and `list_models`. No API key required; connects to `http://localhost:11434` by default.
- `OllamaProviderConfig` — fluent builder with `new(model)`, `with_base_url`, `with_timeout` (default 120 s), `with_keep_alive`, `with_think`, and `with_extra_options`. Streaming uses NDJSON over the `/api/chat` endpoint.
- `OllamaThink` — enum controlling Ollama reasoning mode: `Enabled` (basic), `High`, `Medium`, `Low`. Pass via `OllamaProviderConfig::with_think(OllamaThink::High)`. Requires a thinking-capable model (e.g. `qwen3`, `deepseek-r1`).
- `GeminiProvider` — new first-party provider for the Google Gemini API (`generativelanguage.googleapis.com`). Implements `LlmProvider` with `chat_completion`, `chat_completion_stream`, and `list_models`. Authenticates via `x-goog-api-key` header.
- `GeminiProviderConfig` — fluent builder with `new(model)`, `with_api_key`, `with_timeout`, `with_base_url` (useful for mock servers), and `with_api_version`. Defaults to `v1beta` and a 60 s timeout.
- `GeminiProviderConfig::with_thinking_budget(budget: i32)` — controls the extended-thinking token budget: `-1` = dynamic/unlimited, `0` = disabled (default), positive value = hard cap. Only affects thinking-capable models (e.g. `gemini-2.5-flash`).
- `OpenAiProviderConfig::with_reasoning_effort(effort)` — sets the `reasoning_effort` field for OpenAI o-series models. Accepted values: `"low"`, `"medium"`, `"high"`. Has no effect on non-reasoning models.
- `BedrockProviderConfig::with_additional_model_request_fields(fields: serde_json::Value)` — passes arbitrary key/value pairs as `additionalModelRequestFields` in the Bedrock Converse API request. Use this for model-specific parameters not covered by `InferenceConfig` (e.g. `{"budgetTokens": 5000}` for extended thinking on Claude models).
- `ModelResponse::thinking: Option<String>` — reasoning/chain-of-thought text produced by thinking models. Not stored in conversation history and not re-submitted to the LLM on subsequent turns.
- `StreamEvent::ThinkingDelta { text }` — emitted when a model outputs chain-of-thought reasoning. All four providers surface this event. Also emitted by the non-streaming `chat` path when the response contains thinking text, so a `StreamingHandler` can display reasoning in both modes.
- `Agent::last_turn_input_tokens() -> u32` — returns the input token count reported by the provider for the most recently completed turn. Returns `0` until the first turn completes or if the provider does not report usage.
- `StreamEvent::UsageEvent { input_tokens, output_tokens }` — emitted at the end of streaming turns so callers can track token consumption per call. Supported by all four providers.
- `StopReason::as_str()` — returns a string representation suitable for logging and span attributes.
- `TelemetryConfig::service_name` field (default `"kova"`) and `TelemetryConfigBuilder::service_name()` builder method — sets the `service.name` OTEL resource attribute on all exporter backends.
- `llm.input_tokens`, `llm.output_tokens`, `llm.stop_reason`, and `llm.iterations` span attributes on `agent.chat` and `agent.chat_stream` spans.

### Changed
- Thinking/reasoning output from all providers is now surfaced in `ModelResponse::thinking` and never stored in conversation history. Affected providers: Bedrock (`ReasoningContent`), Gemini (`thought: true` parts), OpenAI (`reasoning_content`), Ollama (`thinking` field).
- `agent.chat` (non-streaming) now forwards thinking text to any registered `StreamingHandler` via `ThinkingDelta` events so reasoning is visible in both streaming and blocking modes.
- Bedrock streaming now emits `StreamEvent::UsageEvent` from `Metadata` events (previously dropped) and `StreamEvent::ThinkingDelta` from `ReasoningContent` deltas (previously dropped).

## 0.1.0 — Initial release

### Added
- `Agent` with blocking (`chat`) and streaming (`chat_stream`) agentic loops.
- `AgentBuilder` fluent API with validation.
- `LlmProvider` trait + `OpenAiCompatibleProvider` (OpenAI, Azure, Ollama, vLLM, LM Studio).
- `LlmProvider` trait + `BedrockProvider` (AWS Bedrock Converse / ConverseStream, SigV4).
- `Tool` trait + `ToolRegistry` (thread-safe, `Arc<RwLock<HashMap>>`).
- `MemoryStore` trait + `InMemoryStore` (unbounded and capped variants).
- `McpClient` with stdio and HTTP+SSE transports (JSON-RPC 2.0).
- `McpTool` adapter — wraps MCP tool definitions into the `Tool` trait.
- `StreamingHandler` trait + SSE line parser.
- `Orchestrator` with Sequential, Parallel, and Router patterns.
- `TelemetryConfig` with feature-gated OTEL (OTLP gRPC/HTTP, Jaeger, stdout).
- `MetricsCollector` (always available, no feature flag required).
- `KovaError` unified error enum (14 variants, `thiserror 2`).
- Compile-time `Send + Sync` assertions.
- Integration tests: agent, tools, streaming, orchestrator, Bedrock, concurrency, property-based.
- Mock implementations: `MockLlmProvider`, mock tools.

### Changed
- Split monolithic `tests/property_tests.rs` into 8 concern-specific files (`models_property_tests.rs`, `tool_registry_property_tests.rs`, `tool_calling_property_tests.rs`, `memory_property_tests.rs`, `agent_memory_property_tests.rs`, `streaming_property_tests.rs`, `mcp_property_tests.rs`, `orchestration_property_tests.rs`).
- Tightened visibility modifiers: `streaming::sse` module and its `SseLine`, `parse_sse_line`, `parse_sse_data` symbols narrowed from `pub` to `pub(crate)`; `bedrock::config::encode_path_segment` narrowed from `pub(crate)` to `pub(super)`; `BedrockProvider::sign_request` narrowed from `pub(crate)` to private.
- Moved `prop_sse_line_parsing_roundtrip` proptest into the `sse` module's `#[cfg(test)]` block.
