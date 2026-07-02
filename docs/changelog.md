# Changelog

All notable changes to the `kova-sdk` library are documented here.

## 0.8.0 — Provider error classification + MCP resilience

### Added
- `ProviderErrorClass` — `AuthInvalid` (401), `AuthForbidden` (403), `RateLimited { retry_after }` (429, with the `Retry-After` header when sent), `Overloaded` (408/500/502/503/504/529), `InvalidRequest` (400/413/422), `NotFound` (404), `Other`. Exposed on `KovaError::Provider { class }`, via `err.provider_class()`, and in the prelude.
- Constructors `KovaError::provider_http`, `provider_invalid`, `provider_auth` — all provider errors are now built through these, so classification is uniform across OpenAI-compatible, Gemini, Bedrock (exception types normalized to statuses first), and Ollama. Bedrock credential-chain failures classify as `AuthInvalid`.

- `McpClient::reconnect()` — tears down the connection and re-establishes it from the stored transport (respawns the stdio child, re-runs the `initialize` handshake). `tools_list`/`tools_call` do this automatically: a dead transport (`KovaError::Connection`) triggers one reconnect-and-retry, so a crashed MCP child no longer bricks the session. Connect itself retries transient failures once with a 500ms backoff.
- `tools/list` caching on `McpClient` — repeated agent builds against the same client don't re-query the server; the cache invalidates on `reconnect()`.
- `McpClient::tools_call_with_timeout(name, args, timeout)` — per-call timeout override for tools known to run long (or that must fail fast).

### Changed (breaking)
- `KovaError::Provider` gained the required `class` field; construct via the new constructors instead of struct literals.
- `KovaError::is_retryable()` now derives from the class (`RateLimited`/`Overloaded`); behavior is unchanged for all previously retryable statuses.
- MCP transport-level I/O failures (dead child process, broken HTTP stream) now surface as `KovaError::Connection` instead of `KovaError::Mcp`; server-reported JSON-RPC errors remain `Mcp`. Callers matching on `Mcp` for connectivity problems must match `Connection`.

## 0.7.0

## Streamable HTTP transport + OAuth tokens

### Added
- `McpTransport::StreamableHttp { url, headers, auth }` — the MCP 2025 Streamable-HTTP transport. Unlike `HttpSse`, it performs the `initialize` / `notifications/initialized` handshake, tracks the server's `Mcp-Session-Id` and echoes it on every request, sends `Accept: application/json, text/event-stream`, and parses both plain-JSON and SSE (`event: message` / `data:`) responses.
- `TokenProvider` trait (`async fn token()`, `async fn refresh()`) — a pluggable bearer-token source for `StreamableHttp`. The transport attaches `Authorization: Bearer <token()>` to each request and, on a `401`, calls `refresh()` once and retries. kova owns no OAuth logic; the host supplies tokens (e.g. via an OAuth flow + token store). `auth: Option<Arc<dyn TokenProvider>>` is `None` for unauthenticated servers.

### Notes
- Additive, backward-compatible: `Stdio` and `HttpSse` are unchanged. `HttpSse` remains the legacy static-header path (no `initialize` handshake); prefer `StreamableHttp` for modern remote servers.

## 0.6.0 — Thinking-token accounting

### Added
- `UsageStats::thinking_tokens: Option<u32>` — reasoning/thinking token count when the provider reports it separately. It is a *subset* of `output_tokens` (not additive). `None` means the provider gives no separate count, surfaced as "unknown" rather than a misleading `0`. Aggregated across all provider calls in a turn: `None + None` stays `None`, any reported value makes the running total `Some`.
- `StreamEvent::UsageEvent::thinking_tokens: Option<u32>` — reasoning tokens carried on streaming usage events, mirrored onto `UsageStats::thinking_tokens`.

### Changed
- Providers that report reasoning tokens now surface them: **OpenAI** (o-series, via `completion_tokens_details.reasoning_tokens`) and **Gemini** (`thoughtsTokenCount`). **Bedrock** and **Ollama** fold reasoning into `output_tokens` and report `thinking_tokens: None`.

## 0.5.0 — MCP server configuration

### Added
- `McpTransport::Stdio { env: HashMap<String, String> }` — extra environment variables set on the spawned MCP server process, on top of the inherited parent environment.
- `McpTransport::HttpSse { headers: HashMap<String, String> }` — extra HTTP headers sent with every JSON-RPC request to an HTTP+SSE MCP server (e.g. `Authorization` bearer tokens).

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
