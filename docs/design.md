# Design

## Architecture Overview

Kova is structured as a set of orthogonal, trait-based modules. Each module owns one concern; the `Agent` composes them at runtime.

```
┌──────────────────────────────────────────────────────────┐
│                        Agent (stateless)                 │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Provider │  │ ToolRegistry │  │  InferenceConfig  │  │
│  │(LlmProv.)│  │(RwLock<Map>) │  │                   │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │ ApprovalHandler  │  │  ToolLifecycleHook / metrics │  │
│  └──────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

The agent holds no conversation state: `run` takes caller-supplied history and
returns the messages it produced. Sessions, persistence, compaction, and
multi-agent orchestration are the host's concern (in muaz, `ChatSession` and
`PipelineRunner`).

## Agentic Loop

The loop in `agent/mod.rs` is the core of the library:

```
run(messages)   // caller-owned history in
  │
  ├─ working = system_prompt + messages
  │
  └─ loop (up to max_iterations + 1 total provider calls):
       ├─ context-budget guard (heuristic) → Err(ContextBudgetExceeded) if over
       ├─ provider.chat_completion(working, tool_defs, config)
       │
       ├─ StopReason::ToolUse?
       │    ├─ append assistant message (tool-use blocks) to working
       │    ├─ execute tools concurrently (Semaphore, default cap 10)
       │    ├─ append each ToolResult as Role::Tool message
       │    └─ continue loop
       │
       ├─ StopReason::EndTurn | MaxTokens | Unknown?
       │    ├─ record input_tokens in last_turn_input_tokens (AtomicU32) if reported
       │    └─ return AgentResponse { text, new_messages, usage, … }
       │
       └─ loop limit hit → Err(KovaError::MaxIterations)
```

`run_stream` is the pull-based sibling: same loop over `chat_completion_stream`, yielding `AgentEvent`s (`TextDelta` / `ThinkingDelta` / `ToolCall*` / `TurnCompleted`) instead of forwarding to a handler.

## Provider Abstraction

`LlmProvider` is an object-safe async trait. All five implementations are stateless HTTP clients — all mutable state lives in the caller.

```
LlmProvider (trait, object-safe)
 ├── AnthropicProvider          ← reqwest + native Messages API SSE, adaptive thinking, prompt caching
 ├── OpenAiCompatibleProvider   ← reqwest + SSE parsing
 ├── BedrockProvider            ← aws-sdk + SigV4 + ConverseStream
 ├── GeminiProvider             ← reqwest + SSE (alt=sse), x-goog-api-key, thinking-model filtering
 └── OllamaProvider             ← reqwest + NDJSON streaming, no auth, /api/chat + /api/tags
```

**Streaming contract**: `chat_completion_stream` returns a `Pin<Box<dyn Stream<…> + Send>>`. The agent polls this stream, accumulates `ToolUseDelta` events into complete tool-call records, then executes them after the stream closes. `ThinkingDelta` events surface as `AgentEvent::ThinkingDelta` on `run_stream` but are otherwise ignored by the tool-call accumulator. `ThinkingBlock` events (Anthropic only) instead feed the accumulator, since they must be preserved verbatim as a `ContentBlock::Thinking` — surfacing the text once via `ThinkingDelta` would double it.

**Anthropic path (native Messages API)**: `AnthropicProvider` speaks `POST /v1/messages` directly rather than going through the OpenAI-compatible shape. `AnthropicProviderConfig` defaults `adaptive_thinking` and `cache` to on — every request sends `thinking: {"type": "adaptive"}` and a top-level `cache_control: {"type": "ephemeral"}` that marks the stable prefix (system prompt, tools, prior turns) for Anthropic's prompt cache; `with_effort("low".."max")` maps to `output_config.effort`. It is also the only provider overriding `LlmProvider::count_tokens` with an exact call to `/v1/messages/count_tokens` instead of the offline heuristic. Because Anthropic requires a *signed* thinking block echoed back unchanged on the next tool-use turn, its reasoning output round-trips as `ContentBlock::Thinking { thinking, signature }` in conversation history — the one case where "thinking never touches history" (see below) doesn't hold. Other providers drop `Thinking` blocks when building requests, so cross-provider history stays valid.

**OpenAI path override**: `OpenAiProviderConfig` exposes `with_chat_completions_path` and `with_models_path` so Azure deployments, local servers, and proxies can override the default `/v1/chat/completions` and `/v1/models` paths without subclassing. `with_reasoning_effort` sets the `reasoning_effort` field for o-series models.

**Bedrock credential resolution**: explicit credentials → named profile → SDK default chain. The `BedrockProvider` resolves credentials once at construction; re-construction is required to rotate them. `BedrockProviderConfig::with_additional_model_request_fields` passes arbitrary JSON as `additionalModelRequestFields` in the Converse API request — used for model-specific knobs such as `budgetTokens` for extended thinking on Claude models.

**Gemini path override**: `GeminiProviderConfig` exposes `with_base_url` and `with_api_version` for test servers and future API versions. Streaming requests append `?alt=sse` so the shared SSE parser handles Gemini stream chunks without a separate code path. `with_thinking_budget` controls the extended-thinking token budget (`-1` = dynamic, `0` = off, positive = cap).

**Ollama streaming**: uses newline-delimited JSON (NDJSON) over `/api/chat` rather than SSE. Each line is a complete `OllamaResponse` object; the final chunk has `done: true` and carries token counts. The `OllamaThink` enum serialises as a bool (`true`) or string (`"high"` / `"medium"` / `"low"`) to match Ollama's `think` field.

**Thinking/reasoning across all providers**: chain-of-thought output (OpenAI `reasoning_content`, Bedrock `ReasoningContent`, Gemini `thought: true`, Ollama `thinking` field) is extracted by each provider's converter and placed in `ModelResponse::thinking`. It is never written into the conversation history and is never re-submitted to the LLM. During streaming, reasoning text arrives as `StreamEvent::ThinkingDelta`. Anthropic is the sole exception, per above: its signed thinking blocks are required history, not just display text.

## Tool Execution Model

```
ToolRegistry: Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>
```

Clone of a `ToolRegistry` shares the same inner `Arc` — all clones see the same registrations. This allows the agent to pass a clone to concurrent tasks without re-locking the outer `Arc`.

Concurrent execution uses `tokio::sync::Semaphore` to cap parallelism. Tool results are collected with `futures::future::join_all` and stored in order.

### Built-in tools (`src/tools/`, feature-gated)

The crate ships generic filesystem, shell, and web tools behind the `tools` /
`web-tools` features. They are **decoupled from any host application**: every
runtime constraint is injected through a `ToolPolicy`, and the tools never read
configuration or environment-specific state themselves. This is a deliberate
dependency-inversion boundary — `WebPolicy::default()` owns the safe defaults in
the SDK, and an embedder maps *its* configuration onto `ToolPolicy`/`WebPolicy`,
so the SDK takes no dependency on any downstream config types. The two-tier
feature split keeps the heavy HTML/readability stack out of builds that only need
the light filesystem/shell tools.

Security boundaries live with the tools that need them: filesystem tools resolve
and contain paths through `resolve_for_containment` (canonicalizing the deepest
existing ancestor, so `..` and symlink escapes fail a `starts_with` check), and
the web tools defend against SSRF by rejecting private/internal addresses and
pinning the HTTP client to the validated IP. Tool failures are surfaced in-band
as error `ToolResult`s so the model can recover, not as transport `KovaError`s.

## InferenceConfig

`InferenceConfig { model, max_tokens, temperature, top_p, stop_sequences, response_format }` controls LLM call parameters. It is stored on `Agent` as a field (set via `AgentBuilder::inference_config(cfg)`) and cloned into each provider call inside the agentic loop. This replaces the earlier pattern of constructing `InferenceConfig::default()` inline on every iteration.

`AgentBuilder` defaults to `InferenceConfig::default()` (all fields `None`), so embedders that do not call `inference_config()` see no behaviour change.

## Structured Output

`response_format: Option<ResponseFormat>` is just another `InferenceConfig` field — it flows through `run`/`run_with_config` like `temperature` does. `Agent::run_structured::<T>(messages, format)` is a thin convenience: it sets the field for one turn and parses the resulting text into `T`, but the constraint itself is applied by the provider, not the agent loop. Each converter maps `ResponseFormat` to its native mechanism (OpenAI `response_format` with `strict: true`, Anthropic `output_config.format`, Gemini `responseMimeType` + a sanitized `responseSchema`, Ollama `format`) rather than kova post-hoc validating JSON — a model that natively enforces the schema fails less often than one asked to and checked after the fact. Bedrock has no equivalent in the Converse API, so it rejects a request that sets `response_format` rather than silently ignoring the constraint.

## Token Counting & Context Budgets

`LlmProvider::count_tokens` defaults to `heuristic_count_tokens` (~4 chars/token plus per-message overhead) so every provider gets a free, offline estimate; Anthropic overrides it with an exact call to `/v1/messages/count_tokens`. The agent's own `context_budget` guard deliberately uses the cheap heuristic on every call rather than a provider's exact counter — an exact count would mean a network round trip before *every* provider call just to decide whether to make one. The heuristic is conservative enough to guard against runaway prompts; call `provider.count_tokens()` directly when a host needs an exact figure (e.g. before a compaction decision).

## Cancellation

`run_cancellable` / `run_stream_cancellable` thread a `CancellationToken` through the loop, raced against the in-flight provider call and each tool execution via `tokio::select!`. A cancellation mid-tool-execution relies on the tool's own future being dropped promptly — for the shell tool this matters because a dropped future must still reap the child process, which is why it's spawned with `kill_on_drop(true)` rather than left to `Drop::drop` on the `Child` handle (which does not by itself kill the process). A cancelled turn returns `KovaError::Cancelled` and produces no messages, so callers never have to reconcile a partial turn against their history.

## Embeddings

`EmbeddingProvider` is deliberately the smallest possible seam: `embed(&[String]) -> Vec<Vec<f32>>` plus an optional `dimensions()`. kova stops there — chunking strategy, indexing, and vector search are all opinionated choices that belong to the host, and baking one in would mean every consumer either fights kova's choice or ignores this part of the crate entirely. `OpenAiEmbeddingProvider` and `OllamaEmbeddingProvider` both re-sort responses by their returned `index` before returning, since providers document order-preservation but not all guarantee it under retries.

## MCP Integration

MCP tools are discovered at build time (via `AgentBuilder::mcp_client`) and registered as `McpTool` adapters in the `ToolRegistry`. At runtime they are indistinguishable from native tools — the agent calls `execute` and `McpTool` serialises the JSON-RPC request to the MCP server.

`McpClient` serialises all JSON-RPC calls through `Arc<Mutex<McpConnection>>`, since the MCP spec requires ordered request/response pairs over a single connection.

Three transports are supported via `McpTransport`: `Stdio` (subprocess), `HttpSse` (legacy static-header HTTP), and `StreamableHttp` (the MCP 2025 transport — `initialize` handshake, `Mcp-Session-Id` tracking, JSON-or-SSE responses). Authentication is decoupled from the transport: `StreamableHttp` accepts an optional `TokenProvider`, and kova attaches the bearer token and retries once on `401` after calling `refresh()`. kova holds no OAuth state — the host owns the flow and token storage.


## Error Design

`KovaError` is a flat enum with `thiserror 2` derives. Each variant has a clear owner:

| Variant | Owner |
|---------|-------|
| `Provider`, `Connection`, `Timeout` | Provider layer |
| `ToolExecution`, `ToolNotFound` | Tool layer |
| `Mcp` | MCP layer |
| `Stream` | Streaming layer |
| `MaxIterations`, `ContextBudgetExceeded`, `Cancelled` | Agent loop |
| `Build` | AgentBuilder validation |
| `Serialization`, `Io` | Cross-cutting I/O |

`AgentBuilder::build()` is the only place where configuration errors (`Build` variant) can originate — everything else is a runtime error.

## Concurrency & Thread Safety

All public-facing types implement `Send + Sync`. The compile-time assertions in `tests/send_sync_assertions.rs` act as a regression guard — if a non-`Send` type is accidentally introduced, the build fails.

Internal synchronisation primitives:

| Type | Primitive | Reason |
|------|-----------|--------|
| `ToolRegistry` | `RwLock` | Many concurrent reads, infrequent writes |
| `McpClient` | `Mutex` | Ordered JSON-RPC request/response |
| Tool parallelism | `Semaphore` | Bounded fan-out |
| `last_turn_input_tokens` | `AtomicU32` | Lock-free token counter read from any thread |

## Telemetry Design

The `telemetry` cargo feature gates all OTEL crates. This is a deliberate trade-off: OTEL crates are heavy (compile time + binary size); most embedders don't need them.

`TelemetryConfig::init()` installs either a full OTEL pipeline or a lightweight `tracing_subscriber` depending on the feature flag. The API surface is identical in both cases.

`MetricsCollector` uses atomic integers for counters and a fixed-bucket `RwLock<Histogram>` for latency/duration distributions — constant memory regardless of request volume, unlike a naive `Vec<f64>` of raw samples. It intentionally does not integrate with OTEL metrics — it is a lightweight, always-available introspection tool, not a production metrics pipeline.
