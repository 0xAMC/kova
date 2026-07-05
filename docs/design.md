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

`LlmProvider` is an object-safe async trait. Both implementations (`OpenAiCompatibleProvider`, `BedrockProvider`) are stateless HTTP clients — all mutable state lives in the caller.

```
LlmProvider (trait, object-safe)
 ├── OpenAiCompatibleProvider   ← reqwest + SSE parsing
 ├── BedrockProvider            ← aws-sdk + SigV4 + ConverseStream
 ├── GeminiProvider             ← reqwest + SSE (alt=sse), x-goog-api-key, thinking-model filtering
 └── OllamaProvider             ← reqwest + NDJSON streaming, no auth, /api/chat + /api/tags
```

**Streaming contract**: `chat_completion_stream` returns a `Pin<Box<dyn Stream<…> + Send>>`. The agent polls this stream, accumulates `ToolUseDelta` events into complete tool-call records, then executes them after the stream closes. `ThinkingDelta` events surface as `AgentEvent::ThinkingDelta` on `run_stream` but are otherwise ignored by the tool-call accumulator.

**OpenAI path override**: `OpenAiProviderConfig` exposes `with_chat_completions_path` and `with_models_path` so Azure deployments, local servers, and proxies can override the default `/v1/chat/completions` and `/v1/models` paths without subclassing. `with_reasoning_effort` sets the `reasoning_effort` field for o-series models.

**Bedrock credential resolution**: explicit credentials → named profile → SDK default chain. The `BedrockProvider` resolves credentials once at construction; re-construction is required to rotate them. `BedrockProviderConfig::with_additional_model_request_fields` passes arbitrary JSON as `additionalModelRequestFields` in the Converse API request — used for model-specific knobs such as `budgetTokens` for extended thinking on Claude models.

**Gemini path override**: `GeminiProviderConfig` exposes `with_base_url` and `with_api_version` for test servers and future API versions. Streaming requests append `?alt=sse` so the shared SSE parser handles Gemini stream chunks without a separate code path. `with_thinking_budget` controls the extended-thinking token budget (`-1` = dynamic, `0` = off, positive = cap).

**Ollama streaming**: uses newline-delimited JSON (NDJSON) over `/api/chat` rather than SSE. Each line is a complete `OllamaResponse` object; the final chunk has `done: true` and carries token counts. The `OllamaThink` enum serialises as a bool (`true`) or string (`"high"` / `"medium"` / `"low"`) to match Ollama's `think` field.

**Thinking/reasoning across all providers**: chain-of-thought output (OpenAI `reasoning_content`, Bedrock `ReasoningContent`, Gemini `thought: true`, Ollama `thinking` field) is extracted by each provider's converter and placed in `ModelResponse::thinking`. It is never written into the conversation history and is never re-submitted to the LLM. During streaming, reasoning text arrives as `StreamEvent::ThinkingDelta`.

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

`InferenceConfig { model, max_tokens, temperature }` controls LLM call parameters. It is stored on `Agent` as a field (set via `AgentBuilder::inference_config(cfg)`) and cloned into each provider call inside the agentic loop. This replaces the earlier pattern of constructing `InferenceConfig::default()` inline on every iteration.

`AgentBuilder` defaults to `InferenceConfig::default()` (all fields `None`), so embedders that do not call `inference_config()` see no behaviour change.


## MCP Integration

MCP tools are discovered at build time (via `AgentBuilder::mcp_client`) and registered as `McpTool` adapters in the `ToolRegistry`. At runtime they are indistinguishable from native tools — the agent calls `execute` and `McpTool` serialises the JSON-RPC request to the MCP server.

`McpClient` serialises all JSON-RPC calls through `Arc<Mutex<McpConnection>>`, since the MCP spec requires ordered request/response pairs over a single connection.

Three transports are supported via `McpTransport`: `Stdio` (subprocess), `HttpSse` (legacy static-header HTTP), and `StreamableHttp` (the MCP 2025 transport — `initialize` handshake, `Mcp-Session-Id` tracking, JSON-or-SSE responses). Authentication is decoupled from the transport: `StreamableHttp` accepts an optional `TokenProvider`, and kova attaches the bearer token and retries once on `401` after calling `refresh()`. kova holds no OAuth state — the host owns the flow and token storage.


## Error Design

`KovaError` is a flat enum with `thiserror 2` derives. Each variant has a clear owner:

| Variant | Owner |
|---------|-------|
| `Provider`, `Connection`, `Http`, `Timeout` | Provider layer |
| `ToolExecution`, `ToolNotFound` | Tool layer |
| `Mcp` | MCP layer |
| `Stream` | Streaming layer |
| `MaxIterations`, `ContextBudgetExceeded` | Agent loop |
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

`MetricsCollector` uses atomic integers and a `RwLock<Vec<f64>>` for histograms. It intentionally does not integrate with OTEL metrics — it is a lightweight, always-available introspection tool, not a production metrics pipeline.
