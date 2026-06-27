# Design

## Architecture Overview

Kova is structured as a set of orthogonal, trait-based modules. Each module owns one concern; the `Agent` composes them at runtime.

```
┌──────────────────────────────────────────────────────────┐
│                        Agent                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Provider │  │ ToolRegistry │  │   MemoryStore     │  │
│  │(LlmProv.)│  │(RwLock<Map>) │  │ (RwLock<Map>)     │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │ StreamingHandler │  │    InferenceConfig           │  │
│  └──────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
         │ composes
┌────────▼──────────────────────────────────────────────────┐
│                    Orchestrator                            │
│  HashMap<String, Arc<Agent>> + timeout + OrchestratorPattern
└───────────────────────────────────────────────────────────┘
```

## Agentic Loop

The loop in `agent/mod.rs` is the core of the library:

```
chat(conversation_id, user_message)
  │
  ├─ store user message in MemoryStore
  │
  └─ loop (up to max_iterations + 1 total provider calls):
       ├─ build messages = system_prompt + memory.get_history()
       ├─ provider.chat_completion(messages, tool_defs, config)
       │
       ├─ StopReason::ToolUse?
       │    ├─ store assistant message (tool-use blocks)
       │    ├─ execute tools concurrently (Semaphore, default cap 10)
       │    ├─ store each ToolResult as Role::Tool message
       │    └─ continue loop
       │
       ├─ StopReason::EndTurn | MaxTokens | Unknown?
       │    ├─ store input_tokens in last_turn_input_tokens (AtomicU32) if provider reports usage
       │    └─ store assistant message + return text
       │
       └─ loop limit hit → Err(KovaError::MaxIterations)
```

`chat_stream` is identical but uses `provider.chat_completion_stream` and forwards `StreamEvent`s to the registered `StreamingHandler` before returning the full text.

## Provider Abstraction

`LlmProvider` is an object-safe async trait. Both implementations (`OpenAiCompatibleProvider`, `BedrockProvider`) are stateless HTTP clients — all mutable state lives in the caller.

```
LlmProvider (trait, object-safe)
 ├── OpenAiCompatibleProvider   ← reqwest + SSE parsing
 ├── BedrockProvider            ← aws-sdk + SigV4 + ConverseStream
 ├── GeminiProvider             ← reqwest + SSE (alt=sse), x-goog-api-key, thinking-model filtering
 └── OllamaProvider             ← reqwest + NDJSON streaming, no auth, /api/chat + /api/tags
```

**Streaming contract**: `chat_completion_stream` returns a `Pin<Box<dyn Stream<…> + Send>>`. The agent polls this stream, accumulates `ToolUseDelta` events into complete tool-call records, then executes them after the stream closes. `ThinkingDelta` events are forwarded to the `StreamingHandler` but otherwise ignored by the accumulator.

**OpenAI path override**: `OpenAiProviderConfig` exposes `with_chat_completions_path` and `with_models_path` so Azure deployments, local servers, and proxies can override the default `/v1/chat/completions` and `/v1/models` paths without subclassing. `with_reasoning_effort` sets the `reasoning_effort` field for o-series models.

**Bedrock credential resolution**: explicit credentials → named profile → SDK default chain. The `BedrockProvider` resolves credentials once at construction; re-construction is required to rotate them. `BedrockProviderConfig::with_additional_model_request_fields` passes arbitrary JSON as `additionalModelRequestFields` in the Converse API request — used for model-specific knobs such as `budgetTokens` for extended thinking on Claude models.

**Gemini path override**: `GeminiProviderConfig` exposes `with_base_url` and `with_api_version` for test servers and future API versions. Streaming requests append `?alt=sse` so the shared SSE parser handles Gemini stream chunks without a separate code path. `with_thinking_budget` controls the extended-thinking token budget (`-1` = dynamic, `0` = off, positive = cap).

**Ollama streaming**: uses newline-delimited JSON (NDJSON) over `/api/chat` rather than SSE. Each line is a complete `OllamaResponse` object; the final chunk has `done: true` and carries token counts. The `OllamaThink` enum serialises as a bool (`true`) or string (`"high"` / `"medium"` / `"low"`) to match Ollama's `think` field.

**Thinking/reasoning across all providers**: chain-of-thought output (OpenAI `reasoning_content`, Bedrock `ReasoningContent`, Gemini `thought: true`, Ollama `thinking` field) is extracted by each provider's converter and placed in `ModelResponse::thinking`. It is never written into the conversation history and is never re-submitted to the LLM. During streaming, reasoning text arrives as `StreamEvent::ThinkingDelta`. In the non-streaming `chat` path the agent replays any thinking text as `ThinkingDelta` + empty `ContentDelta` events to the registered `StreamingHandler` so callers see reasoning output in both modes.

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

## Memory Design

`MemoryStore` is keyed by `conversation_id`, allowing a single store instance to serve many concurrent conversations.

`InMemoryStore::with_max_messages(n)` truncates the oldest non-system messages when the count exceeds `n`. System prompt messages (Role::System) at position 0 are preserved across truncation to avoid breaking provider context requirements.

## MCP Integration

MCP tools are discovered at build time (via `AgentBuilder::mcp_client`) and registered as `McpTool` adapters in the `ToolRegistry`. At runtime they are indistinguishable from native tools — the agent calls `execute` and `McpTool` serialises the JSON-RPC request to the MCP server.

`McpClient` serialises all JSON-RPC calls through `Arc<Mutex<McpConnection>>`, since the MCP spec requires ordered request/response pairs over a single connection.

## Orchestrator Design

The `Orchestrator` holds a `HashMap<String, Arc<Agent>>`. Patterns reference agents by name (string keys). This decouples pattern configuration from agent construction and makes it easy to swap agents without rebuilding the orchestrator.

- **Sequential**: implemented as an iterative fold — no recursion, bounded stack.
- **Parallel**: uses `tokio::time::timeout` + `futures::future::join_all`; individual agent failures are captured rather than short-circuiting.
- **Router**: the router agent is called first; its response text is used as the key to look up the downstream agent name.

## Error Design

`KovaError` is a flat enum with `thiserror 2` derives. Each variant has a clear owner:

| Variant | Owner |
|---------|-------|
| `Provider`, `Connection`, `Http`, `Timeout` | Provider layer |
| `ToolExecution`, `ToolNotFound` | Tool layer |
| `Memory` | Memory layer |
| `Mcp` | MCP layer |
| `Stream` | Streaming layer |
| `Orchestration`, `MaxIterations` | Orchestrator / Agent loop |
| `Build` | AgentBuilder validation |
| `Serialization`, `Io` | Cross-cutting I/O |

`AgentBuilder::build()` is the only place where configuration errors (`Build` variant) can originate — everything else is a runtime error.

## Concurrency & Thread Safety

All public-facing types implement `Send + Sync`. The compile-time assertions in `tests/send_sync_assertions.rs` act as a regression guard — if a non-`Send` type is accidentally introduced, the build fails.

Internal synchronisation primitives:

| Type | Primitive | Reason |
|------|-----------|--------|
| `ToolRegistry` | `RwLock` | Many concurrent reads, infrequent writes |
| `InMemoryStore` | `RwLock` | Many reads, sequential append |
| `McpClient` | `Mutex` | Ordered JSON-RPC request/response |
| Tool parallelism | `Semaphore` | Bounded fan-out |
| `last_turn_input_tokens` | `AtomicU32` | Lock-free token counter read from any thread |

## Telemetry Design

The `telemetry` cargo feature gates all OTEL crates. This is a deliberate trade-off: OTEL crates are heavy (compile time + binary size); most embedders don't need them.

`TelemetryConfig::init()` installs either a full OTEL pipeline or a lightweight `tracing_subscriber` depending on the feature flag. The API surface is identical in both cases.

`MetricsCollector` uses atomic integers and a `RwLock<Vec<f64>>` for histograms. It intentionally does not integrate with OTEL metrics — it is a lightweight, always-available introspection tool, not a production metrics pipeline.
