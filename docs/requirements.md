# Requirements

## Functional Requirements

### Agent
- Execute a multi-turn agentic loop: receive user input, call an LLM, invoke tools, and loop until the LLM signals completion.
- Support both blocking (`chat`) and streaming (`chat_stream`) response modes.
- Enforce a configurable iteration cap (`max_iterations`) to prevent infinite loops.
- Execute multiple tools concurrently up to a configurable concurrency limit (`max_concurrent_tools`).
- Accept an optional system prompt prepended to every conversation.

### Providers
- Expose a single `LlmProvider` trait that decouples the agent from any specific LLM backend.
- Ship built-in implementations for OpenAI-compatible APIs, AWS Bedrock, and Google Gemini.
- Support tool/function calling in both blocking and streaming modes.
- OpenAI provider must work with OpenAI, Azure OpenAI, Ollama, vLLM, and LM Studio without code changes.
- Bedrock provider must resolve AWS credentials via explicit creds, named profile, or the default SDK chain.
- Gemini provider must authenticate via `x-goog-api-key` and support thinking models (filter chain-of-thought parts from user-visible output).

### Tools
- Define a `Tool` trait with `name`, `description`, `parameters_schema` (JSON Schema), and `execute`.
- Provide a thread-safe `ToolRegistry` that maps tool names to implementations.
- Support registering, overwriting, and looking up tools at runtime.

### Memory
- Define a `MemoryStore` trait: `add_message`, `get_history`, `clear` per `conversation_id`.
- Provide `InMemoryStore` with optional message-count cap that preserves system prompts on truncation.
- Default to an unbounded `InMemoryStore` when no store is supplied.

### MCP (Model Context Protocol)
- Connect to MCP servers over stdio (subprocess) or HTTP+SSE transports.
- Discover tools via `tools/list` and invoke them via `tools/call` (JSON-RPC 2.0).
- Automatically register discovered MCP tools into the agent's `ToolRegistry`.

### Streaming
- Define a `StreamingHandler` trait: `on_chunk`, `on_complete`, `on_error`.
- Parse Server-Sent Events from provider streams and surface them as typed `StreamEvent`s.

### Orchestration
- Coordinate multiple named `Agent` instances via three patterns:
  - **Sequential**: chain agents so each output feeds the next.
  - **Parallel**: fan-out to all agents with the same input; collect successes and failures.
  - **Router**: a designated router agent selects which downstream agent handles the input.
- Apply a per-execution timeout across all patterns.

### Telemetry
- Feature-gate OpenTelemetry (OTEL) dependencies under a `telemetry` cargo feature.
- Without the feature, install a lightweight `tracing_subscriber` (zero OTEL overhead).
- With the feature, support OTLP (gRPC/HTTP), Jaeger, and stdout span exporters.
- Provide a `MetricsCollector` for in-process counters and histograms that works regardless of the feature flag.

## Non-Functional Requirements

### Thread Safety
- All public types (`Agent`, `ToolRegistry`, `InMemoryStore`, `McpClient`, providers) must be `Send + Sync`.
- Shared mutable state must be guarded by `Arc<RwLock<…>>` or `Arc<Mutex<…>>`.
- Compile-time `Send + Sync` assertions live in `tests/send_sync_assertions.rs`.

### Async
- All I/O-bound operations are `async` and run on a `tokio` multi-thread runtime.
- No blocking calls inside async contexts.

### Error Handling
- All fallible operations return `Result<_, KovaError>`.
- No panics in library code; errors propagate to the caller.
- `KovaError` covers every failure domain (provider, tool, memory, MCP, streaming, orchestration, build, I/O).

### Observability
- All async spans are instrumented with `tracing::info_span!` and an `otel.status_code` field.

### Minimal Footprint
- Without the `telemetry` feature, OTEL crates are not compiled in.
- The library imposes no runtime framework beyond `tokio`.
