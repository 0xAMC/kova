# Changelog

All notable changes to the `kova-sdk` library are documented here.

## Unreleased

### Added
- `GeminiProvider` — new first-party provider for the Google Gemini API (`generativelanguage.googleapis.com`). Implements `LlmProvider` with `chat_completion`, `chat_completion_stream`, and `list_models`. Authenticates via `x-goog-api-key` header.
- `GeminiProviderConfig` — fluent builder with `new(model)`, `with_api_key`, `with_timeout`, `with_base_url` (useful for mock servers), and `with_api_version`. Defaults to `v1beta` and a 60 s timeout.
- Gemini streaming uses SSE (`alt=sse`) to reuse the shared SSE parser; `UsageEvent` is emitted from the final `usageMetadata` chunk.
- Thinking-model support: chain-of-thought parts (`thought: true`) are filtered from user-visible output; `thoughtSignature` is preserved as `provider_metadata` on tool-use blocks for downstream tools that require it.
- `Agent::last_turn_input_tokens() -> u32` — returns the input token count reported by the provider for the most recently completed turn (both `chat` and `chat_stream` paths). Returns `0` until the first turn completes or if the provider does not report usage. Updated atomically after every turn via `Arc<AtomicU32>`.
- `StreamEvent::UsageEvent { input_tokens, output_tokens }` — both OpenAI-compatible and Bedrock providers now emit this event during streaming so callers can track token consumption per call.
- `StopReason::as_str()` helper for span attribute recording.
- `TelemetryConfig::service_name` field (default `"kova"`) and corresponding `TelemetryConfigBuilder::service_name()` builder method — sets the `service.name` OTEL resource attribute on all exporter backends.
- `llm.input_tokens`, `llm.output_tokens`, `llm.stop_reason`, and `llm.iterations` span attributes recorded on `agent.chat`, `agent.chat_stream`, and per-call `llm.chat_completion_stream` child spans.
- `OaiStreamOptions { include_usage: true }` sent on streaming requests to OpenAI-compatible providers to surface token counts in stream chunks.

### Changed
- Bedrock `BedrockStreamEvent::Metadata` now maps to `StreamEvent::UsageEvent` instead of being silently dropped.

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
