# Changelog

All notable changes to the `kova` library are documented here.

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
