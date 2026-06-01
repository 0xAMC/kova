# Changelog

All notable changes to the `kova-sdk` library are documented here.

## Unreleased

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
