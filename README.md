# kova

[![Crates.io](https://img.shields.io/crates/v/kova-sdk.svg)](https://crates.io/crates/kova-sdk)
[![Docs.rs](https://docs.rs/kova-sdk/badge.svg)](https://docs.rs/kova-sdk)
[![License](https://img.shields.io/crates/l/kova-sdk.svg)](LICENSE)

Async-first Rust library for building LLM-powered agents. Trait-based architecture with pluggable providers, tool calling, memory, streaming, thinking-model support, MCP integration, multi-agent orchestration, and telemetry.

## Installation

```toml
[dependencies]
kova-sdk = "0.1"

# With OpenTelemetry tracing
kova-sdk = { version = "0.1", features = ["telemetry"] }
```

Or with cargo:

```bash
cargo add kova-sdk
```

## Architecture

```
kova
├── agent        # Agent + AgentBuilder — the main orchestration loop
├── provider     # LlmProvider trait + OpenAI / Bedrock / Gemini / Ollama implementations
├── tool         # Tool trait + thread-safe ToolRegistry
├── memory       # MemoryStore trait + InMemoryStore
├── mcp          # MCP client (stdio / HTTP+SSE) + McpTool adapter
├── orchestrator # Multi-agent patterns (sequential, parallel, router)
├── streaming    # StreamingHandler trait + SSE parser
├── telemetry    # TelemetryConfig + MetricsCollector
├── models       # Shared data types (messages, content blocks, events)
└── error        # Unified KovaError enum
```

## Providers

| Provider | Auth | Thinking models |
|----------|------|-----------------|
| `OpenAiCompatibleProvider` | Bearer token | `with_reasoning_effort("high")` for o-series models |
| `BedrockProvider` | SigV4 (explicit / profile / default chain) | `with_additional_model_request_fields(json!({"budgetTokens": N}))` for Claude |
| `GeminiProvider` | `x-goog-api-key` | `with_thinking_budget(N)` for `gemini-3.5-*` models etc |
| `OllamaProvider` | None (local) | `with_think(OllamaThink::High)` for `qwen3`, `deepseek-r1`, etc. |

Chain-of-thought output from thinking models is returned in `ModelResponse::thinking` and never stored in conversation history. During streaming it arrives as `StreamEvent::ThinkingDelta`; in blocking mode the agent forwards it to any registered `StreamingHandler`.

## Quick Start

```rust
use std::sync::Arc;
use kova_sdk::prelude::*;
use kova_sdk::provider::openai::{OpenAiCompatibleProvider, OpenAiProviderConfig};

#[tokio::main]
async fn main() -> Result<(), KovaError> {
    let config = OpenAiProviderConfig::new("http://127.0.0.1:1234", "my-model");
    let provider = Arc::new(OpenAiCompatibleProvider::new(config)?);

    let agent = AgentBuilder::new().provider(provider).build()?;
    let reply = agent.chat("conv-1", "Hello!").await?;
    println!("{reply}");
    Ok(())
}
```

## Stateless or session-backed — your choice

`Agent::run` is the core primitive: you own the conversation history, the agent owns the
turn. Nothing is read from or written to a store.

```rust
let mut history = vec![user_message("What's the weather in Tokyo?")];
let response = agent.run(&history).await?;          // tools execute, loop runs
history.extend(response.new_messages);              // you persist
println!("{} ({} tokens)", response.text, response.usage.total_tokens);
```

`chat` / `chat_response` layer persistence on top via the configured `MemoryStore`.
Writes are transactional per turn: a failed turn never leaves partial state behind.

## Streaming

Pull-based, no handler registration needed:

```rust
use futures::StreamExt;
use kova_sdk::agent::AgentEvent;

let stream = agent.run_stream(&history);
futures::pin_mut!(stream);
while let Some(event) = stream.next().await {
    match event? {
        AgentEvent::TextDelta { text } => print!("{text}"),
        AgentEvent::ToolCallStarted { name, .. } => eprintln!("⚙ {name}…"),
        AgentEvent::TurnCompleted { response } => history.extend(response.new_messages),
        _ => {}
    }
}
```

The callback-style `StreamingHandler` + `chat_stream` API is also available.

## Reliability

Transient provider failures (connection errors, timeouts, 408/429/5xx) are retried
automatically with exponential backoff — default 2 retries, configurable or disableable
via `AgentBuilder::retry_config(RetryConfig { .. })`.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `openai` | on | OpenAI-compatible provider |
| `gemini` | on | Google Gemini provider |
| `ollama` | on | Ollama provider |
| `bedrock` | on | AWS Bedrock provider (pulls in the AWS SDK crates) |
| `telemetry` | off | Adds OpenTelemetry dependencies; enables OTLP/Jaeger/stdout span export |

Skip the AWS dependency tree entirely if you don't need Bedrock:

```toml
[dependencies]
kova-sdk = { version = "0.3", default-features = false, features = ["openai"] }
```

Without the `telemetry` feature, `TelemetryConfig::init()` installs a lightweight `tracing_subscriber` — zero OTEL overhead.

## Documentation

| Document | Contents |
|----------|---------|
| [docs/design.md](docs/design.md) | Architecture, agentic loop, design decisions |
| [docs/api-reference.md](docs/api-reference.md) | Full API with code examples for every module |
| [docs/changelog.md](docs/changelog.md) | Version history |
| [docs/contributing.md](docs/contributing.md) | Conventions, adding providers/tools, test guide |
