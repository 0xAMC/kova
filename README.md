# kova

[![Crates.io](https://img.shields.io/crates/v/kova-sdk.svg)](https://crates.io/crates/kova-sdk)
[![Docs.rs](https://docs.rs/kova-sdk/badge.svg)](https://docs.rs/kova-sdk)
[![License](https://img.shields.io/crates/l/kova-sdk.svg)](LICENSE)

Async-first Rust library for building LLM-powered agents. Trait-based architecture with pluggable providers, tool calling, memory, streaming, MCP integration, multi-agent orchestration, and telemetry.

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
├── provider     # LlmProvider trait + OpenAI / Bedrock implementations
├── tool         # Tool trait + thread-safe ToolRegistry
├── memory       # MemoryStore trait + InMemoryStore
├── mcp          # MCP client (stdio / HTTP+SSE) + McpTool adapter
├── orchestrator # Multi-agent patterns (sequential, parallel, router)
├── streaming    # StreamingHandler trait + SSE parser
├── telemetry    # TelemetryConfig + MetricsCollector
├── models       # Shared data types (messages, content blocks, events)
└── error        # Unified KovaError enum
```

## Quick Start

```rust
use std::sync::Arc;
use kova_sdk::agent::AgentBuilder;
use kova_sdk::provider::openai::{OpenAiCompatibleProvider, OpenAiProviderConfig};

#[tokio::main]
async fn main() -> Result<(), kova_sdk::error::KovaError> {
    let config = OpenAiProviderConfig::new("http://127.0.0.1:1234", "my-model");
    let provider = Arc::new(OpenAiCompatibleProvider::new(config)?);

    let agent = AgentBuilder::new().provider(provider).build()?;
    let reply = agent.chat("conv-1", "Hello!").await?;
    println!("{reply}");
    Ok(())
}
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `telemetry` | off | Adds OpenTelemetry dependencies; enables OTLP/Jaeger/stdout span export |

Without the `telemetry` feature, `TelemetryConfig::init()` installs a lightweight `tracing_subscriber` — zero OTEL overhead.

```toml
[dependencies]
kova-sdk = { path = "../kova" }                                    # no OTEL
kova-sdk = { path = "../kova", features = ["telemetry"] }          # with OTEL
```

## Documentation

| Document | Contents |
|----------|---------|
| [docs/requirements.md](docs/requirements.md) | Functional and non-functional requirements |
| [docs/design.md](docs/design.md) | Architecture, agentic loop, design decisions |
| [docs/api-reference.md](docs/api-reference.md) | Full API with code examples for every module |
| [docs/changelog.md](docs/changelog.md) | Version history |
| [docs/contributing.md](docs/contributing.md) | Conventions, adding providers/tools, test guide |
