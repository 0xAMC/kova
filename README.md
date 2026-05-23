# kova

[![Crates.io](https://img.shields.io/crates/v/kova.svg)](https://crates.io/crates/kova)
[![Docs.rs](https://docs.rs/kova/badge.svg)](https://docs.rs/kova)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Async-first Rust library for building LLM-powered agents. Trait-based architecture with pluggable providers, tool calling, memory, streaming, MCP integration, multi-agent orchestration, and telemetry.

## Installation

```toml
[dependencies]
kova = "0.1"

# With OpenTelemetry tracing
kova = { version = "0.1", features = ["telemetry"] }
```

Or with cargo:

```bash
cargo add kova
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
use kova::agent::AgentBuilder;
use kova::provider::openai::{OpenAiCompatibleProvider, OpenAiProviderConfig};

#[tokio::main]
async fn main() -> Result<(), kova::error::KovaError> {
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
kova = { path = "../kova" }                                    # no OTEL
kova = { path = "../kova", features = ["telemetry"] }          # with OTEL
```

## Documentation

| Document | Contents |
|----------|---------|
| [docs/requirements.md](docs/requirements.md) | Functional and non-functional requirements |
| [docs/design.md](docs/design.md) | Architecture, agentic loop, design decisions |
| [docs/api-reference.md](docs/api-reference.md) | Full API with code examples for every module |
| [docs/changelog.md](docs/changelog.md) | Version history |
| [docs/contributing.md](docs/contributing.md) | Conventions, adding providers/tools, test guide |
