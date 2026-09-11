# kova

[![Crates.io](https://img.shields.io/crates/v/kova-sdk.svg)](https://crates.io/crates/kova-sdk)
[![Docs.rs](https://docs.rs/kova-sdk/badge.svg)](https://docs.rs/kova-sdk)
[![License](https://img.shields.io/crates/l/kova-sdk.svg)](LICENSE)

Async-first Rust library for building LLM-powered agents. Trait-based architecture with pluggable providers, tool calling, structured output, prompt caching, embeddings, pull-based streaming, thinking-model support, MCP integration, and telemetry. The agent is stateless — the host owns conversation history.

## Installation

```toml
[dependencies]
kova-sdk = "0.9"

# With OpenTelemetry tracing
kova-sdk = { version = "0.9", features = ["telemetry"] }
```

Or with cargo:

```bash
cargo add kova-sdk
```

## Architecture

```
kova
├── agent        # Agent + AgentBuilder — the main orchestration loop
├── provider     # LlmProvider trait + Anthropic / OpenAI / Bedrock / Gemini / Ollama implementations
├── embedding    # EmbeddingProvider trait + OpenAI / Ollama implementations
├── tool         # Tool trait + thread-safe ToolRegistry
├── tools        # Built-in fs/shell/web tools (feature `tools` / `web-tools`)
├── mcp          # MCP client (stdio / HTTP+SSE / Streamable HTTP + OAuth) + McpTool adapter
├── streaming    # SSE / line-framing helpers for provider streams
├── telemetry    # TelemetryConfig + MetricsCollector
├── models       # Shared data types (messages, content blocks, events)
└── error        # Unified KovaError enum
```

## Providers

| Provider | Auth | Thinking models |
|----------|------|-----------------|
| `AnthropicProvider` | `x-api-key` | On by default (`with_adaptive_thinking(true)`); tune with `with_effort("high")`. Also the only provider with automatic prompt caching and an exact, network-backed `count_tokens` |
| `OpenAiCompatibleProvider` | Bearer token | `with_reasoning_effort("high")` for o-series models |
| `BedrockProvider` | SigV4 (explicit / profile / default chain) | `with_additional_model_request_fields(json!({"budgetTokens": N}))` for Claude |
| `GeminiProvider` | `x-goog-api-key` | `with_thinking_budget(N)` for `gemini-3.5-*` models etc |
| `OllamaProvider` | None (local) | `with_think(OllamaThink::High)` for `qwen3`, `deepseek-r1`, etc. |

Chain-of-thought output from thinking models is returned in `ModelResponse::thinking` and never stored in conversation history. During streaming it arrives as `AgentEvent::ThinkingDelta`. Anthropic's signed thinking blocks are the exception — they round-trip verbatim in conversation history as `ContentBlock::Thinking`, since the API requires them echoed back unchanged on the next tool-use turn.

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
    let messages = vec![user_message("Hello!")];
    let response = agent.run(&messages).await?;
    println!("{}", response.text);
    Ok(())
}
```

## Stateless by design

`Agent::run` is the core primitive: you own the conversation history, the agent owns the
turn. Nothing is read from or written to any store — persistence, sessions, and
compaction belong to the host.

```rust
let mut history = vec![user_message("What's the weather in Tokyo?")];
let response = agent.run(&history).await?;          // tools execute, loop runs
history.extend(response.new_messages);              // you persist
println!("{} ({} tokens)", response.text, response.usage.total_tokens);
```

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

## Reliability

Transient provider failures (connection errors, timeouts, 408/429/5xx) are retried
automatically with exponential backoff — default 2 retries, configurable or disableable
via `AgentBuilder::retry_config(RetryConfig { .. })`.

Two more knobs for production turns:

- **Context budgets** — `AgentBuilder::context_budget(max_prompt_tokens)` checks an offline heuristic before every provider call and fails fast with `KovaError::ContextBudgetExceeded` instead of shipping a request the provider will reject.
- **Cancellation** — `run_cancellable` / `run_stream_cancellable` take a `CancellationToken` (re-exported in the prelude) and abort mid-provider-call or mid-tool-execution, killing any spawned tool processes, with no partial messages left behind.

## Structured Output & Embeddings

```rust
use kova_sdk::models::ResponseFormat;

#[derive(serde::Deserialize)]
struct Route { route: String }

let format = ResponseFormat::named("route", serde_json::json!({
    "type": "object",
    "properties": { "route": { "type": "string" } },
    "required": ["route"],
    "additionalProperties": false
}));
let (route, response) = agent.run_structured::<Route>(&messages, format).await?;
```

Mapped natively per provider (OpenAI `response_format`, Anthropic `output_config.format`, Gemini `responseSchema`, Ollama `format`); Bedrock has no equivalent and rejects requests that set it.

For retrieval on top of kova, `EmbeddingProvider` gives you `embed(&[String]) -> Vec<Vec<f32>>` via `embedding::openai::OpenAiEmbeddingProvider` or `embedding::ollama::OllamaEmbeddingProvider`. kova ships no vector store — chunking, indexing, and search stay with the host.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `anthropic` | on | Native Anthropic Messages API provider (streaming, tool use, adaptive thinking, prompt caching) |
| `openai` | on | OpenAI-compatible provider |
| `gemini` | on | Google Gemini provider |
| `ollama` | on | Ollama provider |
| `bedrock` | on | AWS Bedrock provider (pulls in the AWS SDK crates) |
| `telemetry` | off | Adds OpenTelemetry dependencies; enables OTLP/Jaeger/stdout span export |
| `tools` | off | Built-in filesystem and shell tools (`read_file`, `list_dir`, `search`, `edit_file`, `write_file`, `patch_file`, `shell`) |
| `web-tools` | off | Adds the `fetch_webpage` tool and the SSRF-guarded `fetch_text` helper (for building further network tools); pulls in an HTML parser and readability extraction. Implies `tools` |

Skip the AWS dependency tree entirely if you don't need Bedrock:

```toml
[dependencies]
kova-sdk = { version = "0.9", default-features = false, features = ["anthropic"] }
```

Without the `telemetry` feature, `TelemetryConfig::init()` installs a lightweight `tracing_subscriber` — zero OTEL overhead.

The built-in tools are opt-in to keep the core dependency-light. Enable them and register against an injected [`ToolPolicy`](https://docs.rs/kova-sdk/latest/kova_sdk/tools/struct.ToolPolicy.html) that confines filesystem access and guards the web tools against SSRF:

```toml
[dependencies]
kova-sdk = { version = "0.9", features = ["web-tools"] }
```

```rust
use std::sync::Arc;
use kova_sdk::tools::{ToolPolicy, register_all_tools_with_policy};

let policy = Arc::new(ToolPolicy {
    workspace_root: Some("/srv/project".into()),
    ..ToolPolicy::default()
});

let mut builder = AgentBuilder::new().provider(provider);
for tool in register_all_tools_with_policy(policy) {
    builder = builder.tool(tool);
}
let agent = builder.build()?;
```

## Documentation

| Document | Contents |
|----------|---------|
| [docs/design.md](docs/design.md) | Architecture, agentic loop, design decisions |
| [docs/api-reference.md](docs/api-reference.md) | Full API with code examples for every module |
| [docs/changelog.md](docs/changelog.md) | Version history |
| [docs/contributing.md](docs/contributing.md) | Conventions, adding providers/tools, test guide |
