# API Reference

## AgentBuilder

```rust
AgentBuilder::new()
    .provider(Arc<dyn LlmProvider>)               // required
    .system_prompt(impl Into<String>)             // optional
    .inference_config(InferenceConfig)            // optional (model, max_tokens, temperature, top_p, stop_sequences)
    .tool(Arc<dyn Tool>)                          // optional, repeatable
    .tool_registry(ToolRegistry)                  // optional (composes with .tool())
    .memory(Arc<dyn MemoryStore>)                // optional (default: InMemoryStore unlimited)
    .streaming_handler(Arc<dyn StreamingHandler>) // optional
    .mcp_client(Arc<McpClient>, &str).await?      // optional, async
    .max_iterations(usize)                        // optional (default: 10)
    .max_concurrent_tools(usize)                  // optional (default: 10)
    .retry_config(RetryConfig)                    // optional (default: 2 retries, exp backoff)
    .metrics(Arc<MetricsCollector>)               // optional
    .with_approval_handler(Arc<dyn ToolApprovalHandler>) // optional
    .with_lifecycle_hook(Arc<dyn ToolLifecycleHook>)     // optional
    .build()?
```

| Method | Required | Default | Notes |
|--------|----------|---------|-------|
| `provider` | yes | — | |
| `system_prompt` | no | `None` | Prepended to every conversation |
| `inference_config` | no | all-`None` | Sets `max_tokens`, `temperature`, `model` for every call |
| `tool` | no | — | Repeatable; merged into `tool_registry` on build |
| `tool_registry` | no | empty | Composes with `tool` |
| `memory` | no | `InMemoryStore` (unbounded) | |
| `streaming_handler` | no | `None` | Used by `chat_stream` only |
| `mcp_client` | no | — | Async; discovers tools at build time |
| `max_iterations` | no | `10` | Max tool-call loop iterations |
| `max_concurrent_tools` | no | `10` | Semaphore cap for parallel tool calls |
| `retry_config` | no | 2 retries | Exponential backoff on transient errors; `RetryConfig::disabled()` to turn off |
| `metrics` | no | `None` | Agent records LLM latency/tokens/errors + tool durations |
| `with_approval_handler` | no | `None` | Gates every tool execution; called before `execute` |
| `with_lifecycle_hook` | no | `None` | Observes tool start/end; skipped for denied calls |
| `build()` | — | — | Returns `Result<Agent, KovaError>` |

## Agent

### Stateless API (core)

```rust
// One agentic turn over caller-owned history. No memory-store involvement.
agent.run(messages: &[ConversationMessage]) -> Result<AgentResponse, KovaError>

// Same, with per-call InferenceConfig overrides (unset fields fall back to agent defaults).
agent.run_with_config(messages, overrides: InferenceConfig) -> Result<AgentResponse, KovaError>

// Pull-based streaming: TextDelta / ThinkingDelta / ToolCallStarted /
// ToolCallFinished / TurnCompleted { response }.
agent.run_stream(messages) -> impl Stream<Item = Result<AgentEvent, KovaError>>

// Cancellable variants: the token (prelude re-export of tokio-util's
// CancellationToken) aborts the turn mid-provider-call or mid-tool-execution
// with KovaError::Cancelled; no messages are produced.
agent.run_cancellable(messages, cancel: CancellationToken) -> Result<AgentResponse, KovaError>
agent.run_stream_cancellable(messages, cancel: CancellationToken) -> impl Stream<...>
```

```rust
pub struct AgentResponse {
    pub text: String,                          // final assistant text
    pub new_messages: Vec<ConversationMessage>, // everything the turn produced — persist this
    pub stop_reason: StopReason,
    pub usage: UsageStats,                     // summed across all provider calls in the turn
    pub llm_calls: u64,
    pub thinking: Option<String>,
}
```

Typical stateless usage:

```rust
let mut history = vec![/* user message */];
let response = agent.run(&history).await?;
history.extend(response.new_messages);
```

### Session API (memory-backed convenience)

```rust
// Blocking response — waits for full LLM output
agent.chat(conversation_id: &str, user_message: &str) -> Result<String, KovaError>

// Like chat, but returns the full AgentResponse
agent.chat_response(conversation_id: &str, user_message: &str) -> Result<AgentResponse, KovaError>

// Streaming response — delivers chunks to StreamingHandler, returns full text
agent.chat_stream(conversation_id: &str, user_message: &str) -> Result<String, KovaError>

// Input tokens reported by the provider for the most recently completed turn.
agent.last_turn_input_tokens() -> u32
```

Session writes are transactional per turn: the user message and produced messages are
persisted only after the turn succeeds, so a failed turn never corrupts the conversation.

## Providers

### OpenAI-Compatible

```rust
use kova_sdk::provider::openai::{OpenAiCompatibleProvider, OpenAiProviderConfig};
use std::time::Duration;

let config = OpenAiProviderConfig::new("https://api.openai.com", "gpt-4")
    .with_api_key("sk-...")
    .with_timeout(Duration::from_secs(60))
    .with_max_tokens(4096)
    .with_temperature(0.7)
    // OpenAI o-series reasoning models:
    .with_reasoning_effort("high")
    // Azure:
    .with_api_version("2024-02-01")
    .with_chat_completions_path("/openai/deployments/gpt-4/chat/completions")
    .with_models_path("/openai/deployments/models");

let provider = Arc::new(OpenAiCompatibleProvider::new(config)?);
```

### Anthropic (native Messages API)

```rust
use kova_sdk::provider::anthropic::{AnthropicProvider, AnthropicProviderConfig};

let config = AnthropicProviderConfig::new("claude-opus-4-8")
    .with_api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .with_effort("high");          // optional: output_config.effort
    // .with_adaptive_thinking(false) // thinking: {"type":"adaptive"} is on by default
    // .with_cache(false)             // automatic prompt caching is on by default

let provider = Arc::new(AnthropicProvider::new(config)?);
```

Automatic prompt caching places a top-level `cache_control: {"type": "ephemeral"}`
on every request, so the stable prefix (system prompt, tools, prior turns) is
served from Anthropic's cache; per-call reads/writes surface as
`UsageStats::cache_read_tokens` / `cache_creation_tokens`. Signed thinking
blocks round-trip through history as `ContentBlock::Thinking` — required for
tool loops with extended thinking.

| Field | Default | Description |
|-------|---------|-------------|
| `base_url` | required | API base URL |
| `model` | required | Model identifier |
| `api_key` | `None` | Bearer token |
| `timeout` | 30s | Request timeout |
| `max_tokens` | `None` | Max completion tokens |
| `temperature` | `None` | Sampling temperature |
| `chat_completions_path` | `/v1/chat/completions` | Chat endpoint path |
| `models_path` | `/v1/models` | Models list endpoint path |
| `api_version` | `None` | Query param (e.g. Azure `api-version`) |
| `reasoning_effort` | `None` | `"low"` / `"medium"` / `"high"` — o-series models only |

### AWS Bedrock

```rust
use kova_sdk::provider::bedrock::{BedrockProvider, BedrockProviderConfig};

// Default credential chain
let config = BedrockProviderConfig::new("us-east-1", "anthropic.claude-sonnet-4-20250514-v1:0");
let provider = Arc::new(BedrockProvider::new(config).await?);

// Named profile
let config = BedrockProviderConfig::new("us-west-2", "anthropic.claude-sonnet-4-20250514-v1:0")
    .with_profile("my-profile");

// Explicit credentials
let config = BedrockProviderConfig::new("us-east-1", "anthropic.claude-sonnet-4-20250514-v1:0")
    .with_credentials("AKIA...", "secret", Some("session-token".into()));

// Custom endpoint (LocalStack)
let config = BedrockProviderConfig::new("us-east-1", "my-model")
    .with_endpoint_url("http://localhost:4566");

// Extended thinking on Claude models (passes additionalModelRequestFields)
let config = BedrockProviderConfig::new("us-east-1", "anthropic.claude-sonnet-4-20250514-v1:0")
    .with_additional_model_request_fields(serde_json::json!({ "budgetTokens": 5000 }));
```

**Credential resolution order:** explicit → named profile → default AWS chain.

| Field | Default | Description |
|-------|---------|-------------|
| `region` | required | AWS region |
| `model_id` | required | Bedrock model ID |
| `profile` | `None` | AWS named profile |
| `access_key_id` | `None` | Explicit access key |
| `secret_access_key` | `None` | Explicit secret key |
| `session_token` | `None` | STS session token |
| `timeout` | 60s | Request timeout |
| `endpoint_url` | `None` | Override endpoint |
| `additional_model_request_fields` | `None` | Arbitrary JSON passed as `additionalModelRequestFields` |

### Google Gemini

```rust
use kova_sdk::provider::gemini::{GeminiProvider, GeminiProviderConfig};
use std::time::Duration;

let config = GeminiProviderConfig::new("gemini-2.0-flash")
    .with_api_key("AIza...")
    .with_timeout(Duration::from_secs(60))
    // Override base URL for testing with a mock server:
    .with_base_url("http://localhost:8080")
    // Override API version (defaults to "v1beta"):
    .with_api_version("v1beta");

// Enable extended thinking on thinking-capable models:
// -1 = dynamic/unlimited, 0 = disabled (default), positive = token cap
let config = GeminiProviderConfig::new("gemini-2.5-flash")
    .with_api_key("AIza...")
    .with_thinking_budget(5000);

let provider = Arc::new(GeminiProvider::new(config)?);
```

| Field | Default | Description |
|-------|---------|-------------|
| `model` | required | Gemini model ID, e.g. `"gemini-2.0-flash"` |
| `api_key` | `None` | Sent as `x-goog-api-key` header |
| `timeout` | 60s | Request timeout |
| `base_url` | `https://generativelanguage.googleapis.com` | API base URL |
| `api_version` | `"v1beta"` | URL path segment before `/models/` |
| `thinking_budget` | `None` (disabled) | Token budget for extended thinking: `-1` unlimited, `0` off, positive = cap |

Streaming uses `alt=sse` to reuse the shared SSE parser. Chain-of-thought parts from thinking models (`thought: true`) are filtered from user-visible `content` and placed in `ModelResponse::thinking`; `thoughtSignature` is preserved as `provider_metadata` on tool-use blocks.

### Ollama

```rust
use kova_sdk::provider::ollama::{OllamaProvider, OllamaProviderConfig, OllamaThink};

// Local instance (default: http://localhost:11434)
let config = OllamaProviderConfig::new("llama3.2");
let provider = Arc::new(OllamaProvider::new(config)?);

// Remote instance
let config = OllamaProviderConfig::new("llama3.2")
    .with_base_url("http://my-server:11434")
    .with_timeout(Duration::from_secs(180))
    .with_keep_alive("10m");

// Thinking-capable model (qwen3, deepseek-r1, etc.)
let config = OllamaProviderConfig::new("qwen3")
    .with_think(OllamaThink::High);

// Extra generation options
use serde_json::json;
let mut opts = serde_json::Map::new();
opts.insert("temperature".into(), json!(0.7));
opts.insert("num_ctx".into(), json!(8192));
let config = OllamaProviderConfig::new("llama3.2")
    .with_extra_options(opts);
```

| Field | Default | Description |
|-------|---------|-------------|
| `model` | required | Ollama model identifier, e.g. `"llama3.2"`, `"qwen3"`, `"deepseek-r1"` |
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `timeout` | 120s | Request timeout (local inference can be slow) |
| `keep_alive` | `None` | Model keep-alive duration, e.g. `"5m"` or `"0"` to unload immediately |
| `think` | `None` | `OllamaThink::Enabled` / `High` / `Medium` / `Low` — requires a thinking model |
| `extra_options` | `None` | Merged into the `options` object (e.g. `top_k`, `seed`, `num_ctx`) |

No API key is required. Streaming uses NDJSON over `/api/chat`; `list_models` uses `/api/tags`. Tool calls are supported. When `think` is set, reasoning text arrives as `StreamEvent::ThinkingDelta`.

### Custom Provider

```rust
use kova_sdk::provider::LlmProvider;
use kova_sdk::models::*;
use kova_sdk::error::KovaError;
use async_trait::async_trait;
use std::pin::Pin;
use futures::Stream;

#[async_trait]
impl LlmProvider for MyProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> { todo!() }

    async fn chat_completion_stream(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError> {
        todo!()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> { todo!() }
}
```

## Tools

### Defining a Tool

```rust
use kova_sdk::tool::Tool;
use kova_sdk::models::ToolResult;
use kova_sdk::error::KovaError;
use async_trait::async_trait;
use serde_json::{json, Value};

struct GetWeather;

#[async_trait]
impl Tool for GetWeather {
    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> &str { "Get current weather for a city" }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(ToolResult { content: format!("72°F, sunny in {city}"), is_error: false })
    }
}
```

### ToolRegistry

```rust
use kova_sdk::tool::registry::ToolRegistry;

let registry = ToolRegistry::new();
registry.register(Arc::new(GetWeather)).await;

let tool  = registry.get("get_weather").await;           // Option<Arc<dyn Tool>>
let names = registry.list().await;                        // Vec<String>
let defs  = registry.tool_definitions().await;           // Vec<ToolDefinition> for the LLM
```

### Built-in Tools (`tools` / `web-tools` features)

`kova_sdk::tools` ships ready-made tools. Enable the feature and register them
against an injected `ToolPolicy`:

```toml
kova-sdk = { version = "0.3", features = ["web-tools"] }
```

| Feature | Tools |
|---------|-------|
| `tools` | `read_file`, `list_dir`, `search`, `edit_file`, `write_file`, `patch_file`, `shell` |
| `web-tools` (implies `tools`) | adds `fetch_webpage` and the `fetch_text` SSRF-guarded HTTP helper |

```rust
use std::sync::Arc;
use std::time::Duration;
use kova_sdk::tools::{ToolPolicy, WebPolicy, register_all_tools_with_policy};

let policy = Arc::new(ToolPolicy {
    workspace_root: Some("/srv/project".into()), // file tools confined here; shell runs here
    protected_paths: vec!["/srv/project/.secrets".into()], // never read/written
    shell_timeout: Duration::from_secs(60),
    web: WebPolicy::default(),                    // HTTPS-only, no private hosts, capped
});

let mut builder = AgentBuilder::new().provider(provider);
for tool in register_all_tools_with_policy(policy) {
    builder = builder.tool(tool);
}
let agent = builder.build()?;
```

`register_all_tools()` is a shorthand that uses `ToolPolicy::default()` (unconfined
— suitable only for trusted, non-interactive use).

**Policy fields**

| Field | Effect |
|-------|--------|
| `workspace_root: Option<PathBuf>` | Confines file reads/writes; relative paths resolve here; `shell` cwd. `None` = unconfined |
| `protected_paths: Vec<PathBuf>` | Directories the file tools may never touch (host config, secrets) |
| `shell_timeout: Duration` | Wall-clock cap per `shell` call; the child is killed on expiry |
| `web: WebPolicy` | `allow_private_hosts`, `https_only`, `allowed_urls`/`denied_urls` globs, byte/char caps, `default_format`, `user_agent` |

The tools are **config-agnostic**: all constraints come from `ToolPolicy`, so a
host maps its own configuration onto these structs rather than the tools reaching
into host state. The web tools guard against SSRF (private-address rejection +
DNS-pinned client) and confine output size; the file tools reject paths that
escape `workspace_root` via `..` or symlinks.

## Memory

```rust
use kova_sdk::memory::in_memory::InMemoryStore;

let store = InMemoryStore::new();                   // Unbounded
let store = InMemoryStore::with_max_messages(100);  // Capped; preserves system prompt on truncation
```

Pass to the builder: `.memory(Arc::new(store))`.

## MCP

```rust
use kova_sdk::mcp::{McpClient, McpTransport};

// Stdio transport (spawn subprocess)
let client = Arc::new(McpClient::connect(McpTransport::Stdio {
    command: "npx".into(),
    args: vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
    env: Default::default(),
}).await?);

// HTTP+SSE transport (legacy: static headers, no initialize handshake)
let client = Arc::new(McpClient::connect(McpTransport::HttpSse {
    url: "http://localhost:8080".into(),
    headers: Default::default(),
}).await?);

// Streamable HTTP transport (MCP 2025: handshake, Mcp-Session-Id, JSON or SSE)
let client = Arc::new(McpClient::connect(McpTransport::StreamableHttp {
    url: "https://mcp.example.com/mcp".into(),
    headers: Default::default(),
    auth: None, // or Some(Arc::new(my_token_provider)) for OAuth
}).await?);

// Discover tools / call a tool directly
let tools = client.tools_list().await?;
let (content, is_error) = client.tools_call("read_file", json!({"path": "/tmp/test.txt"})).await?;

// Register with agent
let agent = AgentBuilder::new()
    .provider(provider)
    .mcp_client(client, "my_server").await?
    .build()?;
```

### Authenticated transports (`TokenProvider`)

`StreamableHttp` can carry a refreshable bearer token. Implement `TokenProvider`
to supply it; the transport attaches `Authorization: Bearer <token()>` to each
request and, on a `401`, calls `refresh()` once and retries. kova owns no OAuth
logic — the host drives the flow and persists tokens.

```rust
use kova_sdk::mcp::TokenProvider;

#[async_trait::async_trait]
impl TokenProvider for MyTokenStore {
    async fn token(&self) -> Result<String, KovaError> { /* current bearer */ }
    async fn refresh(&self) -> Result<String, KovaError> { /* refresh + persist */ }
}
```

### Resilience

- `client.reconnect()` re-establishes the connection from the stored transport (respawns stdio children, re-runs the `initialize` handshake) and clears the tools/list cache.
- `tools_list()`/`tools_call()` auto-recover: a dead transport (`KovaError::Connection`) triggers one reconnect-and-retry. Server-reported JSON-RPC errors (`KovaError::Mcp`) never do.
- `tools_list()` results are cached until reconnect.
- `tools_call_with_timeout(name, args, timeout)` overrides the client's default per-request timeout for one call.
- `connect()` retries transient failures once (500ms backoff).

## Streaming

```rust
use kova_sdk::streaming::StreamingHandler;
use kova_sdk::models::StreamEvent;
use kova_sdk::error::KovaError;
use async_trait::async_trait;

struct MyHandler;

#[async_trait]
impl StreamingHandler for MyHandler {
    async fn on_chunk(&self, event: &StreamEvent) -> Result<(), KovaError> {
        match event {
            StreamEvent::ThinkingDelta { text } => eprint!("{text}"),   // reasoning output
            StreamEvent::ContentDelta { text } => print!("{text}"),     // visible response
            _ => {}
        }
        Ok(())
    }
    async fn on_complete(&self) -> Result<(), KovaError> { println!(); Ok(()) }
    async fn on_error(&self, error: &KovaError) { eprintln!("Stream error: {error}"); }
}

let agent = AgentBuilder::new()
    .provider(provider)
    .streaming_handler(Arc::new(MyHandler))
    .build()?;

let text = agent.chat_stream("conv-1", "Tell me a story").await?;
```

## Orchestrator

```rust
use kova_sdk::orchestrator::{Orchestrator, OrchestratorPattern, OrchestratorOutput};
use std::collections::HashMap;
use std::time::Duration;

let mut agents = HashMap::new();
agents.insert("summarizer".into(), Arc::new(summarizer_agent));
agents.insert("translator".into(), Arc::new(translator_agent));
agents.insert("router".into(),     Arc::new(router_agent));

let orch = Orchestrator::new(agents, Duration::from_secs(120));

// Sequential
let OrchestratorOutput::Single(text) = orch.execute(
    OrchestratorPattern::Sequential(vec!["summarizer".into(), "translator".into()]),
    "Long article...",
).await? else { unreachable!() };

// Parallel
let OrchestratorOutput::Parallel(result) = orch.execute(
    OrchestratorPattern::Parallel(vec!["summarizer".into(), "translator".into()]),
    "Analyze this...",
).await? else { unreachable!() };
for (name, text) in &result.successes { println!("{name}: {text}"); }
for (name, err)  in &result.failures  { eprintln!("{name}: {err}"); }

// Router
orch.execute(
    OrchestratorPattern::Router {
        router_agent: "router".into(),
        downstream: vec!["summarizer".into(), "translator".into()],
    },
    "Translate this to French...",
).await?;
```

## Telemetry

```rust
use kova_sdk::telemetry::{TelemetryConfig, ExporterConfig, OtlpProtocol};

// Basic (no telemetry feature needed)
TelemetryConfig::builder().log_level(tracing::Level::DEBUG).build().init()?;

// OTLP gRPC (requires `telemetry` feature)
TelemetryConfig::builder()
    .log_level(tracing::Level::INFO)
    .exporter(ExporterConfig::Otlp {
        endpoint: "http://localhost:4317".into(),
        protocol: OtlpProtocol::Grpc,
    })
    .sampling_rate(0.5)
    .build()
    .init()?;

// MetricsCollector — always available
use kova_sdk::telemetry::MetricsCollector;

let m = MetricsCollector::new();
m.record_llm_request(150.0, 100, 50);   // latency_ms, input_tokens, output_tokens
m.record_tool_invocation(25.0, true);   // duration_ms, success
m.record_llm_error();

println!("requests: {}", m.llm_request_count());
println!("tokens:   {}", m.total_tokens());
println!("errors:   {}", m.error_count());
```

## Error Handling

```rust
use kova_sdk::error::{KovaError, ProviderErrorClass};

match result {
    Err(KovaError::Provider { message, status_code, class }) => { /* LLM API error */ }
    Err(KovaError::Connection(msg))                  => { /* Network unreachable */ }
    Err(KovaError::ToolExecution { tool_name, .. })  => { /* Tool panicked/failed */ }
    Err(KovaError::ToolNotFound(name))               => { /* LLM called unknown tool */ }
    Err(KovaError::Timeout(duration))                => { /* Request timed out */ }
    Err(KovaError::Mcp(msg))                         => { /* MCP protocol error */ }
    Err(KovaError::Memory(msg))                      => { /* Memory store error */ }
    Err(KovaError::Orchestration(msg))               => { /* Orchestrator error */ }
    Err(KovaError::Build(msg))                       => { /* Builder misconfiguration */ }
    Err(KovaError::Stream(msg))                      => { /* Streaming error */ }
    Err(KovaError::MaxIterations(n))                 => { /* Tool loop hit cap */ }
    Err(KovaError::Serialization(e))                 => { /* JSON error */ }
    Err(KovaError::Io(e))                            => { /* I/O error */ }
    Err(KovaError::Http(e))                          => { /* HTTP client error */ }
    _ => {}
}
```

### Provider error classification

`KovaError::Provider` carries a `class: ProviderErrorClass` so applications can
react precisely without inspecting messages or status codes:

```rust
match err.provider_class() {
    Some(ProviderErrorClass::AuthInvalid)              => { /* 401 — prompt for a new key */ }
    Some(ProviderErrorClass::AuthForbidden)            => { /* 403 — key lacks access */ }
    Some(ProviderErrorClass::RateLimited { retry_after }) => { /* 429 — back off */ }
    Some(ProviderErrorClass::Overloaded)               => { /* 408/5xx/529 — transient, retryable */ }
    Some(ProviderErrorClass::InvalidRequest)           => { /* 400/413/422 — incl. context length */ }
    Some(ProviderErrorClass::NotFound)                 => { /* 404 — unknown model/endpoint */ }
    Some(ProviderErrorClass::Other) | None             => { /* everything else */ }
}
```

Constructors: `KovaError::provider_http(status, retry_after, message)` (classifies
from the status), `KovaError::provider_invalid(message)` (malformed responses),
`KovaError::provider_auth(message)` (pre-HTTP credential failures, e.g. an empty
AWS credential chain). `err.is_retryable()` is true for `RateLimited` and
`Overloaded` classes plus connection failures and timeouts. Bedrock exception
types (`ThrottlingException`, `AccessDeniedException`, …) are normalized to
statuses before classification.

## Data Types

| Type | Description |
|------|-------------|
| `Role` | `User`, `Assistant`, `System`, `Tool` |
| `ContentBlock` | `Text { text }`, `ToolUse { id, name, input }`, `ToolResult { tool_use_id, content, is_error }` |
| `ConversationMessage` | `role: Role` + `content: Vec<ContentBlock>` |
| `ModelResponse` | `content`, `stop_reason`, `usage: Option<UsageStats>`, `thinking: Option<String>` |
| `StopReason` | `EndTurn`, `ToolUse`, `MaxTokens`, `Unknown(String)` |
| `UsageStats` | `input_tokens`, `output_tokens`, `total_tokens` |
| `InferenceConfig` | `model`, `max_tokens`, `temperature` (all `Option`) |
| `ToolDefinition` | `name`, `description`, `parameters` (JSON Schema `Value`) |
| `ToolResult` | `content: String`, `is_error: bool` |
| `StreamEvent` | `ContentDelta { text }`, `ThinkingDelta { text }`, `ToolUseDelta { … }`, `StopEvent`, `Error`, `UsageEvent { input_tokens, output_tokens }` |
| `ModelInfo` | `id`, `object`, `created`, `owned_by` |
