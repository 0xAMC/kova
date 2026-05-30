# API Reference

## AgentBuilder

```rust
AgentBuilder::new()
    .provider(Arc<dyn LlmProvider>)               // required
    .system_prompt(impl Into<String>)             // optional
    .inference_config(InferenceConfig)            // optional (max_tokens, temperature, model)
    .tool(Arc<dyn Tool>)                          // optional, repeatable
    .tool_registry(ToolRegistry)                  // optional (mutually exclusive with .tool())
    .memory(Arc<dyn MemoryStore>)                // optional (default: InMemoryStore unlimited)
    .streaming_handler(Arc<dyn StreamingHandler>) // optional
    .mcp_client(Arc<McpClient>, &str).await?      // optional, async
    .max_iterations(usize)                        // optional (default: 10)
    .max_concurrent_tools(usize)                  // optional (default: 10)
    .with_approval_handler(Arc<dyn ToolApprovalHandler>) // optional
    .with_lifecycle_hook(Arc<dyn ToolLifecycleHook>)     // optional
    .build()?
```

| Method | Required | Default | Notes |
|--------|----------|---------|-------|
| `provider` | yes | — | |
| `system_prompt` | no | `None` | Prepended to every conversation |
| `inference_config` | no | all-`None` | Sets `max_tokens`, `temperature`, `model` for every call |
| `tool` | no | — | Repeatable; mutually exclusive with `tool_registry` |
| `tool_registry` | no | empty | Mutually exclusive with `tool` |
| `memory` | no | `InMemoryStore` (unbounded) | |
| `streaming_handler` | no | `None` | Used by `chat_stream` only |
| `mcp_client` | no | — | Async; discovers tools at build time |
| `max_iterations` | no | `10` | Max tool-call loop iterations |
| `max_concurrent_tools` | no | `10` | Semaphore cap for parallel tool calls |
| `with_approval_handler` | no | `None` | Gates every tool execution; called before `execute` |
| `with_lifecycle_hook` | no | `None` | Observes tool start/end; skipped for denied calls |
| `build()` | — | — | Returns `Result<Agent, KovaError>` |

## Agent

```rust
// Blocking response — waits for full LLM output
agent.chat(conversation_id: &str, user_message: &str) -> Result<String, KovaError>

// Streaming response — delivers chunks to StreamingHandler, returns full text
agent.chat_stream(conversation_id: &str, user_message: &str) -> Result<String, KovaError>

// Input tokens reported by the provider for the most recently completed turn.
// Returns 0 until the first turn completes or if the provider does not report usage.
// Updated atomically after every turn; safe to call from any thread concurrently.
agent.last_turn_input_tokens() -> u32
```

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
    // Azure:
    .with_api_version("2024-02-01")
    .with_chat_completions_path("/openai/deployments/gpt-4/chat/completions")
    .with_models_path("/openai/deployments/models");

let provider = Arc::new(OpenAiCompatibleProvider::new(config)?);
```

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

let provider = Arc::new(GeminiProvider::new(config)?);
```

| Field | Default | Description |
|-------|---------|-------------|
| `model` | required | Gemini model ID, e.g. `"gemini-2.0-flash"` |
| `api_key` | `None` | Sent as `x-goog-api-key` header |
| `timeout` | 60s | Request timeout |
| `base_url` | `https://generativelanguage.googleapis.com` | API base URL |
| `api_version` | `"v1beta"` | URL path segment before `/models/` |

Streaming uses `alt=sse` to reuse the shared SSE parser. Chain-of-thought parts from thinking models (`thought: true`) are filtered from user-visible output; `thoughtSignature` is preserved as `provider_metadata` on tool-use blocks.

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
}).await?);

// HTTP+SSE transport
let client = Arc::new(McpClient::connect(McpTransport::HttpSse {
    url: "http://localhost:8080".into(),
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
        if let StreamEvent::ContentDelta { text } = event { print!("{text}"); }
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
use kova_sdk::error::KovaError;

match result {
    Err(KovaError::Provider { message, status_code }) => { /* LLM API error */ }
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

## Data Types

| Type | Description |
|------|-------------|
| `Role` | `User`, `Assistant`, `System`, `Tool` |
| `ContentBlock` | `Text { text }`, `ToolUse { id, name, input }`, `ToolResult { tool_use_id, content, is_error }` |
| `ConversationMessage` | `role: Role` + `content: Vec<ContentBlock>` |
| `ModelResponse` | `content`, `stop_reason`, `usage: Option<UsageStats>` |
| `StopReason` | `EndTurn`, `ToolUse`, `MaxTokens`, `Unknown(String)` |
| `UsageStats` | `input_tokens`, `output_tokens`, `total_tokens` |
| `InferenceConfig` | `model`, `max_tokens`, `temperature` (all `Option`) |
| `ToolDefinition` | `name`, `description`, `parameters` (JSON Schema `Value`) |
| `ToolResult` | `content: String`, `is_error: bool` |
| `StreamEvent` | `ContentDelta { text }`, `ToolUseDelta { … }`, `StopEvent`, `Error`, `UsageEvent { input_tokens, output_tokens }` |
| `ModelInfo` | `id`, `object`, `created`, `owned_by` |
