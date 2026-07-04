# Contributing

## Conventions

- **No comments by default.** Add one only when the *why* is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific bug.
- **No panics in library code.** All fallible operations must return `Result<_, KovaError>`.
- **All async spans** must be wrapped in `tracing::info_span!` with an `otel.status_code` field.
- **Error types**: use `thiserror 2` derives. Add new variants to `KovaError` rather than inventing a new error type.
- **`Send + Sync`**: every new public type must be `Send + Sync`. Add a compile-time assertion to `tests/send_sync_assertions.rs`.

## Adding a Provider

1. Create `src/provider/<name>/` with `mod.rs`, `config.rs`, `provider.rs`, `convert.rs`, `types.rs`.
2. Implement `LlmProvider` for both `chat_completion` (blocking) and `chat_completion_stream` (streaming).
3. Implement `list_models`.
4. Re-export from `src/provider/mod.rs`.
5. Add integration tests under `tests/`.

## Adding a Tool

Implement the `Tool` trait:

```rust
use kova_sdk::tool::Tool;
use kova_sdk::models::ToolResult;
use kova_sdk::error::KovaError;
use async_trait::async_trait;
use serde_json::Value;

struct MyTool;

#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something useful" }
    fn parameters_schema(&self) -> Value { serde_json::json!({ "type": "object", "properties": {} }) }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        Ok(ToolResult { content: "done".into(), is_error: false })
    }
}
```

Then register it:

```rust
AgentBuilder::new().provider(p).tool(Arc::new(MyTool)).build()?;
// or
registry.register(Arc::new(MyTool)).await;
```

### Built-in tools (`src/tools/`)

The crate ships ready-made filesystem, shell, and web tools behind the `tools`
and `web-tools` features (off by default). When adding or changing one:

- **Keep tools config-agnostic.** Every runtime constraint is injected through
  `ToolPolicy` — tools must not read host configuration or environment-specific
  state directly. A host maps its own config onto `ToolPolicy`/`WebPolicy`, never
  the reverse. This is what lets the SDK ship the tools without depending on any
  downstream crate's config types.
- **Place new tools by weight.** Light deps go in `tools` (`fs.rs`/`shell.rs`);
  anything pulling a heavy dependency tree (HTML parsing, etc.) goes behind
  `web-tools` and is `push`ed conditionally in `register_all_tools_with_policy`.
- **Report tool failures in-band** as `tool_error(...)` results (so the model can
  recover), reserving `Err(KovaError)` for genuine transport faults.
- **Respect the security boundary.** Filesystem tools resolve paths through
  `ToolPolicy::resolve_arg_path` and check containment with
  `resolve_for_containment`; web tools must route through the SSRF validation in
  `web.rs` (never build an unpinned client). Add tests alongside the tool.

## Running Tests

```bash
# All tests
cargo test --workspace

# Library only
cargo test -p kova-sdk

# With OTEL telemetry feature
cargo test -p kova-sdk --features telemetry

# Property-based tests (proptest)
cargo test -p kova-sdk property
```

## Test Structure

| Location | Purpose |
|----------|---------|
| `tests/agent_integration.rs` | Agent + tool-call loop |
| `tests/tool_calling.rs` | Tool invocation and parsing |
| `tests/streaming.rs` | Streaming and SSE parsing |
| `tests/concurrent_registry_tests.rs` | ToolRegistry thread safety |
| `tests/error_propagation.rs` | Error surface |
| `tests/parallel_tool_calling.rs` | Concurrent tool execution |
| `tests/models_property_tests.rs` | Model type serde roundtrips |
| `tests/tool_registry_property_tests.rs` | ToolRegistry lookup invariants |
| `tests/tool_calling_property_tests.rs` | Tool-call loop properties |
| `tests/mcp_property_tests.rs` | MCP tool conversion |
| `tests/provider_property_tests.rs` | Provider behaviour properties |
| `tests/telemetry_property_tests.rs` | Telemetry properties |
| `tests/send_sync_assertions.rs` | Compile-time Send+Sync guards |
| `tests/bedrock_integration.rs` | Bedrock provider (requires AWS) |
| `tests/mock/` | `MockLlmProvider`, mock tools |

## Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `telemetry` | off | Enables OTEL crates; `TelemetryConfig` uses a full pipeline |
| `tools` | off | Built-in filesystem + shell tools (`glob`, `regex`, `diffy`) |
| `web-tools` | off | Adds `fetch_webpage` + the `fetch_text` SSRF-guarded HTTP helper (`scraper`, `dom_smoothie`, `htmd`, `tokio/net`); implies `tools` |

Keep the feature flags in mind when adding new dependencies — OTEL crates and the
web-tool HTML stack are heavy and should not be compiled for users who don't need
them. Run `cargo test --features web-tools` to exercise the tool suite.
