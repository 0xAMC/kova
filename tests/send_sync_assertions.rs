//! Compile-time Send+Sync assertions for all public types that hold
//! shared state.
//!
//! These tests verify at compile time that the SDK's core types satisfy
//! the `Send + Sync` bounds required for safe concurrent usage across
//! async tasks and threads.

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn all_public_types_are_send_sync() {
    // Core agent types
    assert_send_sync::<kova::agent::Agent>();
    assert_send_sync::<kova::agent::AgentBuilder>();

    // Provider types
    assert_send_sync::<kova::provider::openai::OpenAiCompatibleProvider>();
    assert_send_sync::<kova::provider::openai::OpenAiProviderConfig>();

    // Tool registry
    assert_send_sync::<kova::tool::registry::ToolRegistry>();

    // Memory store
    assert_send_sync::<kova::memory::in_memory::InMemoryStore>();

    // Orchestrator
    assert_send_sync::<kova::orchestrator::Orchestrator>();

    // MCP client
    assert_send_sync::<kova::mcp::McpClient>();
}
