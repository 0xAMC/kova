mod mock;

use std::sync::Arc;

use kova_sdk::agent::AgentBuilder;
use kova_sdk::error::KovaError;
use kova_sdk::models::*;

use mock::provider::{MockLlmProvider, make_text_response};

// ── Basic chat loop ────────────────────────────────────────────────

#[tokio::test]
async fn agent_chat_returns_assistant_text() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        "Hello from the LLM!",
    )));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .expect("build should succeed");

    let reply = agent.chat("conv-1", "Hi there").await.unwrap();
    assert_eq!(reply, "Hello from the LLM!");
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test]
async fn agent_chat_with_system_prompt() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(
        "I am helpful.",
    )));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .system_prompt("You are a helpful assistant.")
        .build()
        .expect("build should succeed");

    let reply = agent.chat("conv-1", "Who are you?").await.unwrap();
    assert_eq!(reply, "I am helpful.");
}

#[tokio::test]
async fn agent_chat_multiple_turns() {
    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_text_response("First reply"),
        make_text_response("Second reply"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .unwrap();

    let r1 = agent.chat("conv-1", "msg 1").await.unwrap();
    let r2 = agent.chat("conv-1", "msg 2").await.unwrap();

    assert_eq!(r1, "First reply");
    assert_eq!(r2, "Second reply");
    assert_eq!(provider.call_count(), 2);
}

#[tokio::test]
async fn agent_chat_empty_content_returns_empty_string() {
    // A response with no text content blocks should return "".
    let resp = ModelResponse {
        content: vec![],
        stop_reason: StopReason::EndTurn,
        usage: None,
        thinking: None,
    };

    let provider = Arc::new(MockLlmProvider::with_response(resp));

    let agent = AgentBuilder::new().provider(provider).build().unwrap();

    let reply = agent.chat("conv-1", "hello").await.unwrap();
    assert_eq!(reply, "");
}

// ── AgentBuilder validation ────────────────────────────────────────

#[test]
fn agent_builder_fails_without_provider() {
    let result = AgentBuilder::new().build();

    match result {
        Err(KovaError::Build(msg)) => {
            assert!(
                msg.contains("LlmProvider"),
                "error should mention LlmProvider, got: {msg}"
            );
        }
        Err(other) => panic!("expected KovaError::Build, got: {other:?}"),
        Ok(_) => panic!("expected build to fail without a provider"),
    }
}

#[test]
fn agent_builder_succeeds_with_provider() {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response("ok")));
    let result = AgentBuilder::new().provider(provider).build();
    assert!(result.is_ok());
}

// ── Compile-time Send + Sync assertions ────────────────────────────

#[test]
fn _assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<kova_sdk::agent::Agent>();
    assert_send_sync::<kova_sdk::provider::openai::OpenAiCompatibleProvider>();
}
