mod mock;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use kova::agent::{Agent, AgentBuilder};
use kova::error::KovaError;
use kova::orchestrator::{Orchestrator, OrchestratorOutput, OrchestratorPattern};

use mock::provider::{MockLlmProvider, make_text_response};

/// Build a simple agent that always returns the given text.
fn build_agent(text: &str) -> Arc<Agent> {
    let provider = Arc::new(MockLlmProvider::with_response(make_text_response(text)));
    Arc::new(AgentBuilder::new().provider(provider).build().unwrap())
}

/// Build an agent whose provider returns an error on every call.
fn build_failing_agent() -> Arc<Agent> {
    let provider = Arc::new(FailingMockProvider);
    Arc::new(AgentBuilder::new().provider(provider).build().unwrap())
}

/// Build an agent that sleeps for the given duration before responding.
fn build_slow_agent(delay: Duration, text: &str) -> Arc<Agent> {
    let provider = Arc::new(SlowMockProvider {
        delay,
        response: make_text_response(text),
    });
    Arc::new(AgentBuilder::new().provider(provider).build().unwrap())
}

// ── Helper providers ───────────────────────────────────────────────

/// A provider that always returns a Provider error.
struct FailingMockProvider;

#[async_trait::async_trait]
impl kova::provider::LlmProvider for FailingMockProvider {
    async fn chat_completion(
        &self,
        _messages: &[kova::models::ConversationMessage],
        _tools: &[kova::models::ToolDefinition],
        _config: &kova::models::InferenceConfig,
    ) -> Result<kova::models::ModelResponse, KovaError> {
        Err(KovaError::Provider {
            message: "intentional failure".into(),
            status_code: Some(500),
        })
    }

    async fn chat_completion_stream(
        &self,
        _messages: &[kova::models::ConversationMessage],
        _tools: &[kova::models::ToolDefinition],
        _config: &kova::models::InferenceConfig,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<kova::models::StreamEvent, KovaError>> + Send>,
        >,
        KovaError,
    > {
        Err(KovaError::Stream("not implemented".into()))
    }

    async fn list_models(&self) -> Result<Vec<kova::models::ModelInfo>, KovaError> {
        Ok(vec![])
    }
}

/// A provider that sleeps before returning a response.
struct SlowMockProvider {
    delay: Duration,
    response: kova::models::ModelResponse,
}

#[async_trait::async_trait]
impl kova::provider::LlmProvider for SlowMockProvider {
    async fn chat_completion(
        &self,
        _messages: &[kova::models::ConversationMessage],
        _tools: &[kova::models::ToolDefinition],
        _config: &kova::models::InferenceConfig,
    ) -> Result<kova::models::ModelResponse, KovaError> {
        tokio::time::sleep(self.delay).await;
        Ok(self.response.clone())
    }

    async fn chat_completion_stream(
        &self,
        _messages: &[kova::models::ConversationMessage],
        _tools: &[kova::models::ToolDefinition],
        _config: &kova::models::InferenceConfig,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<kova::models::StreamEvent, KovaError>> + Send>,
        >,
        KovaError,
    > {
        Err(KovaError::Stream("not implemented".into()))
    }

    async fn list_models(&self) -> Result<Vec<kova::models::ModelInfo>, KovaError> {
        Ok(vec![])
    }
}

// ── Sequential pipeline ────────────────────────────────────────────

#[tokio::test]
async fn sequential_pipeline_chains_output() {
    // Agent A echoes "A:<input>", B echoes "B:<input>", C echoes "C:<input>".
    // With chaining: input → A → B → C, each agent sees the previous output.
    // Since mock providers ignore input and return fixed text, we verify
    // the final output is from the last agent in the chain.
    let agents: HashMap<String, Arc<Agent>> = HashMap::from([
        ("a".into(), build_agent("output-from-a")),
        ("b".into(), build_agent("output-from-b")),
        ("c".into(), build_agent("output-from-c")),
    ]);

    let orch = Orchestrator::new(agents, Duration::from_secs(5));
    let result = orch
        .execute(
            OrchestratorPattern::Sequential(vec!["a".into(), "b".into(), "c".into()]),
            "initial-input",
        )
        .await
        .unwrap();

    match result {
        OrchestratorOutput::Single(output) => {
            assert_eq!(output, "output-from-c");
        }
        OrchestratorOutput::Parallel(_) => panic!("expected Single output"),
    }
}

// ── Parallel execution ─────────────────────────────────────────────

#[tokio::test]
async fn parallel_execution_all_receive_same_input() {
    let agents: HashMap<String, Arc<Agent>> = HashMap::from([
        ("a".into(), build_agent("reply-a")),
        ("b".into(), build_agent("reply-b")),
        ("c".into(), build_agent("reply-c")),
    ]);

    let orch = Orchestrator::new(agents, Duration::from_secs(5));
    let result = orch
        .execute(
            OrchestratorPattern::Parallel(vec!["a".into(), "b".into(), "c".into()]),
            "shared-input",
        )
        .await
        .unwrap();

    match result {
        OrchestratorOutput::Parallel(par) => {
            assert_eq!(par.failures.len(), 0, "no failures expected");
            assert_eq!(par.successes.len(), 3, "all 3 agents should succeed");

            // Verify all agents produced their expected outputs.
            let mut outputs: Vec<String> = par.successes.iter().map(|(_, v)| v.clone()).collect();
            outputs.sort();
            assert_eq!(outputs, vec!["reply-a", "reply-b", "reply-c"]);
        }
        OrchestratorOutput::Single(_) => panic!("expected Parallel output"),
    }
}

// ── Routing ────────────────────────────────────────────────────────

#[tokio::test]
async fn routing_selects_correct_downstream_agent() {
    // The router agent returns the name of the downstream agent to use.
    let router = build_agent("handler-b");
    let handler_a = build_agent("handled-by-a");
    let handler_b = build_agent("handled-by-b");

    let agents: HashMap<String, Arc<Agent>> = HashMap::from([
        ("router".into(), router),
        ("handler-a".into(), handler_a),
        ("handler-b".into(), handler_b),
    ]);

    let orch = Orchestrator::new(agents, Duration::from_secs(5));
    let result = orch
        .execute(
            OrchestratorPattern::Router {
                router_agent: "router".into(),
                downstream: vec!["handler-a".into(), "handler-b".into()],
            },
            "route me",
        )
        .await
        .unwrap();

    match result {
        OrchestratorOutput::Single(output) => {
            assert_eq!(output, "handled-by-b");
        }
        OrchestratorOutput::Parallel(_) => panic!("expected Single output"),
    }
}

// ── Sequential failure stops pipeline ──────────────────────────────

#[tokio::test]
async fn sequential_failure_stops_pipeline() {
    // Agent "a" succeeds, "b" fails, "c" should never run.
    let agents: HashMap<String, Arc<Agent>> = HashMap::from([
        ("a".into(), build_agent("ok-from-a")),
        ("b".into(), build_failing_agent()),
        ("c".into(), build_agent("should-not-reach")),
    ]);

    let orch = Orchestrator::new(agents, Duration::from_secs(5));
    let result = orch
        .execute(
            OrchestratorPattern::Sequential(vec!["a".into(), "b".into(), "c".into()]),
            "start",
        )
        .await;

    match result {
        Err(KovaError::Orchestration(msg)) => {
            assert!(
                msg.contains("'b'"),
                "error should mention failing agent 'b', got: {msg}"
            );
        }
        Err(other) => panic!("expected Orchestration error, got: {other:?}"),
        Ok(_) => panic!("expected pipeline to fail"),
    }
}

// ── Parallel partial failure ───────────────────────────────────────

#[tokio::test]
async fn parallel_partial_failure_collects_all_results() {
    let agents: HashMap<String, Arc<Agent>> = HashMap::from([
        ("ok-1".into(), build_agent("success-1")),
        ("fail".into(), build_failing_agent()),
        ("ok-2".into(), build_agent("success-2")),
    ]);

    let orch = Orchestrator::new(agents, Duration::from_secs(5));
    let result = orch
        .execute(
            OrchestratorPattern::Parallel(vec!["ok-1".into(), "fail".into(), "ok-2".into()]),
            "input",
        )
        .await
        .unwrap();

    match result {
        OrchestratorOutput::Parallel(par) => {
            assert_eq!(par.successes.len(), 2, "two agents should succeed");
            assert_eq!(par.failures.len(), 1, "one agent should fail");

            let failed_name = &par.failures[0].0;
            assert_eq!(failed_name, "fail");

            let mut success_names: Vec<&str> =
                par.successes.iter().map(|(n, _)| n.as_str()).collect();
            success_names.sort();
            assert_eq!(success_names, vec!["ok-1", "ok-2"]);
        }
        OrchestratorOutput::Single(_) => panic!("expected Parallel output"),
    }
}

// ── Timeout ────────────────────────────────────────────────────────

#[tokio::test]
async fn orchestrator_timeout_returns_error() {
    // Agent takes 2 seconds but orchestrator timeout is 100ms.
    let agents: HashMap<String, Arc<Agent>> = HashMap::from([(
        "slow".into(),
        build_slow_agent(Duration::from_secs(2), "too late"),
    )]);

    let orch = Orchestrator::new(agents, Duration::from_millis(100));
    let result = orch
        .execute(
            OrchestratorPattern::Sequential(vec!["slow".into()]),
            "hurry",
        )
        .await;

    match result {
        Err(KovaError::Orchestration(msg)) => {
            assert!(
                msg.contains("timed out"),
                "error should mention timeout, got: {msg}"
            );
        }
        Err(other) => panic!("expected Orchestration timeout error, got: {other:?}"),
        Ok(_) => panic!("expected timeout error"),
    }
}
