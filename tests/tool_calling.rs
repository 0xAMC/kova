mod mock;

use std::sync::Arc;

use kova_sdk::agent::AgentBuilder;
use kova_sdk::error::KovaError;
use kova_sdk::models::*;

use mock::provider::{
    CapturingMockProvider, MockLlmProvider, make_text_response, make_tool_call_response, run_text,
};
use mock::tool::MockTool;

// ── Helpers ────────────────────────────────────────────────────────

fn tool_call(id: &str, name: &str, args: &str) -> (String, String, serde_json::Value) {
    let input: serde_json::Value = serde_json::from_str(args).unwrap_or(serde_json::json!({}));
    (id.to_string(), name.to_string(), input)
}

// ── Single tool call flow ──────────────────────────────────────────

#[tokio::test]
async fn single_tool_call_executes_and_returns_final_text() {
    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_tool_call_response(vec![tool_call("tc-1", "greet", "{}")]),
        make_text_response("Hello, world!"),
    ]));

    let tool = Arc::new(MockTool::new("greet", "greeting result"));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .tool(tool.clone())
        .build()
        .unwrap();

    let reply = run_text(&agent, "say hi").await.unwrap();

    assert_eq!(reply, "Hello, world!");
    assert_eq!(tool.call_count(), 1);
    assert_eq!(provider.call_count(), 2);
}

// ── Multi-turn tool calls ──────────────────────────────────────────

#[tokio::test]
async fn multi_turn_tool_calls() {
    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_tool_call_response(vec![tool_call("tc-1", "step_one", "{}")]),
        make_tool_call_response(vec![tool_call("tc-2", "step_two", "{}")]),
        make_text_response("all done"),
    ]));

    let tool_a = Arc::new(MockTool::new("step_one", "result A"));
    let tool_b = Arc::new(MockTool::new("step_two", "result B"));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .tool(tool_a.clone())
        .tool(tool_b.clone())
        .build()
        .unwrap();

    let reply = run_text(&agent, "do the thing").await.unwrap();

    assert_eq!(reply, "all done");
    assert_eq!(tool_a.call_count(), 1);
    assert_eq!(tool_b.call_count(), 1);
    assert_eq!(provider.call_count(), 3);
}

// ── Tool not found ─────────────────────────────────────────────────

#[tokio::test]
async fn tool_not_found_sends_error_to_llm() {
    let provider = Arc::new(CapturingMockProvider::new(vec![
        make_tool_call_response(vec![tool_call("tc-1", "nonexistent", "{}")]),
        make_text_response("I see the tool was not found"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .unwrap();

    let reply = run_text(&agent, "use a tool").await.unwrap();
    assert_eq!(reply, "I see the tool was not found");

    let requests = provider.captured_requests().await;
    assert_eq!(requests.len(), 2);

    let second_msgs = &requests[1];
    let tool_msg = second_msgs
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("should have a tool-role message");

    // Extract tool_use_id and content from the ToolResult content block.
    let (tool_use_id, content) = tool_msg
        .content
        .iter()
        .find_map(|b| {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } = b
            {
                Some((tool_use_id.as_str(), content.as_str()))
            } else {
                None
            }
        })
        .expect("should have a ToolResult content block");

    assert_eq!(tool_use_id, "tc-1");
    assert!(
        content.contains("not found") || content.contains("nonexistent"),
        "tool error message should mention the missing tool, got: {content}"
    );
}

// ── Tool execution error ───────────────────────────────────────────

#[tokio::test]
async fn tool_execution_error_forwarded_to_llm() {
    let provider = Arc::new(CapturingMockProvider::new(vec![
        make_tool_call_response(vec![tool_call("tc-1", "broken", "{}")]),
        make_text_response("I handled the error"),
    ]));

    let tool = Arc::new(MockTool::failing("broken", "something went wrong"));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .tool(tool.clone())
        .build()
        .unwrap();

    let reply = run_text(&agent, "call the tool").await.unwrap();
    assert_eq!(reply, "I handled the error");
    assert_eq!(tool.call_count(), 1);

    let requests = provider.captured_requests().await;
    let second_msgs = &requests[1];
    let tool_msg = second_msgs
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("should have a tool-role message");

    let (tool_use_id, content) = tool_msg
        .content
        .iter()
        .find_map(|b| {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } = b
            {
                Some((tool_use_id.as_str(), content.as_str()))
            } else {
                None
            }
        })
        .expect("should have a ToolResult content block");

    assert_eq!(tool_use_id, "tc-1");
    assert!(
        content.contains("something went wrong"),
        "error content should contain the failure message, got: {content}"
    );
}

// ── Max iterations ─────────────────────────────────────────────────

#[tokio::test]
async fn max_iterations_terminates_tool_loop() {
    let always_tool_call = make_tool_call_response(vec![tool_call("tc-loop", "echo", "{}")]);
    let provider = Arc::new(MockLlmProvider::with_response(always_tool_call));

    let tool = Arc::new(MockTool::new("echo", "echoed"));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .tool(tool.clone())
        .max_iterations(3)
        .build()
        .unwrap();

    let result = run_text(&agent, "loop forever").await;

    match result {
        Err(KovaError::MaxIterations(n)) => assert_eq!(n, 3),
        Err(other) => panic!("expected MaxIterations, got: {other:?}"),
        Ok(text) => panic!("expected error, got text: {text}"),
    }

    assert_eq!(tool.call_count(), 3);
    assert_eq!(provider.call_count(), 4);
}

// ── Multiple tool calls in a single response ───────────────────────

#[tokio::test]
async fn multiple_tool_calls_in_single_response() {
    let provider = Arc::new(CapturingMockProvider::new(vec![
        make_tool_call_response(vec![
            tool_call("tc-a", "alpha", "{}"),
            tool_call("tc-b", "beta", "{}"),
        ]),
        make_text_response("both tools done"),
    ]));

    let tool_a = Arc::new(MockTool::new("alpha", "alpha result"));
    let tool_b = Arc::new(MockTool::new("beta", "beta result"));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .tool(tool_a.clone())
        .tool(tool_b.clone())
        .build()
        .unwrap();

    let reply = run_text(&agent, "use both").await.unwrap();
    assert_eq!(reply, "both tools done");
    assert_eq!(tool_a.call_count(), 1);
    assert_eq!(tool_b.call_count(), 1);

    let requests = provider.captured_requests().await;
    let second_msgs = &requests[1];
    let tool_msgs: Vec<_> = second_msgs
        .iter()
        .filter(|m| m.role == Role::Tool)
        .collect();

    assert_eq!(tool_msgs.len(), 2, "should have 2 tool-role messages");

    let ids: Vec<_> = tool_msgs
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|b| {
                if let ContentBlock::ToolResult { tool_use_id, .. } = b {
                    Some(tool_use_id.as_str())
                } else {
                    None
                }
            })
        })
        .collect();
    assert!(ids.contains(&"tc-a"));
    assert!(ids.contains(&"tc-b"));
}

// ── Tool call with no registered tools ─────────────────────────────

#[tokio::test]
async fn tool_call_with_empty_registry_sends_not_found() {
    let provider = Arc::new(CapturingMockProvider::new(vec![
        make_tool_call_response(vec![tool_call("tc-1", "missing", "{}")]),
        make_text_response("ok, no tools"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .build()
        .unwrap();

    let reply = run_text(&agent, "try a tool").await.unwrap();
    assert_eq!(reply, "ok, no tools");

    let requests = provider.captured_requests().await;
    let second_msgs = &requests[1];
    let tool_msg = second_msgs
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("should have tool-role error message");

    let (tool_use_id, content) = tool_msg
        .content
        .iter()
        .find_map(|b| {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } = b
            {
                Some((tool_use_id.as_str(), content.as_str()))
            } else {
                None
            }
        })
        .expect("should have a ToolResult content block");

    assert_eq!(tool_use_id, "tc-1");
    assert!(content.contains("not found"));
}

// ── Approval session caching ───────────────────────────────────────

mod approval_caching {
    use super::*;
    use async_trait::async_trait;
    use kova_sdk::{ApprovalDecision, ToolApprovalHandler};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingApprovalHandler {
        decision: ApprovalDecision,
        calls: AtomicUsize,
    }

    impl CountingApprovalHandler {
        fn new(decision: ApprovalDecision) -> Self {
            Self {
                decision,
                calls: AtomicUsize::new(0),
            }
        }
        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ToolApprovalHandler for CountingApprovalHandler {
        async fn approve(&self, _tool_name: &str, _args: &serde_json::Value) -> ApprovalDecision {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.decision.clone()
        }
    }

    fn two_round_agent(
        provider: Arc<MockLlmProvider>,
        tool: Arc<MockTool>,
        handler: Arc<CountingApprovalHandler>,
    ) -> kova_sdk::agent::Agent {
        AgentBuilder::new()
            .provider(provider)
            .tool(tool)
            .with_approval_handler(handler)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn approved_for_session_asks_handler_only_once() {
        let provider = Arc::new(MockLlmProvider::with_responses(vec![
            make_tool_call_response(vec![tool_call("tc-1", "greet", "{}")]),
            make_tool_call_response(vec![tool_call("tc-2", "greet", "{}")]),
            make_text_response("done"),
        ]));
        let tool = Arc::new(MockTool::new("greet", "hi"));
        let handler = Arc::new(CountingApprovalHandler::new(
            ApprovalDecision::ApprovedForSession,
        ));

        let agent = two_round_agent(provider, tool.clone(), handler.clone());
        let reply = run_text(&agent, "go").await.unwrap();

        assert_eq!(reply, "done");
        assert_eq!(tool.call_count(), 2, "both invocations execute");
        assert_eq!(handler.call_count(), 1, "handler consulted only once");
    }

    #[tokio::test]
    async fn denied_always_blocks_without_reasking() {
        let provider = Arc::new(MockLlmProvider::with_responses(vec![
            make_tool_call_response(vec![tool_call("tc-1", "greet", "{}")]),
            make_tool_call_response(vec![tool_call("tc-2", "greet", "{}")]),
            make_text_response("done"),
        ]));
        let tool = Arc::new(MockTool::new("greet", "hi"));
        let handler = Arc::new(CountingApprovalHandler::new(ApprovalDecision::DeniedAlways));

        let agent = two_round_agent(provider, tool.clone(), handler.clone());
        let reply = run_text(&agent, "go").await.unwrap();

        assert_eq!(reply, "done");
        assert_eq!(tool.call_count(), 0, "denied tool never executes");
        assert_eq!(handler.call_count(), 1, "handler consulted only once");
    }

    #[tokio::test]
    async fn plain_approved_asks_handler_every_time() {
        let provider = Arc::new(MockLlmProvider::with_responses(vec![
            make_tool_call_response(vec![tool_call("tc-1", "greet", "{}")]),
            make_tool_call_response(vec![tool_call("tc-2", "greet", "{}")]),
            make_text_response("done"),
        ]));
        let tool = Arc::new(MockTool::new("greet", "hi"));
        let handler = Arc::new(CountingApprovalHandler::new(ApprovalDecision::Approved));

        let agent = two_round_agent(provider, tool.clone(), handler.clone());
        run_text(&agent, "go").await.unwrap();

        assert_eq!(tool.call_count(), 2);
        assert_eq!(handler.call_count(), 2);
    }
}

// ── Metrics wiring ─────────────────────────────────────────────────

#[tokio::test]
async fn agent_records_metrics_when_collector_registered() {
    use kova_sdk::telemetry::MetricsCollector;

    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_tool_call_response(vec![tool_call("tc-1", "greet", "{}")]),
        make_text_response("done"),
    ]));
    let tool = Arc::new(MockTool::new("greet", "hi"));
    let metrics = Arc::new(MetricsCollector::new());

    let agent = AgentBuilder::new()
        .provider(provider)
        .tool(tool)
        .metrics(Arc::clone(&metrics))
        .build()
        .unwrap();

    run_text(&agent, "go").await.unwrap();

    assert_eq!(
        metrics.llm_request_count(),
        2,
        "two provider calls recorded"
    );
    assert_eq!(metrics.tool_invocation_count(), 1, "one tool call recorded");
    assert_eq!(metrics.error_count(), 0);
    assert_eq!(metrics.llm_latency_histogram().len(), 2);
    assert_eq!(metrics.tool_duration_histogram().len(), 1);
}
