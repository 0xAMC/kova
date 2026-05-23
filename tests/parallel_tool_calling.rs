mod mock;

use std::sync::Arc;
use std::time::{Duration, Instant};

use kova_sdk::agent::AgentBuilder;
use kova_sdk::models::*;

use mock::provider::{
    CapturingMockProvider, MockLlmProvider, make_text_response, make_tool_call_response,
};
use mock::tool::MockTool;

// ── Helpers ────────────────────────────────────────────────────────

fn tool_call(id: &str, name: &str, args: &str) -> (String, String, serde_json::Value) {
    let input: serde_json::Value = serde_json::from_str(args).unwrap_or(serde_json::json!({}));
    (id.to_string(), name.to_string(), input)
}

// ── Test 1: Concurrent execution verified via timing ───────────────

#[tokio::test]
async fn parallel_tool_calls_execute_concurrently() {
    let delay = Duration::from_millis(100);

    // 5 tools, each with a 100ms delay.
    let tools: Vec<Arc<dyn kova_sdk::tool::Tool>> = (0..5)
        .map(|i| {
            Arc::new(MockTool::with_delay(
                &format!("tool_{i}"),
                &format!("result_{i}"),
                delay,
            )) as Arc<dyn kova_sdk::tool::Tool>
        })
        .collect();

    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_tool_call_response(vec![
            tool_call("tc-0", "tool_0", "{}"),
            tool_call("tc-1", "tool_1", "{}"),
            tool_call("tc-2", "tool_2", "{}"),
            tool_call("tc-3", "tool_3", "{}"),
            tool_call("tc-4", "tool_4", "{}"),
        ]),
        make_text_response("all five done"),
    ]));

    let mut builder = AgentBuilder::new().provider(provider.clone());
    for t in &tools {
        builder = builder.tool(Arc::clone(t));
    }
    let agent = builder.build().unwrap();

    let start = Instant::now();
    let reply = agent.chat("conv-1", "run all tools").await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(reply, "all five done");

    // If executed sequentially, total time would be >= 500ms.
    // Concurrent execution should complete in well under 300ms.
    assert!(
        elapsed < Duration::from_millis(300),
        "Expected concurrent execution to finish in <300ms, took {:?}",
        elapsed
    );
}

// ── Test 2: Partial failure — some tools fail, others succeed ──────

#[tokio::test]
async fn parallel_tool_calls_partial_failure() {
    let tool_ok_1 = Arc::new(MockTool::new("ok_one", "success_1"));
    let tool_ok_2 = Arc::new(MockTool::new("ok_two", "success_2"));
    let tool_fail = Arc::new(MockTool::failing("bad_tool", "boom"));

    let provider = Arc::new(CapturingMockProvider::new(vec![
        make_tool_call_response(vec![
            tool_call("tc-ok1", "ok_one", "{}"),
            tool_call("tc-fail", "bad_tool", "{}"),
            tool_call("tc-ok2", "ok_two", "{}"),
        ]),
        make_text_response("handled partial failure"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .tool(tool_ok_1.clone())
        .tool(tool_ok_2.clone())
        .tool(tool_fail.clone())
        .build()
        .unwrap();

    let reply = agent.chat("conv-1", "run mixed tools").await.unwrap();
    assert_eq!(reply, "handled partial failure");

    // Inspect the second LLM request — it should contain exactly 3 tool-role messages.
    let requests = provider.captured_requests().await;
    assert_eq!(requests.len(), 2, "expected 2 LLM calls");

    let second_msgs = &requests[1];
    let tool_msgs: Vec<_> = second_msgs
        .iter()
        .filter(|m| m.role == Role::Tool)
        .collect();
    assert_eq!(tool_msgs.len(), 3, "should have 3 tool-role messages");

    // Collect all ToolResult blocks from tool messages.
    let tool_results: Vec<(&str, &str, bool)> = tool_msgs
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|b| {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } = b
                {
                    Some((tool_use_id.as_str(), content.as_str(), *is_error))
                } else {
                    None
                }
            })
        })
        .collect();

    assert_eq!(tool_results.len(), 3);

    // The failed tool should have is_error: true.
    let failed = tool_results
        .iter()
        .find(|(id, _, _)| *id == "tc-fail")
        .expect("should have result for tc-fail");
    assert!(failed.2, "failed tool result should have is_error=true");

    // The successful tools should have is_error: false.
    let ok1 = tool_results
        .iter()
        .find(|(id, _, _)| *id == "tc-ok1")
        .expect("should have result for tc-ok1");
    assert!(!ok1.2, "successful tool result should have is_error=false");

    let ok2 = tool_results
        .iter()
        .find(|(id, _, _)| *id == "tc-ok2")
        .expect("should have result for tc-ok2");
    assert!(!ok2.2, "successful tool result should have is_error=false");
}

// ── Test 3: Semaphore limits concurrency ───────────────────────────

#[tokio::test]
async fn parallel_tool_calls_semaphore_limits_concurrency() {
    let delay = Duration::from_millis(100);

    // 5 tools, each with a 100ms delay.
    let tools: Vec<Arc<dyn kova_sdk::tool::Tool>> = (0..5)
        .map(|i| {
            Arc::new(MockTool::with_delay(
                &format!("sem_tool_{i}"),
                &format!("result_{i}"),
                delay,
            )) as Arc<dyn kova_sdk::tool::Tool>
        })
        .collect();

    let provider = Arc::new(MockLlmProvider::with_responses(vec![
        make_tool_call_response(vec![
            tool_call("tc-0", "sem_tool_0", "{}"),
            tool_call("tc-1", "sem_tool_1", "{}"),
            tool_call("tc-2", "sem_tool_2", "{}"),
            tool_call("tc-3", "sem_tool_3", "{}"),
            tool_call("tc-4", "sem_tool_4", "{}"),
        ]),
        make_text_response("semaphore test done"),
    ]));

    let mut builder = AgentBuilder::new()
        .provider(provider.clone())
        .max_concurrent_tools(2);
    for t in &tools {
        builder = builder.tool(Arc::clone(t));
    }
    let agent = builder.build().unwrap();

    let start = Instant::now();
    let reply = agent.chat("conv-1", "run with semaphore").await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(reply, "semaphore test done");

    // With max_concurrent_tools=2 and 5 tools each taking 100ms:
    // Batch 1: tools 0,1 (100ms)
    // Batch 2: tools 2,3 (100ms)
    // Batch 3: tool 4    (100ms)
    // Total: ~300ms minimum
    // We assert >= 250ms to account for timing variance.
    assert!(
        elapsed >= Duration::from_millis(250),
        "Expected semaphore-limited execution to take >=250ms, took {:?}",
        elapsed
    );
}
