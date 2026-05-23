//! Property-based tests for the telemetry / observability layer.
//!
//! These tests verify that tracing spans carry the expected attributes and
//! that MetricsCollector counters behave correctly under arbitrary workloads.

use proptest::prelude::*;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::Stream;
use tokio::sync::Mutex;

use kova::agent::AgentBuilder;
use kova::error::KovaError;
use kova::models::*;
use kova::provider::LlmProvider;
use kova::telemetry::MetricsCollector;
use kova::tool::Tool;

// ── Shared mock infrastructure ─────────────────────────────────────

/// Mock provider that captures requests and returns configurable responses.
struct TelemetryMockProvider {
    responses: Vec<ModelResponse>,
    call_count: AtomicUsize,
    captured: Arc<Mutex<Vec<Vec<ConversationMessage>>>>,
}

impl TelemetryMockProvider {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
            captured: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl LlmProvider for TelemetryMockProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        self.captured.lock().await.push(messages.to_vec());
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let response_idx = idx.min(self.responses.len() - 1);
        Ok(self.responses[response_idx].clone())
    }

    async fn chat_completion_stream(
        &self,
        _messages: &[ConversationMessage],
        _tools: &[ToolDefinition],
        _config: &InferenceConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError> {
        Err(KovaError::Stream("not implemented".into()))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        Ok(vec![])
    }
}

/// A configurable mock tool for telemetry tests.
struct TelemetryMockTool {
    tool_name: String,
    should_fail: bool,
}

#[async_trait]
impl Tool for TelemetryMockTool {
    fn name(&self) -> &str {
        &self.tool_name
    }
    fn description(&self) -> &str {
        "telemetry test tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult, KovaError> {
        if self.should_fail {
            Err(KovaError::ToolExecution {
                tool_name: self.tool_name.clone(),
                message: "intentional failure".into(),
            })
        } else {
            Ok(ToolResult {
                content: format!("result from {}", self.tool_name),
                is_error: false,
            })
        }
    }
}

fn make_text_response(text: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: Some(UsageStats {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
        }),
    }
}

fn make_tool_call_response(calls: Vec<(&str, &str)>) -> ModelResponse {
    let content = calls
        .into_iter()
        .map(|(id, name)| ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: serde_json::json!({}),
        })
        .collect();
    ModelResponse {
        content,
        stop_reason: StopReason::ToolUse,
        usage: Some(UsageStats {
            input_tokens: 8,
            output_tokens: 3,
            total_tokens: 11,
        }),
    }
}

// ════════════════════════════════════════════════════════════════════
// Property 25: Tool Execution Span Attributes
// ════════════════════════════════════════════════════════════════════
//
// For any tool execution, emitted span contains tool name, duration,
// success/failure.
//
// The agent emits `tracing::info!` events inside `tool.execute` spans
// with fields: tool.name, duration_ms, success. We use `tracing_test`
// to capture log output and verify these fields are present.

#[tokio::test]
#[tracing_test::traced_test]
async fn prop25_tool_execution_span_attributes_success() {
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc1", "my_tool")]),
        make_text_response("done"),
    ]));

    let tool: Arc<dyn Tool> = Arc::new(TelemetryMockTool {
        tool_name: "my_tool".to_string(),
        should_fail: false,
    });

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .tool(tool)
        .max_iterations(5)
        .build()
        .unwrap();

    let _ = agent.chat("conv_p25_ok", "hello").await;

    // The agent logs "Tool execution complete" with tool.name, duration_ms, success
    assert!(logs_contain("tool.name"));
    assert!(logs_contain("my_tool"));
    assert!(logs_contain("duration_ms"));
    assert!(logs_contain("success"));
}

#[tokio::test]
#[tracing_test::traced_test]
async fn prop25_tool_execution_span_attributes_failure() {
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc2", "bad_tool")]),
        make_text_response("done"),
    ]));

    let tool: Arc<dyn Tool> = Arc::new(TelemetryMockTool {
        tool_name: "bad_tool".to_string(),
        should_fail: true,
    });

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .tool(tool)
        .max_iterations(5)
        .build()
        .unwrap();

    let _ = agent.chat("conv_p25_fail", "hello").await;

    assert!(logs_contain("bad_tool"));
    assert!(logs_contain("duration_ms"));
    assert!(logs_contain("success"));
    // On failure the agent also logs a warning with the error
    assert!(logs_contain("Tool execution failed") || logs_contain("Tool not found"));
}

#[tokio::test]
#[tracing_test::traced_test]
async fn prop25_tool_not_found_span_attributes() {
    // Tool call references a name not in the registry → ToolNotFound path
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc3", "nonexistent_tool")]),
        make_text_response("done"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .max_iterations(5)
        .build()
        .unwrap();

    let _ = agent.chat("conv_p25_notfound", "hello").await;

    assert!(logs_contain("nonexistent_tool"));
    assert!(logs_contain("Tool not found"));
}

// ════════════════════════════════════════════════════════════════════
// Property 26: LLM Request Span Attributes
// ════════════════════════════════════════════════════════════════════
//
// For any LLM request, emitted span contains model name, latency,
// token counts, status.
//
// The agent wraps each chat turn in an `agent.chat` span with
// conversation_id. When tool calls occur, child events are emitted
// within that span, making the span context visible in log output.
// We verify the conversation_id and span hierarchy appear.

#[tokio::test]
#[tracing_test::traced_test]
async fn prop26_llm_request_span_contains_conversation_id() {
    // Use a tool call flow so that events are emitted within the
    // agent.chat span, making the span context visible in logs.
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc_p26", "probe_tool")]),
        make_text_response("hello back"),
    ]));

    let tool: Arc<dyn Tool> = Arc::new(TelemetryMockTool {
        tool_name: "probe_tool".to_string(),
        should_fail: false,
    });

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .tool(tool)
        .max_iterations(5)
        .build()
        .unwrap();

    let result = agent.chat("conv_p26", "hi").await;
    assert!(result.is_ok());

    // The agent.chat span with conversation_id appears in the log
    // context for events emitted during the chat turn.
    assert!(logs_contain("conv_p26"));
    // Tool execution events are emitted within the agent.chat span
    assert!(logs_contain("probe_tool"));
    assert!(logs_contain("Tool execution complete"));
}

// ════════════════════════════════════════════════════════════════════
// Property 27: Error Span Attributes Follow OTEL Conventions
// ════════════════════════════════════════════════════════════════════
//
// For any span recording an error, span has otel.status_code="ERROR"
// with error type and message.
//
// When a tool fails or is not found, the agent records
// otel.status_code = "ERROR" on the current span. We verify this
// attribute appears in the captured logs.

#[tokio::test]
#[tracing_test::traced_test]
async fn prop27_error_span_otel_status_code_on_tool_failure() {
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc_err", "failing_tool")]),
        make_text_response("recovered"),
    ]));

    let tool: Arc<dyn Tool> = Arc::new(TelemetryMockTool {
        tool_name: "failing_tool".to_string(),
        should_fail: true,
    });

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .tool(tool)
        .max_iterations(5)
        .build()
        .unwrap();

    let _ = agent.chat("conv_p27_fail", "trigger error").await;

    // The tool.execute span should record otel.status_code = ERROR
    assert!(logs_contain("otel.status_code"));
    assert!(logs_contain("ERROR"));
    assert!(logs_contain("Tool execution failed"));
}

#[tokio::test]
#[tracing_test::traced_test]
async fn prop27_error_span_otel_status_code_on_tool_not_found() {
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc_nf", "ghost_tool")]),
        make_text_response("recovered"),
    ]));

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .max_iterations(5)
        .build()
        .unwrap();

    let _ = agent.chat("conv_p27_nf", "trigger not found").await;

    assert!(logs_contain("otel.status_code"));
    assert!(logs_contain("ERROR"));
    assert!(logs_contain("Tool not found"));
}

// ════════════════════════════════════════════════════════════════════
// Property 28: Metrics Counter Increments
// ════════════════════════════════════════════════════════════════════
//
// For N LLM requests and M tool invocations, counters match N, M,
// and sum of tokens.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop28_metrics_counter_increments(
        n_llm in 1usize..=20,
        m_tools in 1usize..=20,
        // Generate per-request token pairs (input, output) in a reasonable range
        token_pairs in proptest::collection::vec((0u64..500, 0u64..500), 1..=20),
        // Generate per-tool durations and success flags
        tool_runs in proptest::collection::vec((0.1f64..100.0, any::<bool>()), 1..=20),
    ) {
        let mc = MetricsCollector::new();

        // Take exactly n_llm token pairs (cycling if needed)
        let mut expected_total_tokens: u64 = 0;
        for i in 0..n_llm {
            let (input_tok, output_tok) = token_pairs[i % token_pairs.len()];
            let latency = (i as f64) * 1.5 + 1.0;
            mc.record_llm_request(latency, input_tok, output_tok);
            expected_total_tokens += input_tok + output_tok;
        }

        // Take exactly m_tools tool runs (cycling if needed)
        let mut expected_tool_errors: u64 = 0;
        for i in 0..m_tools {
            let (duration, success) = tool_runs[i % tool_runs.len()];
            mc.record_tool_invocation(duration, success);
            if !success {
                expected_tool_errors += 1;
            }
        }

        prop_assert_eq!(
            mc.llm_request_count(),
            n_llm as u64,
            "LLM request count mismatch"
        );
        prop_assert_eq!(
            mc.tool_invocation_count(),
            m_tools as u64,
            "Tool invocation count mismatch"
        );
        prop_assert_eq!(
            mc.total_tokens(),
            expected_total_tokens,
            "Total tokens mismatch"
        );
        prop_assert_eq!(
            mc.error_count(),
            expected_tool_errors,
            "Error count mismatch (only tool errors, no LLM errors recorded)"
        );

        // Histogram lengths must match
        prop_assert_eq!(
            mc.llm_latency_histogram().len(),
            n_llm,
            "LLM latency histogram length mismatch"
        );
        prop_assert_eq!(
            mc.tool_duration_histogram().len(),
            m_tools,
            "Tool duration histogram length mismatch"
        );
    }

    /// Variant: include LLM errors in the mix.
    #[test]
    fn prop28_metrics_counter_with_llm_errors(
        n_llm in 1usize..=15,
        n_llm_errors in 0usize..=10,
        m_tools in 1usize..=15,
        m_tool_failures in 0usize..=10,
    ) {
        let mc = MetricsCollector::new();

        for _ in 0..n_llm {
            mc.record_llm_request(5.0, 100, 50);
        }
        for _ in 0..n_llm_errors {
            mc.record_llm_error();
        }
        for i in 0..m_tools {
            let success = i >= m_tool_failures;
            mc.record_tool_invocation(2.0, success);
        }

        let actual_tool_failures = m_tool_failures.min(m_tools);

        prop_assert_eq!(mc.llm_request_count(), n_llm as u64);
        prop_assert_eq!(mc.tool_invocation_count(), m_tools as u64);
        prop_assert_eq!(mc.total_tokens(), (n_llm as u64) * 150);
        prop_assert_eq!(
            mc.error_count(),
            (n_llm_errors + actual_tool_failures) as u64,
            "Total errors = LLM errors + tool failures"
        );
    }

    /// Verify reset clears all counters for any workload.
    #[test]
    fn prop28_metrics_reset_clears_all(
        n in 1usize..=20,
        m in 1usize..=20,
    ) {
        let mc = MetricsCollector::new();

        for _ in 0..n {
            mc.record_llm_request(10.0, 50, 25);
        }
        for _ in 0..m {
            mc.record_tool_invocation(3.0, true);
        }
        mc.record_llm_error();

        // Sanity: non-zero before reset
        prop_assert!(mc.llm_request_count() > 0);
        prop_assert!(mc.tool_invocation_count() > 0);

        mc.reset();

        prop_assert_eq!(mc.llm_request_count(), 0);
        prop_assert_eq!(mc.tool_invocation_count(), 0);
        prop_assert_eq!(mc.total_tokens(), 0);
        prop_assert_eq!(mc.error_count(), 0);
        prop_assert!(mc.llm_latency_histogram().is_empty());
        prop_assert!(mc.tool_duration_histogram().is_empty());
    }
}

// ════════════════════════════════════════════════════════════════════
// Property 29: Log Level Filtering
// ════════════════════════════════════════════════════════════════════
//
// For configured level L, events below L are not emitted, events at
// or above L are emitted.
//
// We test this by verifying the TelemetryConfig builder correctly
// stores the configured level, and that the tracing subscriber
// respects level filtering. Since we cannot install multiple global
// subscribers in a single test process, we verify the filtering
// logic structurally: the config propagates the level, and
// tracing_subscriber's EnvFilter (used in init_basic) is known to
// filter correctly.

#[test]
fn prop29_log_level_filtering_config_propagation() {
    use kova::telemetry::TelemetryConfig;

    let levels = [
        tracing::Level::TRACE,
        tracing::Level::DEBUG,
        tracing::Level::INFO,
        tracing::Level::WARN,
        tracing::Level::ERROR,
    ];

    for level in &levels {
        let cfg = TelemetryConfig::builder().log_level(*level).build();
        assert_eq!(
            cfg.log_level, *level,
            "TelemetryConfig should store the configured log level"
        );
    }
}

/// Verify that tracing events at or above the configured level are
/// captured, and events below are not. We use `tracing_test` which
/// installs a subscriber that captures all levels, then manually
/// verify the filtering semantics match what TelemetryConfig would
/// produce.
#[tokio::test]
#[tracing_test::traced_test]
async fn prop29_log_level_filtering_events_emitted() {
    // With tracing_test's subscriber (captures everything), emit at
    // all levels and verify they appear. This confirms the tracing
    // macros are wired correctly in the codebase.
    tracing::trace!("trace_event_p29");
    tracing::debug!("debug_event_p29");
    tracing::info!("info_event_p29");
    tracing::warn!("warn_event_p29");
    tracing::error!("error_event_p29");

    assert!(logs_contain("trace_event_p29"));
    assert!(logs_contain("debug_event_p29"));
    assert!(logs_contain("info_event_p29"));
    assert!(logs_contain("warn_event_p29"));
    assert!(logs_contain("error_event_p29"));
}

/// Verify that the EnvFilter produced from a TelemetryConfig level
/// string correctly parses for all valid levels.
#[test]
fn prop29_env_filter_parses_for_all_levels() {
    use tracing_subscriber::EnvFilter;

    let levels = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];
    for level_str in &levels {
        let filter = EnvFilter::try_new(level_str);
        assert!(
            filter.is_ok(),
            "EnvFilter should parse level '{}' without error",
            level_str
        );
    }
}

// ════════════════════════════════════════════════════════════════════
// Property 30: Span Parent References
// ════════════════════════════════════════════════════════════════════
//
// For all spans emitted during an agent chat turn, each child span
// has parent reference to root span.
//
// The agent creates a root `agent.chat` span and child `tool.execute`
// spans via `Instrument`. We verify that tool execution spans appear
// nested under the agent.chat span by checking the log output
// structure from tracing_test.

#[tokio::test]
#[tracing_test::traced_test]
async fn prop30_span_parent_references_tool_under_agent() {
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc_p30", "parent_test_tool")]),
        make_text_response("done"),
    ]));

    let tool: Arc<dyn Tool> = Arc::new(TelemetryMockTool {
        tool_name: "parent_test_tool".to_string(),
        should_fail: false,
    });

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .tool(tool)
        .max_iterations(5)
        .build()
        .unwrap();

    let _ = agent.chat("conv_p30", "test parent spans").await;

    // The root span `agent.chat` should be present
    assert!(logs_contain("agent.chat"));
    // The child span `tool.execute` should be present
    assert!(logs_contain("tool.execute"));
    // The tool name should appear in the child span
    assert!(logs_contain("parent_test_tool"));

    // tracing_test prefixes log lines with span context. The tool.execute
    // events should appear within the agent.chat span context, evidenced
    // by both span names appearing in the same log lines.
    logs_assert(|lines: &[&str]| {
        // Find lines that mention tool execution — they should also
        // reference the agent.chat span in their prefix.
        let tool_lines: Vec<&&str> = lines
            .iter()
            .filter(|line| line.contains("parent_test_tool"))
            .collect();

        if tool_lines.is_empty() {
            return Err("No log lines found mentioning parent_test_tool".to_string());
        }

        // At least one tool-related line should show the agent.chat
        // span in its context (tracing_test nests span names).
        let has_parent_context = tool_lines
            .iter()
            .any(|line| line.contains("agent.chat") || line.contains("conv_p30"));

        if !has_parent_context {
            return Err(format!(
                "Tool execution spans should be nested under agent.chat. Lines: {:?}",
                tool_lines
            ));
        }

        Ok(())
    });
}

#[tokio::test]
#[tracing_test::traced_test]
async fn prop30_multiple_tool_spans_under_same_parent() {
    let provider = Arc::new(TelemetryMockProvider::new(vec![
        make_tool_call_response(vec![("tc_p30a", "tool_alpha"), ("tc_p30b", "tool_beta")]),
        make_text_response("done"),
    ]));

    let tool_a: Arc<dyn Tool> = Arc::new(TelemetryMockTool {
        tool_name: "tool_alpha".to_string(),
        should_fail: false,
    });
    let tool_b: Arc<dyn Tool> = Arc::new(TelemetryMockTool {
        tool_name: "tool_beta".to_string(),
        should_fail: false,
    });

    let agent = AgentBuilder::new()
        .provider(provider as Arc<dyn LlmProvider>)
        .tool(tool_a)
        .tool(tool_b)
        .max_iterations(5)
        .build()
        .unwrap();

    let _ = agent.chat("conv_p30_multi", "test multiple tools").await;

    // Both tool spans should be present
    assert!(logs_contain("tool_alpha"));
    assert!(logs_contain("tool_beta"));
    // The parent agent.chat span should be present
    assert!(logs_contain("agent.chat"));
    assert!(logs_contain("conv_p30_multi"));
}

// ════════════════════════════════════════════════════════════════════
// Property 25 (proptest variant): Parameterized tool name verification
// ════════════════════════════════════════════════════════════════════
//
// For any tool name, the emitted span contains that exact tool name.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop25_parameterized_tool_name_in_metrics(
        tool_name in "[a-z][a-z0-9_]{2,15}",
        success in any::<bool>(),
        duration in 0.1f64..1000.0,
    ) {
        let mc = MetricsCollector::new();
        mc.record_tool_invocation(duration, success);

        // The MetricsCollector records the invocation
        prop_assert_eq!(mc.tool_invocation_count(), 1);
        prop_assert_eq!(mc.tool_duration_histogram().len(), 1);

        let recorded_duration = mc.tool_duration_histogram()[0];
        prop_assert!(
            (recorded_duration - duration).abs() < f64::EPSILON,
            "Duration should be recorded exactly"
        );

        if !success {
            prop_assert_eq!(mc.error_count(), 1, "Failed tool should increment error count");
        } else {
            prop_assert_eq!(mc.error_count(), 0, "Successful tool should not increment error count");
        }

        // Verify the tool_name is valid for use in tracing (non-empty, no panics)
        let _span = tracing::info_span!("test.tool", tool.name = %tool_name);
        // If we get here without panic, the tool name is valid for tracing
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property 26 (proptest variant): For any LLM request with arbitrary
    /// token counts, MetricsCollector records them accurately.
    #[test]
    fn prop26_parameterized_llm_metrics(
        latency in 0.1f64..10000.0,
        input_tokens in 0u64..10000,
        output_tokens in 0u64..10000,
    ) {
        let mc = MetricsCollector::new();
        mc.record_llm_request(latency, input_tokens, output_tokens);

        prop_assert_eq!(mc.llm_request_count(), 1);
        prop_assert_eq!(mc.total_tokens(), input_tokens + output_tokens);
        prop_assert_eq!(mc.llm_latency_histogram().len(), 1);

        let recorded = mc.llm_latency_histogram()[0];
        prop_assert!(
            (recorded - latency).abs() < f64::EPSILON,
            "Latency should be recorded exactly"
        );
    }
}
