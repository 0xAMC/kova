//! Metrics collection for LLM requests, tool invocations, and token usage.
//!
//! `MetricsCollector` uses atomic counters and `RwLock`-protected histograms
//! so it works both with and without the `telemetry` feature flag.
//! It is `Send + Sync` by construction.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Collects quantitative measurements for observability.
///
/// Records histograms for LLM request latency and tool execution duration,
/// and counters for total LLM requests, tool invocations, tokens consumed,
/// and errors.
#[derive(Debug)]
pub struct MetricsCollector {
    total_llm_requests: AtomicU64,
    total_tool_invocations: AtomicU64,
    total_tokens_consumed: AtomicU64,
    total_errors: AtomicU64,
    llm_request_latency_ms: RwLock<Vec<f64>>,
    tool_execution_duration_ms: RwLock<Vec<f64>>,
}

impl MetricsCollector {
    /// Create a new, zeroed `MetricsCollector`.
    pub fn new() -> Self {
        Self {
            total_llm_requests: AtomicU64::new(0),
            total_tool_invocations: AtomicU64::new(0),
            total_tokens_consumed: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            llm_request_latency_ms: RwLock::new(Vec::new()),
            tool_execution_duration_ms: RwLock::new(Vec::new()),
        }
    }

    /// Record a successful LLM request.
    ///
    /// Increments the LLM request counter, adds `latency_ms` to the latency
    /// histogram, and accumulates `input_tokens + output_tokens` into the
    /// total token counter.
    pub fn record_llm_request(&self, latency_ms: f64, input_tokens: u64, output_tokens: u64) {
        self.total_llm_requests.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_consumed
            .fetch_add(input_tokens + output_tokens, Ordering::Relaxed);
        if let Ok(mut hist) = self.llm_request_latency_ms.write() {
            hist.push(latency_ms);
        }
    }

    /// Record an LLM error (increments the error counter).
    pub fn record_llm_error(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a tool invocation.
    ///
    /// Increments the tool invocation counter, adds `duration_ms` to the
    /// duration histogram, and increments the error counter when `success`
    /// is `false`.
    pub fn record_tool_invocation(&self, duration_ms: f64, success: bool) {
        self.total_tool_invocations.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }
        if let Ok(mut hist) = self.tool_execution_duration_ms.write() {
            hist.push(duration_ms);
        }
    }

    /// Total number of LLM requests recorded.
    pub fn llm_request_count(&self) -> u64 {
        self.total_llm_requests.load(Ordering::Relaxed)
    }

    /// Total number of tool invocations recorded.
    pub fn tool_invocation_count(&self) -> u64 {
        self.total_tool_invocations.load(Ordering::Relaxed)
    }

    /// Total tokens consumed (input + output) across all LLM requests.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens_consumed.load(Ordering::Relaxed)
    }

    /// Total error count (LLM errors + failed tool invocations).
    pub fn error_count(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
    }

    /// Clone of the recorded LLM request latencies (milliseconds).
    pub fn llm_latency_histogram(&self) -> Vec<f64> {
        self.llm_request_latency_ms
            .read()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    /// Clone of the recorded tool execution durations (milliseconds).
    pub fn tool_duration_histogram(&self) -> Vec<f64> {
        self.tool_execution_duration_ms
            .read()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    /// Reset all counters and histograms to zero / empty.
    pub fn reset(&self) {
        self.total_llm_requests.store(0, Ordering::Relaxed);
        self.total_tool_invocations.store(0, Ordering::Relaxed);
        self.total_tokens_consumed.store(0, Ordering::Relaxed);
        self.total_errors.store(0, Ordering::Relaxed);
        if let Ok(mut hist) = self.llm_request_latency_ms.write() {
            hist.clear();
        }
        if let Ok(mut hist) = self.tool_execution_duration_ms.write() {
            hist.clear();
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_collector_starts_at_zero() {
        let mc = MetricsCollector::new();
        assert_eq!(mc.llm_request_count(), 0);
        assert_eq!(mc.tool_invocation_count(), 0);
        assert_eq!(mc.total_tokens(), 0);
        assert_eq!(mc.error_count(), 0);
        assert!(mc.llm_latency_histogram().is_empty());
        assert!(mc.tool_duration_histogram().is_empty());
    }

    #[test]
    fn record_llm_request_increments_counters_and_histogram() {
        let mc = MetricsCollector::new();
        mc.record_llm_request(42.5, 100, 50);
        mc.record_llm_request(10.0, 200, 80);

        assert_eq!(mc.llm_request_count(), 2);
        assert_eq!(mc.total_tokens(), 430); // 100+50 + 200+80
        assert_eq!(mc.llm_latency_histogram(), vec![42.5, 10.0]);
    }

    #[test]
    fn record_tool_invocation_increments_counters() {
        let mc = MetricsCollector::new();
        mc.record_tool_invocation(5.0, true);
        mc.record_tool_invocation(12.3, true);
        mc.record_tool_invocation(8.0, false);

        assert_eq!(mc.tool_invocation_count(), 3);
        assert_eq!(mc.tool_duration_histogram(), vec![5.0, 12.3, 8.0]);
    }

    #[test]
    fn error_counting_for_llm_and_tool() {
        let mc = MetricsCollector::new();
        mc.record_llm_error();
        mc.record_llm_error();
        mc.record_tool_invocation(1.0, false);

        assert_eq!(mc.error_count(), 3);
    }

    #[test]
    fn token_accumulation() {
        let mc = MetricsCollector::new();
        mc.record_llm_request(1.0, 10, 5);
        mc.record_llm_request(1.0, 20, 10);
        mc.record_llm_request(1.0, 0, 0);

        assert_eq!(mc.total_tokens(), 45); // 15 + 30 + 0
    }

    #[test]
    fn reset_clears_everything() {
        let mc = MetricsCollector::new();
        mc.record_llm_request(50.0, 100, 200);
        mc.record_tool_invocation(10.0, true);
        mc.record_llm_error();
        mc.record_tool_invocation(5.0, false);

        // Sanity: non-zero before reset
        assert!(mc.llm_request_count() > 0);
        assert!(mc.tool_invocation_count() > 0);
        assert!(mc.total_tokens() > 0);
        assert!(mc.error_count() > 0);
        assert!(!mc.llm_latency_histogram().is_empty());
        assert!(!mc.tool_duration_histogram().is_empty());

        mc.reset();

        assert_eq!(mc.llm_request_count(), 0);
        assert_eq!(mc.tool_invocation_count(), 0);
        assert_eq!(mc.total_tokens(), 0);
        assert_eq!(mc.error_count(), 0);
        assert!(mc.llm_latency_histogram().is_empty());
        assert!(mc.tool_duration_histogram().is_empty());
    }

    #[test]
    fn metrics_collector_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetricsCollector>();
    }
}
