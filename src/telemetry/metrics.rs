//! Metrics collection for LLM requests, tool invocations, and token usage.
//!
//! `MetricsCollector` uses atomic counters and fixed-bucket histograms so it
//! works both with and without the `telemetry` feature flag, and its memory
//! footprint is constant regardless of how long the process runs.
//! It is `Send + Sync` by construction.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Histogram bucket upper bounds in milliseconds (the last bucket is +inf).
const BUCKET_BOUNDS_MS: [f64; 12] = [
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
];

/// Fixed-bucket duration histogram with constant memory footprint.
#[derive(Debug, Default, Clone)]
struct Histogram {
    /// Per-bucket counts; one extra slot for values above the last bound.
    counts: [u64; BUCKET_BOUNDS_MS.len() + 1],
    count: u64,
    sum_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl Histogram {
    fn record(&mut self, value_ms: f64) {
        let idx = BUCKET_BOUNDS_MS
            .iter()
            .position(|&bound| value_ms <= bound)
            .unwrap_or(BUCKET_BOUNDS_MS.len());
        self.counts[idx] += 1;
        if self.count == 0 {
            self.min_ms = value_ms;
            self.max_ms = value_ms;
        } else {
            self.min_ms = self.min_ms.min(value_ms);
            self.max_ms = self.max_ms.max(value_ms);
        }
        self.count += 1;
        self.sum_ms += value_ms;
    }
}

/// Read-only snapshot of a duration histogram.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramSnapshot {
    /// `(upper_bound_ms, count)` per bucket. The final bucket's bound is
    /// `f64::INFINITY`.
    pub buckets: Vec<(f64, u64)>,
    /// Total number of recorded samples.
    pub count: u64,
    /// Sum of all recorded values in milliseconds.
    pub sum_ms: f64,
    /// Smallest recorded value (0.0 when empty).
    pub min_ms: f64,
    /// Largest recorded value (0.0 when empty).
    pub max_ms: f64,
}

impl HistogramSnapshot {
    /// Mean of recorded values, or `None` when no samples were recorded.
    pub fn mean_ms(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum_ms / self.count as f64)
    }

    /// Whether any samples have been recorded.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Total number of recorded samples.
    pub fn len(&self) -> u64 {
        self.count
    }

    fn from_histogram(h: &Histogram) -> Self {
        let buckets = BUCKET_BOUNDS_MS
            .iter()
            .copied()
            .chain(std::iter::once(f64::INFINITY))
            .zip(h.counts.iter().copied())
            .collect();
        Self {
            buckets,
            count: h.count,
            sum_ms: h.sum_ms,
            min_ms: if h.count > 0 { h.min_ms } else { 0.0 },
            max_ms: if h.count > 0 { h.max_ms } else { 0.0 },
        }
    }
}

/// Collects quantitative measurements for observability.
///
/// Records histograms for LLM request latency and tool execution duration,
/// and counters for total LLM requests, tool invocations, tokens consumed,
/// and errors. Register on an agent via
/// [`AgentBuilder::metrics`](crate::agent::AgentBuilder::metrics) to have the
/// agentic loop record into it automatically.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    total_llm_requests: AtomicU64,
    total_tool_invocations: AtomicU64,
    total_tokens_consumed: AtomicU64,
    total_errors: AtomicU64,
    llm_request_latency: RwLock<Histogram>,
    tool_execution_duration: RwLock<Histogram>,
}

impl MetricsCollector {
    /// Create a new, zeroed `MetricsCollector`.
    pub fn new() -> Self {
        Self::default()
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
        if let Ok(mut hist) = self.llm_request_latency.write() {
            hist.record(latency_ms);
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
        if let Ok(mut hist) = self.tool_execution_duration.write() {
            hist.record(duration_ms);
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

    /// Snapshot of the LLM request latency histogram (milliseconds).
    pub fn llm_latency_histogram(&self) -> HistogramSnapshot {
        self.llm_request_latency
            .read()
            .map(|h| HistogramSnapshot::from_histogram(&h))
            .unwrap_or_else(|e| HistogramSnapshot::from_histogram(&e.into_inner()))
    }

    /// Snapshot of the tool execution duration histogram (milliseconds).
    pub fn tool_duration_histogram(&self) -> HistogramSnapshot {
        self.tool_execution_duration
            .read()
            .map(|h| HistogramSnapshot::from_histogram(&h))
            .unwrap_or_else(|e| HistogramSnapshot::from_histogram(&e.into_inner()))
    }

    /// Reset all counters and histograms to zero / empty.
    pub fn reset(&self) {
        self.total_llm_requests.store(0, Ordering::Relaxed);
        self.total_tool_invocations.store(0, Ordering::Relaxed);
        self.total_tokens_consumed.store(0, Ordering::Relaxed);
        self.total_errors.store(0, Ordering::Relaxed);
        if let Ok(mut hist) = self.llm_request_latency.write() {
            *hist = Histogram::default();
        }
        if let Ok(mut hist) = self.tool_execution_duration.write() {
            *hist = Histogram::default();
        }
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
        let hist = mc.llm_latency_histogram();
        assert_eq!(hist.count, 2);
        assert!((hist.sum_ms - 52.5).abs() < f64::EPSILON);
        assert_eq!(hist.min_ms, 10.0);
        assert_eq!(hist.max_ms, 42.5);
        assert_eq!(hist.mean_ms(), Some(26.25));
    }

    #[test]
    fn record_tool_invocation_increments_counters() {
        let mc = MetricsCollector::new();
        mc.record_tool_invocation(5.0, true);
        mc.record_tool_invocation(12.3, true);
        mc.record_tool_invocation(8.0, false);

        assert_eq!(mc.tool_invocation_count(), 3);
        let hist = mc.tool_duration_histogram();
        assert_eq!(hist.count, 3);
        assert!((hist.sum_ms - 25.3).abs() < 1e-9);
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
    fn bucket_counts_sum_to_total() {
        let mc = MetricsCollector::new();
        for v in [0.5, 3.0, 75.0, 800.0, 99999.0] {
            mc.record_llm_request(v, 0, 0);
        }
        let hist = mc.llm_latency_histogram();
        let bucket_total: u64 = hist.buckets.iter().map(|(_, c)| c).sum();
        assert_eq!(bucket_total, 5);
        // The overflow bucket caught the out-of-range value.
        assert_eq!(hist.buckets.last().unwrap().1, 1);
    }

    #[test]
    fn memory_footprint_is_constant() {
        // Histograms must not grow with sample count (regression test for
        // the unbounded Vec implementation).
        let mc = MetricsCollector::new();
        for i in 0..100_000 {
            mc.record_llm_request((i % 1000) as f64, 1, 1);
        }
        let hist = mc.llm_latency_histogram();
        assert_eq!(hist.count, 100_000);
        assert_eq!(hist.buckets.len(), BUCKET_BOUNDS_MS.len() + 1);
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
