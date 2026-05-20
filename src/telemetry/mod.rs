//! Observability and telemetry integration (feature-gated).
//!
//! When the `telemetry` Cargo feature is enabled, this module configures
//! OpenTelemetry-compatible tracing exporters (stdout, OTLP, Jaeger).
//! When disabled, `TelemetryConfig::init()` sets up a basic `tracing`
//! subscriber with the configured log level — zero OTEL overhead.

pub mod metrics;

pub use metrics::MetricsCollector;

use crate::error::KovaError;

/// Protocol used by the OTLP exporter.
#[derive(Debug, Clone, PartialEq)]
pub enum OtlpProtocol {
    Grpc,
    Http,
}

/// Exporter backend configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum ExporterConfig {
    /// Write spans/logs to stdout.
    Stdout,
    /// Export via OTLP (gRPC or HTTP).
    Otlp {
        endpoint: String,
        protocol: OtlpProtocol,
    },
    /// Export to a Jaeger endpoint.
    Jaeger { endpoint: String },
}

/// Configuration for the SDK's observability layer.
///
/// Use the builder methods to customise, then call [`init`](TelemetryConfig::init)
/// to install the global tracing subscriber.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub log_level: tracing::Level,
    pub exporter: ExporterConfig,
    pub sampling_rate: f64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_level: tracing::Level::INFO,
            exporter: ExporterConfig::Stdout,
            sampling_rate: 1.0,
        }
    }
}

impl TelemetryConfig {
    /// Create a new builder starting from defaults.
    pub fn builder() -> TelemetryConfigBuilder {
        TelemetryConfigBuilder::default()
    }

    /// Initialise the global tracing subscriber.
    ///
    /// When the `telemetry` feature is enabled this sets up an
    /// OpenTelemetry pipeline with the configured exporter and sampling
    /// rate.  Without the feature it installs a plain `tracing_subscriber`
    /// with the requested log level.
    pub fn init(&self) -> Result<(), KovaError> {
        #[cfg(feature = "telemetry")]
        {
            self.init_with_otel()
        }
        #[cfg(not(feature = "telemetry"))]
        {
            self.init_basic()
        }
    }

    /// Basic subscriber — no OTEL dependencies.
    #[cfg(not(feature = "telemetry"))]
    fn init_basic(&self) -> Result<(), KovaError> {
        use tracing_subscriber::{fmt, EnvFilter};

        let filter = EnvFilter::new(self.log_level.to_string());
        fmt()
            .with_env_filter(filter)
            .try_init()
            .map_err(|e| KovaError::Build(format!("Failed to init tracing subscriber: {e}")))?;
        Ok(())
    }

    /// Full OTEL subscriber.
    #[cfg(feature = "telemetry")]
    fn init_with_otel(&self) -> Result<(), KovaError> {
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry_otlp::WithExportConfig;
        use opentelemetry_sdk::{
            runtime,
            trace::{Sampler, TracerProvider},
        };
        use tracing_opentelemetry::OpenTelemetryLayer;
        use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

        let sampler = if (self.sampling_rate - 1.0_f64).abs() < f64::EPSILON {
            Sampler::AlwaysOn
        } else if self.sampling_rate <= 0.0 {
            Sampler::AlwaysOff
        } else {
            Sampler::TraceIdRatioBased(self.sampling_rate)
        };

        let provider = match &self.exporter {
            ExporterConfig::Stdout => {
                let exporter = opentelemetry_stdout::SpanExporter::default();
                TracerProvider::builder()
                    .with_batch_exporter(exporter, runtime::Tokio)
                    .with_sampler(sampler)
                    .build()
            }
            ExporterConfig::Otlp { endpoint, protocol } => {
                let exporter = match protocol {
                    OtlpProtocol::Grpc => opentelemetry_otlp::SpanExporter::builder()
                        .with_tonic()
                        .with_endpoint(endpoint)
                        .build()
                        .map_err(|e| KovaError::Build(format!("OTLP gRPC exporter error: {e}")))?,
                    OtlpProtocol::Http => opentelemetry_otlp::SpanExporter::builder()
                        .with_http()
                        .with_endpoint(endpoint)
                        .build()
                        .map_err(|e| KovaError::Build(format!("OTLP HTTP exporter error: {e}")))?,
                };
                TracerProvider::builder()
                    .with_batch_exporter(exporter, runtime::Tokio)
                    .with_sampler(sampler)
                    .build()
            }
            ExporterConfig::Jaeger { endpoint } => {
                // Jaeger supports OTLP natively; export via OTLP/gRPC to the Jaeger endpoint.
                let exporter = opentelemetry_otlp::SpanExporter::builder()
                    .with_tonic()
                    .with_endpoint(endpoint)
                    .build()
                    .map_err(|e| KovaError::Build(format!("Jaeger exporter error: {e}")))?;
                TracerProvider::builder()
                    .with_batch_exporter(exporter, runtime::Tokio)
                    .with_sampler(sampler)
                    .build()
            }
        };

        let tracer = provider.tracer("kova");
        let otel_layer = OpenTelemetryLayer::new(tracer);
        let filter = EnvFilter::new(self.log_level.to_string());

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer())
            .with(otel_layer)
            .try_init()
            .map_err(|e| KovaError::Build(format!("Failed to init tracing subscriber: {e}")))?;

        Ok(())
    }
}

/// Builder for [`TelemetryConfig`].
#[derive(Debug, Clone)]
pub struct TelemetryConfigBuilder {
    config: TelemetryConfig,
}

impl Default for TelemetryConfigBuilder {
    fn default() -> Self {
        Self {
            config: TelemetryConfig::default(),
        }
    }
}

impl TelemetryConfigBuilder {
    /// Set the minimum log level.
    pub fn log_level(mut self, level: tracing::Level) -> Self {
        self.config.log_level = level;
        self
    }

    /// Set the exporter backend.
    pub fn exporter(mut self, exporter: ExporterConfig) -> Self {
        self.config.exporter = exporter;
        self
    }

    /// Set the trace sampling rate (0.0 – 1.0).
    pub fn sampling_rate(mut self, rate: f64) -> Self {
        self.config.sampling_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Build the final [`TelemetryConfig`].
    pub fn build(self) -> TelemetryConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = TelemetryConfig::default();
        assert_eq!(cfg.log_level, tracing::Level::INFO);
        assert_eq!(cfg.exporter, ExporterConfig::Stdout);
        assert!((cfg.sampling_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_sets_all_fields() {
        let cfg = TelemetryConfig::builder()
            .log_level(tracing::Level::DEBUG)
            .exporter(ExporterConfig::Otlp {
                endpoint: "http://localhost:4317".into(),
                protocol: OtlpProtocol::Grpc,
            })
            .sampling_rate(0.5)
            .build();

        assert_eq!(cfg.log_level, tracing::Level::DEBUG);
        assert_eq!(
            cfg.exporter,
            ExporterConfig::Otlp {
                endpoint: "http://localhost:4317".into(),
                protocol: OtlpProtocol::Grpc,
            }
        );
        assert!((cfg.sampling_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_clamps_sampling_rate() {
        let cfg = TelemetryConfig::builder().sampling_rate(2.0).build();
        assert!((cfg.sampling_rate - 1.0).abs() < f64::EPSILON);

        let cfg = TelemetryConfig::builder().sampling_rate(-0.5).build();
        assert!(cfg.sampling_rate.abs() < f64::EPSILON);
    }

    #[test]
    fn exporter_config_variants() {
        let stdout = ExporterConfig::Stdout;
        let otlp_grpc = ExporterConfig::Otlp {
            endpoint: "http://localhost:4317".into(),
            protocol: OtlpProtocol::Grpc,
        };
        let otlp_http = ExporterConfig::Otlp {
            endpoint: "http://localhost:4318".into(),
            protocol: OtlpProtocol::Http,
        };
        let jaeger = ExporterConfig::Jaeger {
            endpoint: "http://localhost:14250".into(),
        };

        // Verify Debug works on all variants
        assert!(!format!("{stdout:?}").is_empty());
        assert!(!format!("{otlp_grpc:?}").is_empty());
        assert!(!format!("{otlp_http:?}").is_empty());
        assert!(!format!("{jaeger:?}").is_empty());
    }

    #[test]
    fn builder_jaeger_exporter() {
        let cfg = TelemetryConfig::builder()
            .log_level(tracing::Level::WARN)
            .exporter(ExporterConfig::Jaeger {
                endpoint: "http://jaeger:14250".into(),
            })
            .build();

        assert_eq!(cfg.log_level, tracing::Level::WARN);
        assert_eq!(
            cfg.exporter,
            ExporterConfig::Jaeger {
                endpoint: "http://jaeger:14250".into(),
            }
        );
    }

    #[test]
    fn builder_all_log_levels() {
        for level in [
            tracing::Level::TRACE,
            tracing::Level::DEBUG,
            tracing::Level::INFO,
            tracing::Level::WARN,
            tracing::Level::ERROR,
        ] {
            let cfg = TelemetryConfig::builder().log_level(level).build();
            assert_eq!(cfg.log_level, level);
        }
    }
}
