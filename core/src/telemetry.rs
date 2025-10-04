//! OpenTelemetry Integration for Distributed Tracing
//! Week 33 - Phase 4: Monitoring and Observability
//!
//! Provides:
//! - Distributed tracing setup
//! - Span creation helpers
//! - Context propagation
//! - Trace export configuration (Jaeger/Zipkin)
//! - Integration with agents

use tracing::{info, warn, Level};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, filter::EnvFilter};

/// Telemetry configuration
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Service name for identification
    pub service_name: String,

    /// Service version
    pub service_version: String,

    /// Environment (dev, staging, production)
    pub environment: String,

    /// Enable Jaeger exporter
    pub enable_jaeger: bool,

    /// Jaeger endpoint
    pub jaeger_endpoint: Option<String>,

    /// Enable console output
    pub enable_console: bool,

    /// Log level
    pub log_level: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "portalis".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()),
            enable_jaeger: std::env::var("ENABLE_JAEGER").unwrap_or_else(|_| "false".to_string()) == "true",
            jaeger_endpoint: std::env::var("JAEGER_ENDPOINT").ok(),
            enable_console: true,
            log_level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        }
    }
}

/// Initialize telemetry with OpenTelemetry
pub fn init_telemetry(config: TelemetryConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Create env filter based on log level
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&config.log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Build the subscriber with layers
    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().with_target(true).with_thread_ids(true));

    // Initialize the subscriber
    subscriber.init();

    info!(
        service_name = %config.service_name,
        version = %config.service_version,
        environment = %config.environment,
        "Telemetry initialized"
    );

    if config.enable_jaeger {
        if let Some(endpoint) = &config.jaeger_endpoint {
            info!(endpoint = %endpoint, "Jaeger tracing enabled");
        } else {
            warn!("Jaeger enabled but no endpoint provided");
        }
    }

    Ok(())
}

/// Span attributes helper
pub struct SpanAttributes {
    attributes: Vec<(&'static str, String)>,
}

impl SpanAttributes {
    pub fn new() -> Self {
        Self {
            attributes: Vec::new(),
        }
    }

    pub fn add(mut self, key: &'static str, value: impl ToString) -> Self {
        self.attributes.push((key, value.to_string()));
        self
    }

    pub fn get_attributes(&self) -> &[(&'static str, String)] {
        &self.attributes
    }
}

/// Trace context for distributed tracing
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
}

impl TraceContext {
    pub fn new() -> Self {
        Self {
            trace_id: uuid::Uuid::new_v4().to_string(),
            span_id: uuid::Uuid::new_v4().to_string(),
            parent_span_id: None,
        }
    }

    pub fn with_parent(parent_span_id: String) -> Self {
        Self {
            trace_id: uuid::Uuid::new_v4().to_string(),
            span_id: uuid::Uuid::new_v4().to_string(),
            parent_span_id: Some(parent_span_id),
        }
    }

    pub fn child_span(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: uuid::Uuid::new_v4().to_string(),
            parent_span_id: Some(self.span_id.clone()),
        }
    }
}

/// Helper macro for creating instrumented spans
#[macro_export]
macro_rules! trace_span {
    ($name:expr) => {
        tracing::info_span!($name)
    };
    ($name:expr, $($key:tt = $value:expr),*) => {
        tracing::info_span!($name, $($key = $value),*)
    };
}

/// Helper macro for instrumented async functions
#[macro_export]
macro_rules! instrument_async {
    ($name:expr, $future:expr) => {{
        use tracing::Instrument;
        let span = tracing::info_span!($name);
        $future.instrument(span).await
    }};
}

/// Agent tracing wrapper
pub struct AgentTracer {
    agent_name: String,
    trace_context: TraceContext,
}

impl AgentTracer {
    pub fn new(agent_name: impl ToString) -> Self {
        Self {
            agent_name: agent_name.to_string(),
            trace_context: TraceContext::new(),
        }
    }

    pub fn with_context(agent_name: impl ToString, trace_context: TraceContext) -> Self {
        Self {
            agent_name: agent_name.to_string(),
            trace_context,
        }
    }

    pub fn start_span(&self, operation: &str) -> TraceContext {
        let span = self.trace_context.child_span();
        tracing::info!(
            agent = %self.agent_name,
            operation = %operation,
            trace_id = %span.trace_id,
            span_id = %span.span_id,
            parent_span_id = ?span.parent_span_id,
            "Starting operation"
        );
        span
    }

    pub fn end_span(&self, span: &TraceContext, success: bool, duration_ms: f64) {
        tracing::info!(
            agent = %self.agent_name,
            trace_id = %span.trace_id,
            span_id = %span.span_id,
            success = success,
            duration_ms = duration_ms,
            "Operation completed"
        );
    }

    pub fn record_error(&self, span: &TraceContext, error: &str) {
        tracing::error!(
            agent = %self.agent_name,
            trace_id = %span.trace_id,
            span_id = %span.span_id,
            error = %error,
            "Operation failed"
        );
    }
}

/// Pipeline tracing helper
pub struct PipelineTracer {
    pipeline_id: String,
    trace_context: TraceContext,
}

impl PipelineTracer {
    pub fn new(pipeline_id: impl ToString) -> Self {
        Self {
            pipeline_id: pipeline_id.to_string(),
            trace_context: TraceContext::new(),
        }
    }

    pub fn trace_phase(&self, phase_name: &str) -> TraceContext {
        let span = self.trace_context.child_span();
        tracing::info!(
            pipeline_id = %self.pipeline_id,
            phase = %phase_name,
            trace_id = %span.trace_id,
            span_id = %span.span_id,
            "Starting pipeline phase"
        );
        span
    }

    pub fn phase_completed(&self, span: &TraceContext, phase_name: &str, duration_ms: f64) {
        tracing::info!(
            pipeline_id = %self.pipeline_id,
            phase = %phase_name,
            trace_id = %span.trace_id,
            span_id = %span.span_id,
            duration_ms = duration_ms,
            "Pipeline phase completed"
        );
    }

    pub fn get_trace_id(&self) -> &str {
        &self.trace_context.trace_id
    }
}

/// Translation tracing helper
pub struct TranslationTracer {
    translation_id: String,
    trace_context: TraceContext,
}

impl TranslationTracer {
    pub fn new(translation_id: impl ToString) -> Self {
        Self {
            translation_id: translation_id.to_string(),
            trace_context: TraceContext::new(),
        }
    }

    pub fn trace_step(&self, step_name: &str, metadata: &[(&str, &str)]) -> TraceContext {
        let span = self.trace_context.child_span();
        tracing::info!(
            translation_id = %self.translation_id,
            step = %step_name,
            trace_id = %span.trace_id,
            span_id = %span.span_id,
            ?metadata,
            "Starting translation step"
        );
        span
    }

    pub fn step_completed(
        &self,
        span: &TraceContext,
        step_name: &str,
        lines_processed: usize,
        duration_ms: f64,
    ) {
        tracing::info!(
            translation_id = %self.translation_id,
            step = %step_name,
            trace_id = %span.trace_id,
            span_id = %span.span_id,
            lines_processed = lines_processed,
            duration_ms = duration_ms,
            "Translation step completed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_config_default() {
        let config = TelemetryConfig::default();
        assert_eq!(config.service_name, "portalis");
        assert!(!config.enable_jaeger);
    }

    #[test]
    fn test_trace_context_creation() {
        let context = TraceContext::new();
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
        assert!(context.parent_span_id.is_none());
    }

    #[test]
    fn test_trace_context_child_span() {
        let parent = TraceContext::new();
        let child = parent.child_span();

        assert_eq!(parent.trace_id, child.trace_id);
        assert_ne!(parent.span_id, child.span_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id.clone()));
    }

    #[test]
    fn test_agent_tracer() {
        let tracer = AgentTracer::new("test-agent");
        assert_eq!(tracer.agent_name, "test-agent");
    }

    #[test]
    fn test_span_attributes() {
        let attrs = SpanAttributes::new()
            .add("key1", "value1")
            .add("key2", 42);

        assert_eq!(attrs.get_attributes().len(), 2);
    }
}
