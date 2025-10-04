//! Structured Logging Infrastructure
//! Week 33 - Phase 4: Monitoring and Observability
//!
//! Provides:
//! - Structured logging setup (tracing_subscriber)
//! - JSON log format
//! - Log levels configuration
//! - Contextual logging
//! - Error logging standards

use serde::Serialize;
use std::fmt;
use tracing::{error, info, warn, Level};

/// Log level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "trace" => LogLevel::Trace,
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" | "warning" => LogLevel::Warn,
            "error" => LogLevel::Error,
            _ => LogLevel::Info,
        }
    }

    pub fn to_tracing_level(&self) -> Level {
        match self {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,

    /// Enable JSON output
    pub json_format: bool,

    /// Enable file output
    pub file_output: bool,

    /// Log file path
    pub file_path: Option<String>,

    /// Include timestamps
    pub include_timestamps: bool,

    /// Include thread IDs
    pub include_thread_ids: bool,

    /// Include source location
    pub include_source: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            json_format: std::env::var("LOG_JSON").unwrap_or_else(|_| "false".to_string()) == "true",
            file_output: false,
            file_path: None,
            include_timestamps: true,
            include_thread_ids: true,
            include_source: true,
        }
    }
}

/// Structured log entry for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
    pub target: String,
    pub thread_id: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub fields: serde_json::Value,
}

/// Error context for detailed error logging
#[derive(Debug, Clone, Serialize)]
pub struct ErrorContext {
    pub error_type: String,
    pub error_message: String,
    pub component: String,
    pub severity: String,
    pub trace_id: Option<String>,
    pub additional_context: serde_json::Value,
}

impl ErrorContext {
    pub fn new(
        error_type: impl ToString,
        error_message: impl ToString,
        component: impl ToString,
    ) -> Self {
        Self {
            error_type: error_type.to_string(),
            error_message: error_message.to_string(),
            component: component.to_string(),
            severity: "error".to_string(),
            trace_id: None,
            additional_context: serde_json::json!({}),
        }
    }

    pub fn with_severity(mut self, severity: impl ToString) -> Self {
        self.severity = severity.to_string();
        self
    }

    pub fn with_trace_id(mut self, trace_id: impl ToString) -> Self {
        self.trace_id = Some(trace_id.to_string());
        self
    }

    pub fn with_context(mut self, key: &str, value: serde_json::Value) -> Self {
        if let Some(obj) = self.additional_context.as_object_mut() {
            obj.insert(key.to_string(), value);
        }
        self
    }

    pub fn log(&self) {
        error!(
            error_type = %self.error_type,
            error_message = %self.error_message,
            component = %self.component,
            severity = %self.severity,
            trace_id = ?self.trace_id,
            context = ?self.additional_context,
            "Error occurred"
        );
    }
}

/// Agent logging helper
pub struct AgentLogger {
    agent_name: String,
}

impl AgentLogger {
    pub fn new(agent_name: impl ToString) -> Self {
        Self {
            agent_name: agent_name.to_string(),
        }
    }

    pub fn info(&self, message: &str) {
        info!(agent = %self.agent_name, "{}", message);
    }

    pub fn warn(&self, message: &str) {
        warn!(agent = %self.agent_name, "{}", message);
    }

    pub fn error(&self, message: &str, error_context: Option<&ErrorContext>) {
        if let Some(ctx) = error_context {
            error!(
                agent = %self.agent_name,
                error_type = %ctx.error_type,
                error_message = %ctx.error_message,
                severity = %ctx.severity,
                "{}",
                message
            );
        } else {
            error!(agent = %self.agent_name, "{}", message);
        }
    }

    pub fn debug(&self, message: &str) {
        tracing::debug!(agent = %self.agent_name, "{}", message);
    }

    pub fn trace(&self, message: &str) {
        tracing::trace!(agent = %self.agent_name, "{}", message);
    }
}

/// Pipeline logging helper
pub struct PipelineLogger {
    pipeline_id: String,
}

impl PipelineLogger {
    pub fn new(pipeline_id: impl ToString) -> Self {
        Self {
            pipeline_id: pipeline_id.to_string(),
        }
    }

    pub fn phase_start(&self, phase_name: &str) {
        info!(
            pipeline_id = %self.pipeline_id,
            phase = %phase_name,
            "Starting pipeline phase"
        );
    }

    pub fn phase_complete(&self, phase_name: &str, duration_ms: f64) {
        info!(
            pipeline_id = %self.pipeline_id,
            phase = %phase_name,
            duration_ms = duration_ms,
            "Pipeline phase completed"
        );
    }

    pub fn phase_error(&self, phase_name: &str, error: &str) {
        error!(
            pipeline_id = %self.pipeline_id,
            phase = %phase_name,
            error = %error,
            "Pipeline phase failed"
        );
    }
}

/// Translation logging helper
pub struct TranslationLogger {
    translation_id: String,
}

impl TranslationLogger {
    pub fn new(translation_id: impl ToString) -> Self {
        Self {
            translation_id: translation_id.to_string(),
        }
    }

    pub fn start(&self, source_lang: &str, target_format: &str, lines: usize) {
        info!(
            translation_id = %self.translation_id,
            source_language = %source_lang,
            target_format = %target_format,
            lines_of_code = lines,
            "Starting translation"
        );
    }

    pub fn progress(&self, step: &str, progress_percent: f64) {
        info!(
            translation_id = %self.translation_id,
            step = %step,
            progress_percent = progress_percent,
            "Translation progress"
        );
    }

    pub fn complete(&self, duration_ms: f64, output_size_bytes: usize) {
        info!(
            translation_id = %self.translation_id,
            duration_ms = duration_ms,
            output_size_bytes = output_size_bytes,
            "Translation completed successfully"
        );
    }

    pub fn failed(&self, error_category: &str, error_message: &str) {
        error!(
            translation_id = %self.translation_id,
            error_category = %error_category,
            error_message = %error_message,
            "Translation failed"
        );
    }
}

/// Performance logging helper
pub struct PerformanceLogger;

impl PerformanceLogger {
    pub fn log_latency(operation: &str, duration_ms: f64) {
        info!(
            operation = %operation,
            duration_ms = duration_ms,
            metric_type = "latency",
            "Performance metric"
        );
    }

    pub fn log_throughput(operation: &str, items_per_second: f64) {
        info!(
            operation = %operation,
            items_per_second = items_per_second,
            metric_type = "throughput",
            "Performance metric"
        );
    }

    pub fn log_resource_usage(component: &str, cpu_percent: f64, memory_mb: f64) {
        info!(
            component = %component,
            cpu_percent = cpu_percent,
            memory_mb = memory_mb,
            metric_type = "resource_usage",
            "Resource usage"
        );
    }
}

/// Audit logging for compliance
pub struct AuditLogger;

impl AuditLogger {
    pub fn log_translation_request(user_id: &str, source_file: &str, ip_address: &str) {
        info!(
            user_id = %user_id,
            source_file = %source_file,
            ip_address = %ip_address,
            event_type = "translation_request",
            "Audit log"
        );
    }

    pub fn log_translation_complete(user_id: &str, translation_id: &str, success: bool) {
        info!(
            user_id = %user_id,
            translation_id = %translation_id,
            success = success,
            event_type = "translation_complete",
            "Audit log"
        );
    }

    pub fn log_error(user_id: &str, error_type: &str, details: &str) {
        error!(
            user_id = %user_id,
            error_type = %error_type,
            details = %details,
            event_type = "error",
            "Audit log"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from_str("info"), LogLevel::Info);
        assert_eq!(LogLevel::from_str("DEBUG"), LogLevel::Debug);
        assert_eq!(LogLevel::from_str("error"), LogLevel::Error);
        assert_eq!(LogLevel::from_str("invalid"), LogLevel::Info);
    }

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.include_timestamps);
    }

    #[test]
    fn test_error_context_builder() {
        let ctx = ErrorContext::new("ParseError", "Invalid syntax", "ingest-agent")
            .with_severity("critical")
            .with_trace_id("trace-123");

        assert_eq!(ctx.error_type, "ParseError");
        assert_eq!(ctx.severity, "critical");
        assert_eq!(ctx.trace_id, Some("trace-123".to_string()));
    }

    #[test]
    fn test_agent_logger() {
        let logger = AgentLogger::new("test-agent");
        assert_eq!(logger.agent_name, "test-agent");
    }

    #[test]
    fn test_pipeline_logger() {
        let logger = PipelineLogger::new("pipeline-123");
        assert_eq!(logger.pipeline_id, "pipeline-123");
    }

    #[test]
    fn test_translation_logger() {
        let logger = TranslationLogger::new("trans-456");
        assert_eq!(logger.translation_id, "trans-456");
    }
}
