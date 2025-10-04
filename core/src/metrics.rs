//! Central Metrics Registry for Portalis
//! Week 33 - Phase 4: Monitoring and Observability
//!
//! This module provides a comprehensive metrics system for tracking:
//! - Translation success/failure rates
//! - Per-agent execution time
//! - Pipeline phase duration
//! - WASM performance metrics
//! - Error categorization
//! - Cache hit rates

use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
};
use std::sync::Arc;
use std::time::Instant;

/// Central metrics registry for the Portalis platform
#[derive(Clone)]
pub struct PortalisMetrics {
    /// Prometheus registry
    pub registry: Arc<Registry>,

    /// Translation metrics
    pub translation: TranslationMetrics,

    /// Agent execution metrics
    pub agents: AgentMetrics,

    /// Pipeline metrics
    pub pipeline: PipelineMetrics,

    /// WASM metrics
    pub wasm: WasmMetrics,

    /// Error metrics
    pub errors: ErrorMetrics,

    /// Cache metrics
    pub cache: CacheMetrics,

    /// System metrics
    pub system: SystemMetrics,
}

impl PortalisMetrics {
    /// Create a new metrics registry with all metric families
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Arc::new(Registry::new());

        Ok(Self {
            translation: TranslationMetrics::new(&registry)?,
            agents: AgentMetrics::new(&registry)?,
            pipeline: PipelineMetrics::new(&registry)?,
            wasm: WasmMetrics::new(&registry)?,
            errors: ErrorMetrics::new(&registry)?,
            cache: CacheMetrics::new(&registry)?,
            system: SystemMetrics::new(&registry)?,
            registry: registry.clone(),
        })
    }

    /// Export metrics in Prometheus text format
    pub fn export(&self) -> Result<String, prometheus::Error> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }
}

/// Translation-specific metrics
#[derive(Clone)]
pub struct TranslationMetrics {
    /// Total translations attempted
    pub translations_total: IntCounterVec,

    /// Successful translations
    pub translations_success: IntCounterVec,

    /// Failed translations
    pub translations_failed: IntCounterVec,

    /// Translation duration histogram
    pub translation_duration: HistogramVec,

    /// Lines of code translated
    pub translation_loc: HistogramVec,

    /// Translation complexity score
    pub translation_complexity: GaugeVec,

    /// Success rate (computed metric)
    pub success_rate: GaugeVec,

    /// Current translations in progress
    pub translations_in_progress: IntGaugeVec,
}

impl TranslationMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let translations_total = IntCounterVec::new(
            Opts::new(
                "portalis_translations_total",
                "Total number of translation attempts",
            ),
            &["source_language", "target_format"],
        )?;
        registry.register(Box::new(translations_total.clone()))?;

        let translations_success = IntCounterVec::new(
            Opts::new(
                "portalis_translations_success_total",
                "Total number of successful translations",
            ),
            &["source_language", "target_format"],
        )?;
        registry.register(Box::new(translations_success.clone()))?;

        let translations_failed = IntCounterVec::new(
            Opts::new(
                "portalis_translations_failed_total",
                "Total number of failed translations",
            ),
            &["source_language", "target_format", "error_category"],
        )?;
        registry.register(Box::new(translations_failed.clone()))?;

        let translation_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_translation_duration_seconds",
                "Time taken to complete translation",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]),
            &["source_language", "complexity_level"],
        )?;
        registry.register(Box::new(translation_duration.clone()))?;

        let translation_loc = HistogramVec::new(
            HistogramOpts::new(
                "portalis_translation_lines_of_code",
                "Number of lines of code translated",
            )
            .buckets(vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]),
            &["source_language"],
        )?;
        registry.register(Box::new(translation_loc.clone()))?;

        let translation_complexity = GaugeVec::new(
            Opts::new(
                "portalis_translation_complexity_score",
                "Cyclomatic complexity score of translation",
            ),
            &["translation_id", "source_language"],
        )?;
        registry.register(Box::new(translation_complexity.clone()))?;

        let success_rate = GaugeVec::new(
            Opts::new(
                "portalis_translation_success_rate",
                "Translation success rate percentage",
            ),
            &["source_language", "target_format"],
        )?;
        registry.register(Box::new(success_rate.clone()))?;

        let translations_in_progress = IntGaugeVec::new(
            Opts::new(
                "portalis_translations_in_progress",
                "Number of translations currently being processed",
            ),
            &["source_language"],
        )?;
        registry.register(Box::new(translations_in_progress.clone()))?;

        Ok(Self {
            translations_total,
            translations_success,
            translations_failed,
            translation_duration,
            translation_loc,
            translation_complexity,
            success_rate,
            translations_in_progress,
        })
    }
}

/// Per-agent execution metrics
#[derive(Clone)]
pub struct AgentMetrics {
    /// Agent execution count
    pub agent_executions: IntCounterVec,

    /// Agent execution duration
    pub agent_duration: HistogramVec,

    /// Agent success/failure
    pub agent_status: IntCounterVec,

    /// Agent resource usage
    pub agent_memory_bytes: GaugeVec,

    /// Agent CPU usage
    pub agent_cpu_percent: GaugeVec,

    /// Active agents
    pub agents_active: IntGaugeVec,
}

impl AgentMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let agent_executions = IntCounterVec::new(
            Opts::new(
                "portalis_agent_executions_total",
                "Total number of agent executions",
            ),
            &["agent_name", "agent_type"],
        )?;
        registry.register(Box::new(agent_executions.clone()))?;

        let agent_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_agent_execution_duration_seconds",
                "Agent execution duration in seconds",
            )
            .buckets(vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]),
            &["agent_name", "agent_type"],
        )?;
        registry.register(Box::new(agent_duration.clone()))?;

        let agent_status = IntCounterVec::new(
            Opts::new(
                "portalis_agent_status_total",
                "Agent execution status counts",
            ),
            &["agent_name", "status"],
        )?;
        registry.register(Box::new(agent_status.clone()))?;

        let agent_memory_bytes = GaugeVec::new(
            Opts::new(
                "portalis_agent_memory_bytes",
                "Agent memory usage in bytes",
            ),
            &["agent_name"],
        )?;
        registry.register(Box::new(agent_memory_bytes.clone()))?;

        let agent_cpu_percent = GaugeVec::new(
            Opts::new(
                "portalis_agent_cpu_percent",
                "Agent CPU usage percentage",
            ),
            &["agent_name"],
        )?;
        registry.register(Box::new(agent_cpu_percent.clone()))?;

        let agents_active = IntGaugeVec::new(
            Opts::new(
                "portalis_agents_active",
                "Number of currently active agents",
            ),
            &["agent_type"],
        )?;
        registry.register(Box::new(agents_active.clone()))?;

        Ok(Self {
            agent_executions,
            agent_duration,
            agent_status,
            agent_memory_bytes,
            agent_cpu_percent,
            agents_active,
        })
    }
}

/// Pipeline phase metrics
#[derive(Clone)]
pub struct PipelineMetrics {
    /// Phase duration
    pub phase_duration: HistogramVec,

    /// Phase success/failure
    pub phase_status: IntCounterVec,

    /// Pipeline end-to-end duration
    pub pipeline_duration: HistogramVec,

    /// Active pipelines
    pub pipelines_active: IntGauge,

    /// Pipeline queue depth
    pub pipeline_queue_depth: IntGaugeVec,
}

impl PipelineMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let phase_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_pipeline_phase_duration_seconds",
                "Duration of each pipeline phase",
            )
            .buckets(vec![0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]),
            &["phase_name"],
        )?;
        registry.register(Box::new(phase_duration.clone()))?;

        let phase_status = IntCounterVec::new(
            Opts::new(
                "portalis_pipeline_phase_status_total",
                "Phase execution status counts",
            ),
            &["phase_name", "status"],
        )?;
        registry.register(Box::new(phase_status.clone()))?;

        let pipeline_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_pipeline_duration_seconds",
                "End-to-end pipeline duration",
            )
            .buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]),
            &["pipeline_type"],
        )?;
        registry.register(Box::new(pipeline_duration.clone()))?;

        let pipelines_active = IntGauge::new(
            "portalis_pipelines_active",
            "Number of currently active pipelines",
        )?;
        registry.register(Box::new(pipelines_active.clone()))?;

        let pipeline_queue_depth = IntGaugeVec::new(
            Opts::new(
                "portalis_pipeline_queue_depth",
                "Number of pipelines waiting in queue",
            ),
            &["priority"],
        )?;
        registry.register(Box::new(pipeline_queue_depth.clone()))?;

        Ok(Self {
            phase_duration,
            phase_status,
            pipeline_duration,
            pipelines_active,
            pipeline_queue_depth,
        })
    }
}

/// WASM-specific performance metrics
#[derive(Clone)]
pub struct WasmMetrics {
    /// WASM compilation time
    pub wasm_compile_duration: HistogramVec,

    /// WASM binary size
    pub wasm_binary_size_bytes: HistogramVec,

    /// WASM optimization level
    pub wasm_optimization_level: GaugeVec,

    /// WASM execution time
    pub wasm_execution_duration: HistogramVec,

    /// WASM memory usage
    pub wasm_memory_bytes: GaugeVec,

    /// WASM module count
    pub wasm_modules_total: IntCounter,
}

impl WasmMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let wasm_compile_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_wasm_compile_duration_seconds",
                "Time taken to compile WASM module",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
            &["optimization_level"],
        )?;
        registry.register(Box::new(wasm_compile_duration.clone()))?;

        let wasm_binary_size_bytes = HistogramVec::new(
            HistogramOpts::new(
                "portalis_wasm_binary_size_bytes",
                "Size of compiled WASM binary",
            )
            .buckets(vec![1024.0, 10240.0, 102400.0, 1024000.0, 10240000.0]),
            &["optimization_level"],
        )?;
        registry.register(Box::new(wasm_binary_size_bytes.clone()))?;

        let wasm_optimization_level = GaugeVec::new(
            Opts::new(
                "portalis_wasm_optimization_level",
                "WASM optimization level (0-3)",
            ),
            &["module_id"],
        )?;
        registry.register(Box::new(wasm_optimization_level.clone()))?;

        let wasm_execution_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_wasm_execution_duration_seconds",
                "WASM module execution time",
            )
            .buckets(vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0]),
            &["module_name"],
        )?;
        registry.register(Box::new(wasm_execution_duration.clone()))?;

        let wasm_memory_bytes = GaugeVec::new(
            Opts::new(
                "portalis_wasm_memory_bytes",
                "WASM module memory usage",
            ),
            &["module_name"],
        )?;
        registry.register(Box::new(wasm_memory_bytes.clone()))?;

        let wasm_modules_total = IntCounter::new(
            "portalis_wasm_modules_total",
            "Total number of WASM modules created",
        )?;
        registry.register(Box::new(wasm_modules_total.clone()))?;

        Ok(Self {
            wasm_compile_duration,
            wasm_binary_size_bytes,
            wasm_optimization_level,
            wasm_execution_duration,
            wasm_memory_bytes,
            wasm_modules_total,
        })
    }
}

/// Error categorization and tracking metrics
#[derive(Clone)]
pub struct ErrorMetrics {
    /// Total errors by category
    pub errors_total: IntCounterVec,

    /// Parse errors
    pub parse_errors: IntCounterVec,

    /// Translation errors
    pub translation_errors: IntCounterVec,

    /// Compilation errors
    pub compilation_errors: IntCounterVec,

    /// Runtime errors
    pub runtime_errors: IntCounterVec,

    /// Error recovery attempts
    pub error_recoveries: IntCounterVec,
}

impl ErrorMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let errors_total = IntCounterVec::new(
            Opts::new(
                "portalis_errors_total",
                "Total errors by category",
            ),
            &["category", "severity", "component"],
        )?;
        registry.register(Box::new(errors_total.clone()))?;

        let parse_errors = IntCounterVec::new(
            Opts::new(
                "portalis_parse_errors_total",
                "Parse errors encountered",
            ),
            &["error_type", "source_language"],
        )?;
        registry.register(Box::new(parse_errors.clone()))?;

        let translation_errors = IntCounterVec::new(
            Opts::new(
                "portalis_translation_errors_total",
                "Translation errors encountered",
            ),
            &["error_type", "phase"],
        )?;
        registry.register(Box::new(translation_errors.clone()))?;

        let compilation_errors = IntCounterVec::new(
            Opts::new(
                "portalis_compilation_errors_total",
                "Compilation errors encountered",
            ),
            &["error_type", "target_format"],
        )?;
        registry.register(Box::new(compilation_errors.clone()))?;

        let runtime_errors = IntCounterVec::new(
            Opts::new(
                "portalis_runtime_errors_total",
                "Runtime errors encountered",
            ),
            &["error_type", "component"],
        )?;
        registry.register(Box::new(runtime_errors.clone()))?;

        let error_recoveries = IntCounterVec::new(
            Opts::new(
                "portalis_error_recoveries_total",
                "Successful error recovery attempts",
            ),
            &["error_category", "recovery_method"],
        )?;
        registry.register(Box::new(error_recoveries.clone()))?;

        Ok(Self {
            errors_total,
            parse_errors,
            translation_errors,
            compilation_errors,
            runtime_errors,
            error_recoveries,
        })
    }
}

/// Cache performance metrics
#[derive(Clone)]
pub struct CacheMetrics {
    /// Cache hits
    pub cache_hits: IntCounterVec,

    /// Cache misses
    pub cache_misses: IntCounterVec,

    /// Cache evictions
    pub cache_evictions: IntCounterVec,

    /// Cache size
    pub cache_size_bytes: GaugeVec,

    /// Cache entry count
    pub cache_entries: GaugeVec,

    /// Cache hit rate (computed)
    pub cache_hit_rate: GaugeVec,
}

impl CacheMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let cache_hits = IntCounterVec::new(
            Opts::new(
                "portalis_cache_hits_total",
                "Total cache hits",
            ),
            &["cache_name", "cache_type"],
        )?;
        registry.register(Box::new(cache_hits.clone()))?;

        let cache_misses = IntCounterVec::new(
            Opts::new(
                "portalis_cache_misses_total",
                "Total cache misses",
            ),
            &["cache_name", "cache_type"],
        )?;
        registry.register(Box::new(cache_misses.clone()))?;

        let cache_evictions = IntCounterVec::new(
            Opts::new(
                "portalis_cache_evictions_total",
                "Total cache evictions",
            ),
            &["cache_name", "reason"],
        )?;
        registry.register(Box::new(cache_evictions.clone()))?;

        let cache_size_bytes = GaugeVec::new(
            Opts::new(
                "portalis_cache_size_bytes",
                "Cache size in bytes",
            ),
            &["cache_name"],
        )?;
        registry.register(Box::new(cache_size_bytes.clone()))?;

        let cache_entries = GaugeVec::new(
            Opts::new(
                "portalis_cache_entries",
                "Number of entries in cache",
            ),
            &["cache_name"],
        )?;
        registry.register(Box::new(cache_entries.clone()))?;

        let cache_hit_rate = GaugeVec::new(
            Opts::new(
                "portalis_cache_hit_rate",
                "Cache hit rate percentage",
            ),
            &["cache_name"],
        )?;
        registry.register(Box::new(cache_hit_rate.clone()))?;

        Ok(Self {
            cache_hits,
            cache_misses,
            cache_evictions,
            cache_size_bytes,
            cache_entries,
            cache_hit_rate,
        })
    }
}

/// System-level metrics
#[derive(Clone)]
pub struct SystemMetrics {
    /// CPU usage
    pub cpu_usage_percent: Gauge,

    /// Memory usage
    pub memory_usage_bytes: Gauge,

    /// Disk usage
    pub disk_usage_bytes: GaugeVec,

    /// Network I/O
    pub network_io_bytes: CounterVec,

    /// Process count
    pub process_count: IntGauge,

    /// Uptime
    pub uptime_seconds: Counter,
}

impl SystemMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let cpu_usage_percent = Gauge::new(
            "portalis_cpu_usage_percent",
            "CPU usage percentage",
        )?;
        registry.register(Box::new(cpu_usage_percent.clone()))?;

        let memory_usage_bytes = Gauge::new(
            "portalis_memory_usage_bytes",
            "Memory usage in bytes",
        )?;
        registry.register(Box::new(memory_usage_bytes.clone()))?;

        let disk_usage_bytes = GaugeVec::new(
            Opts::new(
                "portalis_disk_usage_bytes",
                "Disk usage in bytes",
            ),
            &["mount_point"],
        )?;
        registry.register(Box::new(disk_usage_bytes.clone()))?;

        let network_io_bytes = CounterVec::new(
            Opts::new(
                "portalis_network_io_bytes_total",
                "Network I/O in bytes",
            ),
            &["direction", "interface"],
        )?;
        registry.register(Box::new(network_io_bytes.clone()))?;

        let process_count = IntGauge::new(
            "portalis_process_count",
            "Number of active processes",
        )?;
        registry.register(Box::new(process_count.clone()))?;

        let uptime_seconds = Counter::new(
            "portalis_uptime_seconds_total",
            "System uptime in seconds",
        )?;
        registry.register(Box::new(uptime_seconds.clone()))?;

        Ok(Self {
            cpu_usage_percent,
            memory_usage_bytes,
            disk_usage_bytes,
            network_io_bytes,
            process_count,
            uptime_seconds,
        })
    }
}

/// Helper for timing operations
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn observe_duration(&self, histogram: &Histogram) {
        histogram.observe(self.start.elapsed().as_secs_f64());
    }

    pub fn observe_duration_with_labels(&self, histogram: &HistogramVec, labels: &[&str]) {
        histogram
            .with_label_values(labels)
            .observe(self.start.elapsed().as_secs_f64());
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = PortalisMetrics::new();
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_metrics_export() {
        let metrics = PortalisMetrics::new().unwrap();
        let export = metrics.export();
        assert!(export.is_ok());
        assert!(!export.unwrap().is_empty());
    }

    #[test]
    fn test_translation_metrics() {
        let registry = Registry::new();
        let metrics = TranslationMetrics::new(&registry).unwrap();

        metrics.translations_total
            .with_label_values(&["python", "wasm"])
            .inc();

        metrics.translations_success
            .with_label_values(&["python", "wasm"])
            .inc();

        let families = registry.gather();
        assert!(!families.is_empty());
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_secs();
        assert!(elapsed >= 0.01);
    }
}
