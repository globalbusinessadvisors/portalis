// Prometheus Exporters for Portalis Rust Services
// Week 28 - Production Monitoring Integration

use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter,
    IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry, TextEncoder,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Metrics for Transpiler Service
#[derive(Clone)]
pub struct TranspilerMetrics {
    // Request metrics
    pub requests_total: IntCounterVec,
    pub requests_duration: HistogramVec,
    pub requests_in_flight: IntGaugeVec,

    // Translation metrics
    pub translations_total: IntCounterVec,
    pub translation_duration: HistogramVec,
    pub translation_loc: HistogramVec,
    pub translation_complexity: GaugeVec,

    // GPU metrics
    pub gpu_utilization: GaugeVec,
    pub gpu_memory_used: GaugeVec,
    pub gpu_memory_total: GaugeVec,
    pub gpu_temperature: GaugeVec,

    // Error metrics
    pub errors_total: IntCounterVec,
    pub cache_hits: IntCounterVec,
    pub cache_misses: IntCounterVec,

    // Resource metrics
    pub active_workers: IntGauge,
    pub queue_depth: IntGaugeVec,
}

impl TranspilerMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let requests_total = IntCounterVec::new(
            Opts::new(
                "portalis_transpiler_requests_total",
                "Total number of transpilation requests",
            ),
            &["endpoint", "method", "status"],
        )?;
        registry.register(Box::new(requests_total.clone()))?;

        let requests_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_transpiler_request_duration_seconds",
                "Request duration in seconds",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
            &["endpoint", "method"],
        )?;
        registry.register(Box::new(requests_duration.clone()))?;

        let requests_in_flight = IntGaugeVec::new(
            Opts::new(
                "portalis_transpiler_requests_in_flight",
                "Number of requests currently being processed",
            ),
            &["endpoint"],
        )?;
        registry.register(Box::new(requests_in_flight.clone()))?;

        let translations_total = IntCounterVec::new(
            Opts::new(
                "portalis_transpiler_translations_total",
                "Total number of translations",
            ),
            &["source_type", "target_type", "status"],
        )?;
        registry.register(Box::new(translations_total.clone()))?;

        let translation_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_transpiler_translation_duration_seconds",
                "Translation duration in seconds",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]),
            &["source_type", "complexity"],
        )?;
        registry.register(Box::new(translation_duration.clone()))?;

        let translation_loc = HistogramVec::new(
            HistogramOpts::new(
                "portalis_transpiler_translation_loc",
                "Lines of code translated",
            )
            .buckets(vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]),
            &["source_type"],
        )?;
        registry.register(Box::new(translation_loc.clone()))?;

        let translation_complexity = GaugeVec::new(
            Opts::new(
                "portalis_transpiler_translation_complexity",
                "Cyclomatic complexity of translated code",
            ),
            &["translation_id"],
        )?;
        registry.register(Box::new(translation_complexity.clone()))?;

        let gpu_utilization = GaugeVec::new(
            Opts::new(
                "portalis_transpiler_gpu_utilization_percent",
                "GPU utilization percentage",
            ),
            &["gpu_id", "node_id"],
        )?;
        registry.register(Box::new(gpu_utilization.clone()))?;

        let gpu_memory_used = GaugeVec::new(
            Opts::new(
                "portalis_transpiler_gpu_memory_used_bytes",
                "GPU memory used in bytes",
            ),
            &["gpu_id", "node_id"],
        )?;
        registry.register(Box::new(gpu_memory_used.clone()))?;

        let gpu_memory_total = GaugeVec::new(
            Opts::new(
                "portalis_transpiler_gpu_memory_total_bytes",
                "Total GPU memory in bytes",
            ),
            &["gpu_id", "node_id"],
        )?;
        registry.register(Box::new(gpu_memory_total.clone()))?;

        let gpu_temperature = GaugeVec::new(
            Opts::new(
                "portalis_transpiler_gpu_temperature_celsius",
                "GPU temperature in Celsius",
            ),
            &["gpu_id", "node_id"],
        )?;
        registry.register(Box::new(gpu_temperature.clone()))?;

        let errors_total = IntCounterVec::new(
            Opts::new(
                "portalis_transpiler_errors_total",
                "Total number of errors",
            ),
            &["error_type", "endpoint"],
        )?;
        registry.register(Box::new(errors_total.clone()))?;

        let cache_hits = IntCounterVec::new(
            Opts::new(
                "portalis_transpiler_cache_hits_total",
                "Total cache hits",
            ),
            &["cache_type"],
        )?;
        registry.register(Box::new(cache_hits.clone()))?;

        let cache_misses = IntCounterVec::new(
            Opts::new(
                "portalis_transpiler_cache_misses_total",
                "Total cache misses",
            ),
            &["cache_type"],
        )?;
        registry.register(Box::new(cache_misses.clone()))?;

        let active_workers = IntGauge::new(
            "portalis_transpiler_active_workers",
            "Number of active worker processes",
        )?;
        registry.register(Box::new(active_workers.clone()))?;

        let queue_depth = IntGaugeVec::new(
            Opts::new(
                "portalis_transpiler_queue_depth",
                "Number of items in queue",
            ),
            &["priority", "queue_type"],
        )?;
        registry.register(Box::new(queue_depth.clone()))?;

        Ok(Self {
            requests_total,
            requests_duration,
            requests_in_flight,
            translations_total,
            translation_duration,
            translation_loc,
            translation_complexity,
            gpu_utilization,
            gpu_memory_used,
            gpu_memory_total,
            gpu_temperature,
            errors_total,
            cache_hits,
            cache_misses,
            active_workers,
            queue_depth,
        })
    }
}

/// Metrics for Orchestration Service
#[derive(Clone)]
pub struct OrchestrationMetrics {
    pub jobs_submitted: IntCounterVec,
    pub jobs_completed: IntCounterVec,
    pub jobs_failed: IntCounterVec,
    pub job_duration: HistogramVec,
    pub jobs_active: IntGaugeVec,
    pub jobs_queued: IntGaugeVec,
    pub task_fanout: HistogramVec,
}

impl OrchestrationMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let jobs_submitted = IntCounterVec::new(
            Opts::new(
                "portalis_orchestration_jobs_submitted_total",
                "Total jobs submitted",
            ),
            &["job_type", "priority"],
        )?;
        registry.register(Box::new(jobs_submitted.clone()))?;

        let jobs_completed = IntCounterVec::new(
            Opts::new(
                "portalis_orchestration_jobs_completed_total",
                "Total jobs completed",
            ),
            &["job_type", "status"],
        )?;
        registry.register(Box::new(jobs_completed.clone()))?;

        let jobs_failed = IntCounterVec::new(
            Opts::new(
                "portalis_orchestration_jobs_failed_total",
                "Total jobs failed",
            ),
            &["job_type", "error_type"],
        )?;
        registry.register(Box::new(jobs_failed.clone()))?;

        let job_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_orchestration_job_duration_seconds",
                "Job execution duration",
            )
            .buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]),
            &["job_type", "priority"],
        )?;
        registry.register(Box::new(job_duration.clone()))?;

        let jobs_active = IntGaugeVec::new(
            Opts::new(
                "portalis_orchestration_jobs_active",
                "Number of active jobs",
            ),
            &["job_type", "priority"],
        )?;
        registry.register(Box::new(jobs_active.clone()))?;

        let jobs_queued = IntGaugeVec::new(
            Opts::new(
                "portalis_orchestration_jobs_queued",
                "Number of queued jobs",
            ),
            &["job_type", "priority"],
        )?;
        registry.register(Box::new(jobs_queued.clone()))?;

        let task_fanout = HistogramVec::new(
            HistogramOpts::new(
                "portalis_orchestration_task_fanout",
                "Number of tasks per job",
            )
            .buckets(vec![1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]),
            &["job_type"],
        )?;
        registry.register(Box::new(task_fanout.clone()))?;

        Ok(Self {
            jobs_submitted,
            jobs_completed,
            jobs_failed,
            job_duration,
            jobs_active,
            jobs_queued,
            task_fanout,
        })
    }
}

/// Metrics for NeMo Bridge Service
#[derive(Clone)]
pub struct NeMoBridgeMetrics {
    pub model_inference_requests: IntCounterVec,
    pub model_inference_duration: HistogramVec,
    pub model_batch_size: HistogramVec,
    pub triton_requests: IntCounterVec,
    pub triton_errors: IntCounterVec,
}

impl NeMoBridgeMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let model_inference_requests = IntCounterVec::new(
            Opts::new(
                "portalis_nemo_inference_requests_total",
                "Total model inference requests",
            ),
            &["model_name", "status"],
        )?;
        registry.register(Box::new(model_inference_requests.clone()))?;

        let model_inference_duration = HistogramVec::new(
            HistogramOpts::new(
                "portalis_nemo_inference_duration_seconds",
                "Model inference duration",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
            &["model_name"],
        )?;
        registry.register(Box::new(model_inference_duration.clone()))?;

        let model_batch_size = HistogramVec::new(
            HistogramOpts::new(
                "portalis_nemo_batch_size",
                "Inference batch size",
            )
            .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
            &["model_name"],
        )?;
        registry.register(Box::new(model_batch_size.clone()))?;

        let triton_requests = IntCounterVec::new(
            Opts::new(
                "portalis_nemo_triton_requests_total",
                "Total Triton server requests",
            ),
            &["endpoint", "status"],
        )?;
        registry.register(Box::new(triton_requests.clone()))?;

        let triton_errors = IntCounterVec::new(
            Opts::new(
                "portalis_nemo_triton_errors_total",
                "Total Triton server errors",
            ),
            &["error_type"],
        )?;
        registry.register(Box::new(triton_errors.clone()))?;

        Ok(Self {
            model_inference_requests,
            model_inference_duration,
            model_batch_size,
            triton_requests,
            triton_errors,
        })
    }
}

/// DCGM GPU Metrics Collector
pub struct DcgmMetricsCollector {
    dcgm_url: String,
    client: reqwest::Client,
}

impl DcgmMetricsCollector {
    pub fn new(dcgm_url: String) -> Self {
        Self {
            dcgm_url,
            client: reqwest::Client::new(),
        }
    }

    pub async fn collect_gpu_metrics(&self) -> Result<Vec<GpuMetrics>, Box<dyn std::error::Error>> {
        let response = self
            .client
            .get(&format!("{}/metrics", self.dcgm_url))
            .send()
            .await?;

        let text = response.text().await?;

        // Parse Prometheus format from DCGM exporter
        let metrics = Self::parse_dcgm_metrics(&text);

        Ok(metrics)
    }

    fn parse_dcgm_metrics(text: &str) -> Vec<GpuMetrics> {
        // Simple parser for DCGM metrics
        // In production, use a proper Prometheus parser
        let mut metrics = Vec::new();

        for line in text.lines() {
            if line.starts_with("DCGM_FI_DEV_GPU_UTIL") {
                // Parse GPU utilization
                // Format: DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-..."} 45.0
                // TODO: Implement proper parsing
            }
        }

        metrics
    }
}

#[derive(Debug, Clone)]
pub struct GpuMetrics {
    pub gpu_id: String,
    pub uuid: String,
    pub utilization_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub temperature_celsius: f64,
    pub power_usage_watts: f64,
    pub sm_clock_mhz: u32,
    pub memory_clock_mhz: u32,
}

/// Metrics exporter HTTP endpoint
pub async fn metrics_handler(registry: Arc<Registry>) -> Result<String, Box<dyn std::error::Error>> {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;

    Ok(String::from_utf8(buffer)?)
}

/// Request duration middleware wrapper
pub struct MetricsMiddleware<T> {
    inner: T,
    metrics: Arc<TranspilerMetrics>,
}

impl<T> MetricsMiddleware<T> {
    pub fn new(inner: T, metrics: Arc<TranspilerMetrics>) -> Self {
        Self { inner, metrics }
    }

    pub async fn wrap_request<F, R>(&self, endpoint: &str, method: &str, f: F) -> Result<R, Box<dyn std::error::Error>>
    where
        F: std::future::Future<Output = Result<R, Box<dyn std::error::Error>>>,
    {
        let start = Instant::now();

        self.metrics
            .requests_in_flight
            .with_label_values(&[endpoint])
            .inc();

        let result = f.await;

        self.metrics
            .requests_in_flight
            .with_label_values(&[endpoint])
            .dec();

        let duration = start.elapsed();

        let status = if result.is_ok() { "success" } else { "error" };

        self.metrics
            .requests_total
            .with_label_values(&[endpoint, method, status])
            .inc();

        self.metrics
            .requests_duration
            .with_label_values(&[endpoint, method])
            .observe(duration.as_secs_f64());

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpiler_metrics_creation() {
        let registry = Registry::new();
        let metrics = TranspilerMetrics::new(&registry);
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_metrics_recording() {
        let registry = Registry::new();
        let metrics = TranspilerMetrics::new(&registry).unwrap();

        metrics
            .requests_total
            .with_label_values(&["translate", "POST", "success"])
            .inc();

        metrics
            .gpu_utilization
            .with_label_values(&["0", "node-1"])
            .set(75.5);

        // Verify metrics can be gathered
        let metric_families = registry.gather();
        assert!(!metric_families.is_empty());
    }
}
