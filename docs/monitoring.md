# Portalis Monitoring and Observability Guide
**Week 33 - Phase 4: Production Monitoring Implementation**

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Metrics Reference](#metrics-reference)
4. [Dashboards](#dashboards)
5. [Alert Rules](#alert-rules)
6. [Health Checks](#health-checks)
7. [Logging](#logging)
8. [Distributed Tracing](#distributed-tracing)
9. [Troubleshooting](#troubleshooting)
10. [SLO/SLA Definitions](#slosla-definitions)
11. [Runbooks](#runbooks)

## Overview

Portalis monitoring provides comprehensive observability across all system components with:

- **Metrics**: Prometheus exporters for all services
- **Dashboards**: Grafana dashboards for visualization
- **Alerts**: Intelligent alerting with runbooks
- **Health Checks**: Liveness and readiness probes
- **Logging**: Structured JSON logging
- **Tracing**: OpenTelemetry distributed tracing

### Key Metrics

- Translation success rate (target: >95%)
- P95 latency (target: <5s)
- Throughput (target: >100 QPS)
- Error rates by category
- Resource utilization (CPU, Memory, GPU)
- WASM compilation and execution metrics

## Architecture

```
┌─────────────┐
│  Portalis   │
│   Services  │
└─────┬───────┘
      │ Metrics Export
      ▼
┌─────────────┐
│ Prometheus  │──────┐
│   Server    │      │ Scrape
└─────┬───────┘      │
      │              │
      │ PromQL       │
      ▼              │
┌─────────────┐      │
│   Grafana   │      │
│  Dashboards │      │
└─────────────┘      │
                     │
┌────────────────────┘
│
▼
┌─────────────┐
│ AlertManager│
│   & Alerts  │
└─────────────┘

┌─────────────┐
│ Application │
│    Logs     │──────▶ Structured JSON
└─────────────┘

┌─────────────┐
│   Traces    │──────▶ Jaeger/Zipkin
└─────────────┘
```

## Metrics Reference

### Translation Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `portalis_translations_total` | Counter | Total translation attempts | source_language, target_format |
| `portalis_translations_success_total` | Counter | Successful translations | source_language, target_format |
| `portalis_translations_failed_total` | Counter | Failed translations | source_language, target_format, error_category |
| `portalis_translation_duration_seconds` | Histogram | Translation duration | source_language, complexity_level |
| `portalis_translation_lines_of_code` | Histogram | Lines of code translated | source_language |
| `portalis_translation_complexity_score` | Gauge | Cyclomatic complexity | translation_id, source_language |
| `portalis_translation_success_rate` | Gauge | Success rate percentage | source_language, target_format |
| `portalis_translations_in_progress` | Gauge | Active translations | source_language |

### Agent Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `portalis_agent_executions_total` | Counter | Agent executions | agent_name, agent_type |
| `portalis_agent_execution_duration_seconds` | Histogram | Agent duration | agent_name, agent_type |
| `portalis_agent_status_total` | Counter | Agent status counts | agent_name, status |
| `portalis_agent_memory_bytes` | Gauge | Agent memory usage | agent_name |
| `portalis_agent_cpu_percent` | Gauge | Agent CPU usage | agent_name |
| `portalis_agents_active` | Gauge | Active agents | agent_type |

### Pipeline Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `portalis_pipeline_phase_duration_seconds` | Histogram | Phase duration | phase_name |
| `portalis_pipeline_phase_status_total` | Counter | Phase status | phase_name, status |
| `portalis_pipeline_duration_seconds` | Histogram | End-to-end duration | pipeline_type |
| `portalis_pipelines_active` | Gauge | Active pipelines | - |
| `portalis_pipeline_queue_depth` | Gauge | Queue depth | priority |

### WASM Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `portalis_wasm_compile_duration_seconds` | Histogram | Compilation time | optimization_level |
| `portalis_wasm_binary_size_bytes` | Histogram | Binary size | optimization_level |
| `portalis_wasm_optimization_level` | Gauge | Optimization level (0-3) | module_id |
| `portalis_wasm_execution_duration_seconds` | Histogram | Execution time | module_name |
| `portalis_wasm_memory_bytes` | Gauge | Memory usage | module_name |
| `portalis_wasm_modules_total` | Counter | Modules created | - |

### Error Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `portalis_errors_total` | Counter | Total errors | category, severity, component |
| `portalis_parse_errors_total` | Counter | Parse errors | error_type, source_language |
| `portalis_translation_errors_total` | Counter | Translation errors | error_type, phase |
| `portalis_compilation_errors_total` | Counter | Compilation errors | error_type, target_format |
| `portalis_runtime_errors_total` | Counter | Runtime errors | error_type, component |
| `portalis_error_recoveries_total` | Counter | Error recoveries | error_category, recovery_method |

### Cache Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `portalis_cache_hits_total` | Counter | Cache hits | cache_name, cache_type |
| `portalis_cache_misses_total` | Counter | Cache misses | cache_name, cache_type |
| `portalis_cache_evictions_total` | Counter | Cache evictions | cache_name, reason |
| `portalis_cache_size_bytes` | Gauge | Cache size | cache_name |
| `portalis_cache_entries` | Gauge | Entry count | cache_name |
| `portalis_cache_hit_rate` | Gauge | Hit rate % | cache_name |

### System Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `portalis_cpu_usage_percent` | Gauge | CPU usage | - |
| `portalis_memory_usage_bytes` | Gauge | Memory usage | - |
| `portalis_disk_usage_bytes` | Gauge | Disk usage | mount_point |
| `portalis_network_io_bytes_total` | Counter | Network I/O | direction, interface |
| `portalis_process_count` | Gauge | Process count | - |
| `portalis_uptime_seconds_total` | Counter | Uptime | - |

## Dashboards

### 1. Portalis Overview Dashboard

**File**: `monitoring/grafana/portalis-overview.json`

**Panels**:
- System Health (success rate, active pipelines, error rate)
- Translation Rate (by source language)
- Success vs Failure rates
- Agent Performance (execution duration by agent)
- Active Agents by Type
- Resource Utilization (CPU, Memory, Cache)
- WASM Performance (compilation time, binary size)

**URL**: `http://grafana:3000/d/portalis-overview`

### 2. Portalis Performance Dashboard

**File**: `monitoring/grafana/portalis-performance.json`

**Panels**:
- Latency Percentiles (P50, P95, P99)
- Latency Over Time
- Translation Throughput (QPS)
- Lines of Code Processed
- CPU and Memory Usage
- GPU Utilization
- WASM Performance Metrics
- Pipeline Phase Performance
- Comparison vs Baselines

**URL**: `http://grafana:3000/d/portalis-performance`

### 3. Portalis Error Analysis Dashboard

**File**: `monitoring/grafana/portalis-errors.json`

**Panels**:
- Error Rate Overview
- Error Rate Over Time
- Errors by Category and Severity
- Translation Failures by Category
- Parse Errors by Type
- Errors by Component
- Agent Failures
- Error Recovery Metrics
- Active Alerts
- 7-Day Error Trend

**URL**: `http://grafana:3000/d/portalis-errors`

## Alert Rules

### Critical Alerts

| Alert | Condition | Duration | Action |
|-------|-----------|----------|--------|
| `PortalisServiceDown` | up{job="portalis"} == 0 | 1m | Page on-call engineer |
| `PortalisVeryLowSuccessRate` | Success rate < 80% | 2m | Page on-call engineer |
| `PortalisHighErrorRate` | Error rate > 10% | 5m | Page on-call engineer |
| `PortalisVerySlowTranslations` | P95 > 10s | 3m | Page on-call engineer |
| `PortalisGPUOutOfMemory` | GPU memory > 95% | 2m | Scale or restart |
| `PortalisWASMCompilationFailures` | WASM failures > 0.1/sec | 5m | Investigate compiler |

### Warning Alerts

| Alert | Condition | Duration | Action |
|-------|-----------|----------|--------|
| `PortalisSlowTranslations` | P95 > 5s | 5m | Investigate performance |
| `PortalisLowSuccessRate` | Success rate < 95% | 5m | Review errors |
| `PortalisHighCPUUsage` | CPU > 90% | 5m | Consider scaling |
| `PortalisAgentFailures` | Agent failure rate > 0.05/s | 5m | Check agent logs |
| `PortalisHighGPUMemory` | GPU memory > 80% | 5m | Monitor closely |

### Info Alerts

| Alert | Condition | Duration | Action |
|-------|-----------|----------|--------|
| `PortalisLowGPUUtilization` | Avg GPU < 30% | 15m | Consider scaling down |
| `PortalisLowCacheHitRate` | Hit rate < 30% | 15m | Optimize caching |

## Health Checks

### Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Liveness probe | 200 OK if service running |
| `GET /health/detailed` | Detailed health | Component status |
| `GET /ready` | Readiness probe | 200 OK if ready for traffic |
| `GET /metrics` | Prometheus metrics | Metrics in text format |

### Health Response Format

```json
{
  "status": "healthy",
  "timestamp": 1234567890,
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "components": [
    {
      "name": "core",
      "status": "healthy",
      "message": null,
      "last_check": 1234567890
    },
    {
      "name": "agents",
      "status": "healthy",
      "message": null,
      "last_check": 1234567890
    }
  ]
}
```

### Readiness Response Format

```json
{
  "ready": true,
  "timestamp": 1234567890,
  "checks": [
    {
      "name": "metrics",
      "ready": true,
      "message": "Metrics system initialized"
    },
    {
      "name": "agents",
      "ready": true,
      "message": "All agents initialized"
    }
  ]
}
```

## Logging

### Log Levels

- **TRACE**: Very detailed debug information
- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **WARN**: Warning messages (non-critical issues)
- **ERROR**: Error messages (failures requiring attention)

### Structured Logging

All logs are emitted in JSON format with consistent fields:

```json
{
  "timestamp": "2025-10-03T23:00:00Z",
  "level": "info",
  "message": "Translation completed successfully",
  "target": "portalis::transpiler",
  "thread_id": "12345",
  "file": "transpiler.rs",
  "line": 123,
  "fields": {
    "translation_id": "abc123",
    "duration_ms": 1234.5,
    "lines_of_code": 500
  }
}
```

### Logging Helpers

```rust
use portalis_core::logging::{AgentLogger, PipelineLogger, TranslationLogger};

// Agent logging
let logger = AgentLogger::new("ingest-agent");
logger.info("Starting parse operation");
logger.error("Parse failed", Some(&error_context));

// Pipeline logging
let logger = PipelineLogger::new("pipeline-123");
logger.phase_start("transpile");
logger.phase_complete("transpile", 1234.5);

// Translation logging
let logger = TranslationLogger::new("trans-456");
logger.start("python", "wasm", 500);
logger.progress("parsing", 25.0);
logger.complete(5000.0, 102400);
```

## Distributed Tracing

### Tracing Setup

```rust
use portalis_core::telemetry::{init_telemetry, TelemetryConfig};

let config = TelemetryConfig {
    service_name: "portalis".to_string(),
    enable_jaeger: true,
    jaeger_endpoint: Some("http://jaeger:14268/api/traces".to_string()),
    ..Default::default()
};

init_telemetry(config)?;
```

### Creating Spans

```rust
use portalis_core::telemetry::{AgentTracer, PipelineTracer};
use tracing::instrument;

// Automatic instrumentation
#[instrument(skip(self))]
async fn execute(&self, input: Input) -> Result<Output> {
    // Function automatically traced
}

// Manual span creation
let tracer = AgentTracer::new("ingest-agent");
let span = tracer.start_span("parse_python");
// ... do work ...
tracer.end_span(&span, true, duration_ms);
```

### Trace Context Propagation

```rust
use portalis_core::telemetry::TraceContext;

// Parent span
let parent_context = TraceContext::new();

// Child span
let child_context = parent_context.child_span();
```

## Troubleshooting

### High Error Rate

1. Check error dashboard for error categories
2. Review logs for specific error messages
3. Check alert annotations for potential causes
4. Verify input data quality
5. Review recent deployments or changes

### Slow Translations

1. Check performance dashboard P95/P99 latency
2. Review GPU utilization (should be >60%)
3. Check agent execution times
4. Verify no resource contention (CPU/Memory)
5. Review complexity of input code

### Low Success Rate

1. Check error dashboard for failure categories
2. Common issues:
   - Parse errors: Invalid Python syntax
   - Translation errors: Unsupported constructs
   - Compilation errors: WASM toolchain issues
3. Review error logs for patterns
4. Check input data quality

### GPU Issues

1. Check GPU memory usage
2. Verify GPU temperature (<85°C)
3. Review CUDA agent logs
4. Check for GPU memory leaks
5. Verify NVIDIA drivers are up to date

### WASM Compilation Failures

1. Check compilation error metrics
2. Review rustc version compatibility
3. Verify wasm-pack installation
4. Check for disk space issues
5. Review generated Rust code quality

## SLO/SLA Definitions

### Service Level Objectives (SLOs)

| Metric | Target | Measurement Period |
|--------|--------|-------------------|
| Availability | 99.9% | Monthly |
| Translation Success Rate | >95% | Daily |
| P95 Latency | <5 seconds | Hourly |
| P99 Latency | <10 seconds | Hourly |
| Throughput | >100 QPS | Per minute |
| Error Rate | <5% | Hourly |

### Service Level Agreements (SLAs)

**Tier 1: Critical Production**
- Availability: 99.9% monthly uptime
- P95 Latency: <5s for 95% of requests
- Success Rate: >95% of all translations

**Tier 2: Development/Testing**
- Availability: 99% monthly uptime
- P95 Latency: <10s for 95% of requests
- Success Rate: >90% of all translations

### Error Budget

- Monthly error budget: 0.1% (43.2 minutes of downtime)
- Budget tracking via `portalis:sla:overall_compliance` metric
- Automatic alerting when budget consumed >75%

## Runbooks

### Service Down

**Alert**: `PortalisServiceDown`

1. Check if service is actually down:
   ```bash
   curl http://portalis:8080/health
   ```

2. Check service logs:
   ```bash
   kubectl logs -l app=portalis --tail=100
   ```

3. Check recent deployments:
   ```bash
   kubectl rollout history deployment/portalis
   ```

4. If crashed, restart:
   ```bash
   kubectl rollout restart deployment/portalis
   ```

5. If persists, rollback:
   ```bash
   kubectl rollout undo deployment/portalis
   ```

### High Error Rate

**Alert**: `PortalisHighErrorRate`

1. Check error dashboard for error breakdown
2. Identify most common error category
3. Review recent code changes
4. Check input data quality
5. If parser errors: Verify Python version compatibility
6. If translation errors: Review unsupported constructs
7. If persistent: Enable debug logging

### Slow Performance

**Alert**: `PortalisSlowTranslations`

1. Check GPU utilization (should be >60%)
2. Review agent execution times
3. Check for resource contention
4. Verify no memory leaks
5. Review code complexity of inputs
6. Consider scaling up GPU resources
7. Check cache hit rate

### GPU Out of Memory

**Alert**: `PortalisGPUOutOfMemory`

1. Identify which GPU is affected
2. Check for memory leaks in CUDA kernels
3. Review batch sizes (reduce if needed)
4. Restart CUDA bridge agent:
   ```bash
   kubectl delete pod -l agent=cuda-bridge
   ```
5. If persists: Scale to additional GPUs
6. Monitor memory usage trend

## Configuration

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'portalis'
    scrape_interval: 15s
    static_configs:
      - targets: ['portalis:8080']
    metrics_path: '/metrics'
```

### Alert Manager Configuration

```yaml
# alertmanager.yml
route:
  receiver: 'team-platform'
  group_by: ['alertname', 'component']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<key>'
  - name: 'slack'
    slack_configs:
      - api_url: '<webhook>'
        channel: '#portalis-alerts'
```

### Environment Variables

```bash
# Logging
export LOG_LEVEL=info
export LOG_JSON=true

# Tracing
export ENABLE_JAEGER=true
export JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# Metrics
export METRICS_PORT=8080
export METRICS_PATH=/metrics
```

## Best Practices

1. **Monitor proactively**: Set up alerts before issues occur
2. **Use dashboards**: Regularly review system dashboards
3. **Structured logging**: Always log with context
4. **Trace critical paths**: Enable tracing for debugging
5. **Test alerts**: Regularly verify alert firing
6. **Document runbooks**: Keep troubleshooting steps updated
7. **Review metrics**: Weekly metric reviews
8. **Optimize performance**: Use profiling data

## Additional Resources

- [Agent Instrumentation Guide](./agent-instrumentation-guide.md)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Phase 3 Monitoring (GPU/DGX)](../dgx-cloud/monitoring/)
