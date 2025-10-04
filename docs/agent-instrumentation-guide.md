# Agent Instrumentation Guide
Week 33 - Phase 4: Monitoring and Observability

## Overview

This guide demonstrates how to add comprehensive tracing, metrics, and logging to all Portalis agents for observability.

## Required Dependencies

Add to each agent's `Cargo.toml`:

```toml
[dependencies]
portalis-core = { path = "../../core" }
tracing = "0.1"
prometheus = "0.13"
```

## Instrumentation Pattern

### 1. Import Required Modules

```rust
use portalis_core::{
    Agent, Result,
    metrics::PortalisMetrics,
    telemetry::{AgentTracer, TraceContext},
    logging::AgentLogger,
    middleware::MetricsMiddleware,
};
use tracing::{info, warn, error, instrument};
use std::sync::Arc;
use std::time::Instant;
```

### 2. Update Agent Structure

```rust
pub struct IngestAgent {
    id: AgentId,
    parser: EnhancedParser,
    // Add observability components
    metrics: Arc<PortalisMetrics>,
    tracer: AgentTracer,
    logger: AgentLogger,
}

impl IngestAgent {
    pub fn new(metrics: Arc<PortalisMetrics>) -> Self {
        Self {
            id: AgentId::new(),
            parser: EnhancedParser::new(),
            metrics: metrics.clone(),
            tracer: AgentTracer::new("ingest-agent"),
            logger: AgentLogger::new("ingest-agent"),
        }
    }
}
```

### 3. Instrument Agent Execute Method

```rust
#[async_trait]
impl Agent for IngestAgent {
    type Input = IngestInput;
    type Output = IngestOutput;

    #[instrument(skip(self, input), fields(agent_id = %self.id))]
    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        // Start timing and tracing
        let start = Instant::now();
        let span = self.tracer.start_span("execute");
        self.logger.info(&format!("Starting ingest for {:?}", input.source_path));

        // Track in-progress
        self.metrics
            .agents
            .agents_active
            .with_label_values(&["ingest"])
            .inc();

        // Execute with error handling
        let result = self.execute_with_instrumentation(input, &span).await;

        // Record metrics
        let duration_ms = start.elapsed().as_millis() as f64;

        match &result {
            Ok(output) => {
                // Success metrics
                self.tracer.end_span(&span, true, duration_ms);
                self.logger.info(&format!("Ingest completed in {:.2}ms", duration_ms));

                self.metrics
                    .agents
                    .agent_status
                    .with_label_values(&["ingest", "success"])
                    .inc();

                self.metrics
                    .agents
                    .agent_duration
                    .with_label_values(&["ingest", "parser"])
                    .observe(duration_ms / 1000.0);
            }
            Err(e) => {
                // Failure metrics
                self.tracer.record_error(&span, &format!("{:?}", e));
                self.logger.error(&format!("Ingest failed: {:?}", e), None);

                self.metrics
                    .agents
                    .agent_status
                    .with_label_values(&["ingest", "failure"])
                    .inc();

                self.metrics
                    .errors
                    .parse_errors
                    .with_label_values(&["parse_failure", "python"])
                    .inc();
            }
        }

        // Decrement active count
        self.metrics
            .agents
            .agents_active
            .with_label_values(&["ingest"])
            .dec();

        result
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::Parse]
    }

    fn agent_id(&self) -> &AgentId {
        &self.id
    }
}
```

### 4. Instrument Internal Methods

```rust
impl IngestAgent {
    #[instrument(skip(self, source))]
    async fn execute_with_instrumentation(
        &self,
        input: IngestInput,
        span: &TraceContext,
    ) -> Result<IngestOutput> {
        // Parse with detailed tracing
        self.logger.debug("Starting Python parsing");
        let parse_start = Instant::now();

        let ast = self.parse_python(&input.source_code)?;

        let parse_duration = parse_start.elapsed().as_secs_f64();
        self.metrics
            .pipeline
            .phase_duration
            .with_label_values(&["parse"])
            .observe(parse_duration);

        info!(
            functions_count = ast.functions.len(),
            classes_count = ast.classes.len(),
            imports_count = ast.imports.len(),
            "Parsing completed"
        );

        // Build metadata
        let metadata = ArtifactMetadata {
            agent_id: self.id.clone(),
            timestamp: chrono::Utc::now(),
            version: "1.0.0".to_string(),
        };

        Ok(IngestOutput { ast, metadata })
    }

    #[instrument(skip(self, source))]
    fn parse_python(&self, source: &str) -> Result<PythonAst> {
        // Detailed parsing with metrics
        let lines = source.lines().count();

        self.metrics
            .translation
            .translation_loc
            .with_label_values(&["python"])
            .observe(lines as f64);

        match self.parser.parse(source) {
            Ok(ast) => {
                self.metrics
                    .cache
                    .cache_hits
                    .with_label_values(&["ast_cache", "parse"])
                    .inc();
                Ok(ast)
            }
            Err(e) if self.fallback_regex => {
                warn!("Parser failed, using fallback: {}", e);
                self.metrics
                    .cache
                    .cache_misses
                    .with_label_values(&["ast_cache", "parse"])
                    .inc();
                self.parse_python_regex(source)
            }
            Err(e) => {
                self.metrics
                    .errors
                    .parse_errors
                    .with_label_values(&["syntax_error", "python"])
                    .inc();
                Err(e)
            }
        }
    }
}
```

## Agent-Specific Instrumentation

### Analysis Agent

```rust
// Key metrics for analysis agent
self.metrics.agents.agent_duration
    .with_label_values(&["analysis", "type_inference"])
    .observe(duration);

self.metrics.translation.translation_complexity
    .with_label_values(&[&translation_id, "python"])
    .set(complexity_score);
```

### Transpiler Agent

```rust
// Translation metrics
let guard = middleware.translation_start("python", "rust");

// ... perform translation ...

guard.success(lines_of_code, complexity_score);

// WASM compilation metrics
self.metrics.wasm.wasm_compile_duration
    .with_label_values(&["O2"])
    .observe(compile_time);
```

### Build Agent

```rust
// Build phase metrics
let phase_guard = middleware.phase_start("build");

// ... perform build ...

phase_guard.success();

self.metrics.wasm.wasm_binary_size_bytes
    .with_label_values(&["O2"])
    .observe(binary_size as f64);
```

### Test Agent

```rust
// Test execution metrics
info!(
    test_count = tests.len(),
    passed = passed_count,
    failed = failed_count,
    "Test execution completed"
);

self.metrics.pipeline.phase_status
    .with_label_values(&["test", if all_passed { "success" } else { "failure" }])
    .inc();
```

## Complete Agent List for Instrumentation

1. **agents/ingest/src/lib.rs** - Parse phase metrics
2. **agents/analysis/src/lib.rs** - Type inference metrics
3. **agents/specgen/src/lib.rs** - Spec generation metrics
4. **agents/transpiler/src/lib.rs** - Translation + WASM metrics
5. **agents/build/src/lib.rs** - Build phase metrics
6. **agents/test/src/lib.rs** - Test execution metrics
7. **agents/packaging/src/lib.rs** - Package creation metrics
8. **agents/nemo-bridge/src/lib.rs** - NeMo inference metrics
9. **agents/cuda-bridge/src/lib.rs** - CUDA acceleration metrics

## Metrics Categories by Agent

| Agent | Key Metrics |
|-------|-------------|
| Ingest | parse_errors, translation_loc, agent_duration |
| Analysis | translation_complexity, agent_duration, type_inference_confidence |
| Specgen | agent_duration, phase_status |
| Transpiler | translation_duration, translation_success/failed, wasm_compile_duration |
| Build | wasm_binary_size, wasm_optimization_level, phase_duration |
| Test | phase_status, test_count, test_failures |
| Packaging | phase_duration, phase_status |
| NeMo Bridge | (uses existing dgx-cloud metrics) |
| CUDA Bridge | gpu_utilization, gpu_memory_used (from Phase 3) |

## Tracing Best Practices

1. **Use #[instrument] macro** for all public async methods
2. **Create child spans** for significant sub-operations
3. **Include context** in span fields (agent_id, operation_name)
4. **Log at appropriate levels**:
   - DEBUG: Detailed internal state
   - INFO: Key operations and milestones
   - WARN: Recoverable errors, fallbacks
   - ERROR: Failures requiring attention

## Testing Instrumentation

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_metrics() {
        let metrics = Arc::new(PortalisMetrics::new().unwrap());
        let agent = IngestAgent::new(metrics.clone());

        let input = IngestInput {
            source_path: PathBuf::from("test.py"),
            source_code: "def hello(): pass".to_string(),
        };

        let result = agent.execute(input).await;
        assert!(result.is_ok());

        // Verify metrics were recorded
        let export = metrics.export().unwrap();
        assert!(export.contains("portalis_agent_executions_total"));
    }
}
```

## Performance Impact

Target: <2% overhead from instrumentation
- Use sampling for high-frequency operations
- Batch metric updates where possible
- Use efficient data structures (Arc, lock-free counters)
- Avoid string allocations in hot paths

## Integration Checklist

- [ ] Add metrics, tracer, logger fields to agent struct
- [ ] Instrument execute() method with timing and tracing
- [ ] Add #[instrument] to key internal methods
- [ ] Record success/failure metrics
- [ ] Log at appropriate levels
- [ ] Add unit tests for metrics
- [ ] Verify <2% performance overhead
- [ ] Update agent documentation
