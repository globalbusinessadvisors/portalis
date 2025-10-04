# Portalis Benchmarking Guide

## How to Benchmark and Validate Performance

This guide explains how to run comprehensive benchmarks and interpret the results for the Portalis translation platform.

---

## 1. Quick Start

### Run All Benchmarks

```bash
# Navigate to project root
cd /workspace/portalis

# Run NeMo benchmarks
python benchmarks/benchmark_nemo.py

# Run end-to-end benchmarks
python benchmarks/benchmark_e2e.py

# Run load tests (requires running server)
locust -f load-tests/locust_scenarios.py --host=http://localhost:8000
```

### View Results

```bash
# Benchmark results are saved to JSON
cat benchmarks/nemo_results.json
cat benchmarks/e2e_results.json

# Load test results in Locust web UI
open http://localhost:8089
```

---

## 2. Benchmark Suite Overview

### 2.1 NeMo Translation Benchmarks

**File:** `benchmarks/benchmark_nemo.py`

**What it tests:**
- Single translation latency (10, 100, 1000 LOC)
- Batch processing throughput
- Optimization impact (baseline vs optimized)
- Scalability (1-1000 concurrent users)

**Run time:** ~5-10 minutes

**Usage:**

```python
from benchmarks.benchmark_nemo import NeMoBenchmark

benchmark = NeMoBenchmark()

# Benchmark single translation
result = await benchmark.benchmark_single_translation(
    code_size=100,  # Lines of code
    iterations=100  # Number of iterations
)

# Print results
benchmark.print_results(result)

# Save to JSON
benchmark.save_results('nemo_results.json')
```

**Expected Output:**

```
==========================================================
Benchmark: single_translation_100loc
==========================================================
Iterations:        100
Mean Latency:      185.23 ms
P50 Latency:       178.45 ms
P95 Latency:       315.67 ms
P99 Latency:       385.12 ms
Throughput:        5.40 req/s
GPU Utilization:   75.0%
Memory Used:       1024.0 MB
Success Rate:      100.0%

SLA Compliance:
  P95 < 500ms: ✓ PASS (315.67ms)
```

### 2.2 End-to-End Pipeline Benchmarks

**File:** `benchmarks/benchmark_e2e.py`

**What it tests:**
- Small function translation (10 LOC)
- Medium function translation (100 LOC)
- Large codebase translation (10K, 100K LOC)
- Cost efficiency analysis

**Run time:** ~15-30 minutes

**Usage:**

```python
from benchmarks.benchmark_e2e import E2EBenchmark

benchmark = E2EBenchmark()

# Benchmark small function
result = await benchmark.benchmark_small_function(iterations=100)

# Benchmark large codebase
result = await benchmark.benchmark_large_codebase(
    total_loc=100000,
    num_files=100
)

# Print formatted result
benchmark.print_result(result)
```

**Expected Output:**

```
======================================================================
Benchmark: large_codebase_100kloc
======================================================================
Total LOC:             100,000
Total Time:            1110.5s
                       (18.5 minutes)
Translation Rate:      90.1 LOC/s
Total Cost:            $18.50
Cost per Translation:  $0.185

Stage Breakdown:
  parse               : 145.320s (13.1%)
  analyze             : 223.150s (20.1%)
  translate_validate  : 556.230s (50.1%)
  compile             : 135.450s (12.2%)
  package             : 50.350s (4.5%)

SLA Compliance:
  under_30_minutes              : ✓ PASS
  cost_under_budget             : ✓ PASS
  success_rate_over_90pct       : ✓ PASS
```

### 2.3 Load Tests

**File:** `load-tests/locust_scenarios.py`

**What it tests:**
- Realistic user behavior simulation
- System behavior under load
- Scalability limits
- Failure modes

**Run time:** Configurable (typically 10-60 minutes)

**Usage:**

```bash
# Start load test with web UI
locust -f load-tests/locust_scenarios.py \
    --host=http://localhost:8000 \
    --users=500 \
    --spawn-rate=10

# Start load test headless
locust -f load-tests/locust_scenarios.py \
    --host=http://localhost:8000 \
    --users=500 \
    --spawn-rate=10 \
    --run-time=30m \
    --headless

# Use specific user class
locust -f load-tests/locust_scenarios.py \
    --host=http://localhost:8000 \
    --user-classes=PowerUser
```

**Load Patterns:**

```python
# Step load: gradual increase
locust -f load-tests/locust_scenarios.py \
    --shape=StepLoadShape

# Spike test: sudden traffic bursts
locust -f load-tests/locust_scenarios.py \
    --shape=SpikeLoadShape

# Stress test: push to failure
locust -f load-tests/locust_scenarios.py \
    --shape=StressTestShape
```

---

## 3. Interpreting Results

### 3.1 Latency Metrics

**P50 (Median):**
- Represents typical user experience
- Target: <200ms for 100 LOC

**P95:**
- 95% of requests faster than this
- Primary SLA metric
- Target: <500ms for 100 LOC

**P99:**
- Catches tail latency
- Important for user satisfaction
- Target: <1s for 100 LOC

**When to investigate:**
- P95 >2x P50: High variance, investigate outliers
- P99 >3x P50: Severe tail latency, check resource contention
- Mean > P50: Skewed distribution, check for slow requests

### 3.2 Throughput Metrics

**QPS (Queries Per Second):**
- Total request rate
- Target: >100 QPS

**Translation Rate (LOC/s):**
- Lines of code processed per second
- Expected: 50-100 LOC/s (depends on complexity)

**GPU Utilization:**
- Percentage of GPU compute used
- Target: 70-90% during load
- <60%: Underutilized (increase batch size or parallelism)
- >95%: Saturated (add more GPUs or reduce load)

### 3.3 Success Rate

**Target:** >99%

**Common Failure Reasons:**
- Timeout (increase batch_timeout_ms)
- Out of memory (enable quantization)
- Model error (check model quality)
- Service unavailable (check health endpoints)

### 3.4 Cost Metrics

**Cost per Translation:**
- Target: <$0.10 per translation
- Typical: $0.008-$0.02 for 100 LOC

**GPU Hours:**
- Total GPU time used
- Monitor for cost optimization

**Cost per LOC:**
- Target: <$0.001 per LOC
- Scales with code complexity

---

## 4. Benchmark Scenarios

### Scenario 1: Single Translation Latency

**Purpose:** Measure end-user experience

```python
result = await benchmark.benchmark_single_translation(
    code_size=100,
    iterations=100
)

# Check SLA compliance
assert result.p95_latency_ms < 500, "P95 latency target missed"
assert result.success_rate > 0.90, "Success rate target missed"
```

### Scenario 2: Batch Throughput

**Purpose:** Measure system capacity

```python
results = await benchmark.benchmark_batch_throughput(
    batch_sizes=[1, 8, 16, 32, 64],
    code_size=100
)

# Find optimal batch size
optimal = max(results, key=lambda r: r.throughput_per_sec)
print(f"Optimal batch size: {optimal.metadata['batch_size']}")
```

### Scenario 3: Scalability Test

**Purpose:** Validate behavior under concurrent load

```python
results = await benchmark.benchmark_scalability(
    concurrent_users=[10, 100, 1000],
    code_size=100
)

# Check graceful degradation
for r in results:
    users = r.metadata['concurrent_users']
    latency = r.p95_latency_ms
    print(f"{users} users: {latency:.0f}ms P95")
```

### Scenario 4: Large Codebase

**Purpose:** Validate enterprise use case

```python
result = await benchmark.benchmark_large_codebase(
    total_loc=100000,
    num_files=100
)

# Check time and cost targets
assert result.total_time_seconds < 1800, "30 minute target missed"
assert result.total_cost_usd < 100, "Cost target exceeded"
```

### Scenario 5: Cost Efficiency

**Purpose:** Validate pricing model

```python
result = await benchmark.benchmark_cost_efficiency()

cost_per_1k = result.cost_per_translation_usd * 1000
assert cost_per_1k < 5.0, "1K translation cost target missed"
```

---

## 5. Load Test Scenarios

### Sustained Load Test

**Purpose:** Validate steady-state performance

```bash
locust -f load-tests/locust_scenarios.py \
    --host=http://localhost:8000 \
    --users=500 \
    --spawn-rate=10 \
    --run-time=1h \
    --headless
```

**Success Criteria:**
- P95 latency < 500ms
- Success rate > 99%
- GPU utilization 70-90%
- No memory leaks

### Spike Test

**Purpose:** Validate burst handling

```bash
locust -f load-tests/locust_scenarios.py \
    --host=http://localhost:8000 \
    --shape=SpikeLoadShape \
    --headless
```

**Success Criteria:**
- Graceful degradation during spike
- Quick recovery after spike
- No cascading failures
- Error rate <5% during spike

### Stress Test

**Purpose:** Find breaking point

```bash
locust -f load-tests/locust_scenarios.py \
    --host=http://localhost:8000 \
    --shape=StressTestShape \
    --headless
```

**Success Criteria:**
- Predictable failure mode
- No crashes
- Automatic recovery
- Clear capacity limits

---

## 6. Continuous Benchmarking

### Automated Benchmarking

```bash
#!/bin/bash
# benchmark_suite.sh

# Run all benchmarks
python benchmarks/benchmark_nemo.py > logs/nemo_bench.log
python benchmarks/benchmark_e2e.py > logs/e2e_bench.log

# Compare with baseline
python scripts/compare_benchmarks.py \
    --baseline=benchmarks/baseline_results.json \
    --current=benchmarks/e2e_results.json \
    --threshold=0.10  # Fail if >10% regression

# Save results with timestamp
cp benchmarks/e2e_results.json \
    "benchmarks/archive/results_$(date +%Y%m%d_%H%M%S).json"
```

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v2

      - name: Run benchmarks
        run: |
          python benchmarks/benchmark_e2e.py

      - name: Compare with baseline
        run: |
          python scripts/compare_benchmarks.py

      - name: Comment PR
        uses: actions/github-script@v6
        with:
          script: |
            const results = require('./benchmarks/e2e_results.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: `## Benchmark Results\n\n${formatResults(results)}`
            });
```

### Regression Detection

```python
# scripts/compare_benchmarks.py
def detect_regression(baseline, current, threshold=0.10):
    """
    Detect performance regressions.

    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results
        threshold: Regression threshold (e.g., 0.10 = 10%)

    Returns:
        True if regression detected
    """
    regressions = []

    for metric in ['p95_latency_ms', 'throughput_per_sec']:
        baseline_val = baseline[metric]
        current_val = current[metric]

        if metric.endswith('latency_ms'):
            # Lower is better for latency
            change = (current_val - baseline_val) / baseline_val
            if change > threshold:
                regressions.append(f"{metric}: {change*100:.1f}% slower")
        else:
            # Higher is better for throughput
            change = (baseline_val - current_val) / baseline_val
            if change > threshold:
                regressions.append(f"{metric}: {change*100:.1f}% slower")

    return regressions
```

---

## 7. Monitoring Integration

### Export Metrics to Prometheus

```python
from prometheus_client import start_http_server, Histogram, Counter

# Define metrics
translation_latency = Histogram(
    'portalis_translation_duration_seconds',
    'Translation latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

translation_requests = Counter(
    'portalis_translation_requests_total',
    'Total translation requests',
    ['method', 'status']
)

# Start metrics server
start_http_server(9090)

# Record metrics during benchmark
with translation_latency.time():
    result = await translate_code(code)

translation_requests.labels(method='nemo', status='success').inc()
```

### View in Grafana

```bash
# Import benchmark results into Grafana
python scripts/export_to_grafana.py \
    --results=benchmarks/e2e_results.json \
    --dashboard=performance_dashboard
```

---

## 8. Benchmark Report Template

```markdown
# Benchmark Report: [Date]

## Environment
- Platform: NVIDIA DGX A100
- GPUs: 8x A100 (80GB)
- CUDA Version: 12.1
- Portalis Version: 1.0.0

## Results Summary

| Benchmark | Target | Actual | Status |
|-----------|--------|--------|--------|
| P95 Latency (100 LOC) | <500ms | 315ms | ✅ PASS |
| Throughput | >100 QPS | 142 QPS | ✅ PASS |
| Success Rate | >99% | 99.4% | ✅ PASS |
| Cost/Translation | <$0.10 | $0.08 | ✅ PASS |

## Detailed Results

### Single Translation Latency
- P50: 185ms
- P95: 315ms
- P99: 385ms

### Batch Throughput
- Optimal batch size: 32
- Max QPS: 325
- GPU utilization: 85%

### Large Codebase
- 100K LOC: 18.5 minutes
- Translation rate: 90 LOC/s
- Total cost: $18.50

## Optimizations Applied
- ✅ TensorRT FP16
- ✅ INT8 Quantization
- ✅ Flash Attention
- ✅ Dynamic Batching
- ✅ Stage Fusion

## Recommendations
1. All targets met, ready for production
2. Consider INT4 quantization for cost reduction
3. Monitor cache hit rate for optimization opportunities

## Regression Analysis
- No regressions detected vs baseline
- 15% improvement in P95 latency
- 22% improvement in throughput
```

---

## 9. Troubleshooting Benchmarks

### Problem: Inconsistent Results

**Symptoms:** High variance between runs

**Solutions:**
1. Warm up model before benchmarking
2. Run more iterations (100+)
3. Use fixed random seed
4. Disable background processes

### Problem: OOM During Benchmarks

**Symptoms:** CUDA out of memory

**Solutions:**
1. Reduce batch size
2. Enable quantization
3. Clear GPU cache between runs
4. Use smaller test data

### Problem: Slow Benchmarks

**Symptoms:** Benchmarks take too long

**Solutions:**
1. Reduce iterations
2. Use smaller code samples
3. Run subset of benchmarks
4. Parallelize benchmarks

### Problem: Results Don't Match SLA

**Symptoms:** Failing SLA checks

**Solutions:**
1. Review optimization configuration
2. Check GPU utilization
3. Verify model is optimized
4. Check for resource contention

---

## 10. Best Practices

### Before Benchmarking

✅ Warm up system (run 10-20 warm-up iterations)
✅ Close unnecessary applications
✅ Pin GPU to exclusive mode
✅ Clear GPU cache
✅ Use consistent test data

### During Benchmarking

✅ Monitor GPU utilization
✅ Check for thermal throttling
✅ Watch for OOM errors
✅ Log all parameters
✅ Save intermediate results

### After Benchmarking

✅ Analyze results thoroughly
✅ Compare with baseline
✅ Document optimizations
✅ Archive results
✅ Update regression thresholds

---

## 11. Quick Reference

### Run Specific Benchmark

```bash
# Single translation latency
python -c "import asyncio; from benchmarks.benchmark_nemo import *; asyncio.run(NeMoBenchmark().benchmark_single_translation(100, 100))"

# Batch throughput
python -c "import asyncio; from benchmarks.benchmark_nemo import *; asyncio.run(NeMoBenchmark().benchmark_batch_throughput([8, 16, 32]))"

# Large codebase
python -c "import asyncio; from benchmarks.benchmark_e2e import *; asyncio.run(E2EBenchmark().benchmark_large_codebase(100000, 100))"
```

### Analyze Results

```bash
# View latest results
cat benchmarks/e2e_results.json | jq '.results[] | {name, p95_latency_ms, throughput_per_sec}'

# Compare two results
diff <(jq -S . baseline.json) <(jq -S . current.json)

# Plot latency distribution
python scripts/plot_results.py benchmarks/nemo_results.json --metric=latency
```

---

**Last Updated:** 2025-10-03
**Version:** 1.0
