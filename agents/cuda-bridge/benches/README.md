# CUDA Bridge Benchmark Suite

Comprehensive performance benchmarks for the Portalis CUDA Bridge component using Criterion.rs.

## Overview

This benchmark suite provides extensive performance validation for the CUDA Bridge's AST parsing capabilities, covering:

- Multiple file sizes (small, medium, large)
- Batch processing at scale
- Configuration variations
- Memory usage profiling
- CPU vs GPU comparison
- Scalability analysis

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
cargo bench --bench cuda_benchmarks

# Run specific benchmark group
cargo bench --bench cuda_benchmarks parsing_by_size
cargo bench --bench cuda_benchmarks batch_processing
cargo bench --bench cuda_benchmarks scalability
```

### Benchmark Groups

#### 1. parsing_by_size
Measures parsing performance across different file sizes:
- Small: 645 bytes (~25 lines)
- Medium: 19.6 KB (~500 lines, 25 classes)
- Large: 118.3 KB (~2,000 lines, 100 classes)

**Metrics:** Time, Throughput (MiB/s)

```bash
cargo bench --bench cuda_benchmarks parsing_by_size
```

#### 2. batch_processing
Validates batch parsing efficiency:
- 10, 50, 100 files per batch
- Measures scaling efficiency
- Throughput in elements/second

**Metrics:** Time, Throughput (K elem/s)

```bash
cargo bench --bench cuda_benchmarks batch_processing
```

#### 3. parser_configs
Compares different parser configurations:
- Default: Standard settings
- High Capacity: 5x limits for large codebases
- Low Overhead: Minimal metrics collection

**Metrics:** Time

```bash
cargo bench --bench cuda_benchmarks parser_configs
```

#### 4. memory_usage
Profiles memory allocation and metrics overhead:
- Memory stats retrieval speed
- Parsing with metrics collection
- Overhead analysis

**Metrics:** Time

```bash
cargo bench --bench cuda_benchmarks memory_usage
```

#### 5. cpu_vs_gpu_comparison
Baseline CPU performance for GPU comparison:
- CPU fallback performance
- Establishes baseline for GPU speedup calculations

**Metrics:** Time

```bash
cargo bench --bench cuda_benchmarks cpu_vs_gpu_comparison
```

#### 6. parsing_phases
Analyzes parsing stage breakdown:
- Full parsing pipeline
- Tokenization vs AST construction
- Metrics extraction overhead

**Metrics:** Time

```bash
cargo bench --bench cuda_benchmarks parsing_phases
```

#### 7. scalability
Tests performance with increasing code complexity:
- 10, 50, 100, 200 classes
- Validates O(n) scaling
- Identifies performance degradation

**Metrics:** Time, Throughput (M elem/s)

```bash
cargo bench --bench cuda_benchmarks scalability
```

## Understanding Results

### Sample Output

```
parsing_by_size/medium/19596
                        time:   [31.927 µs 32.464 µs 33.171 µs]
                        thrpt:  [563.39 MiB/s 575.66 MiB/s 585.35 MiB/s]
Found 6 outliers among 100 measurements (6.00%)
  3 (3.00%) high mild
  3 (3.00%) high severe
```

**Interpretation:**
- **time:** Lower bound, estimate, upper bound (95% confidence)
- **thrpt:** Throughput range (inverse of time)
- **outliers:** Statistical outliers (normal if < 10%)

### HTML Reports

Detailed interactive reports are generated at:
```
target/criterion/report/index.html
```

Open in browser to view:
- Performance graphs
- Statistical analysis
- Historical comparisons
- Detailed outlier information

## Benchmark Results Summary

### Current Performance (CPU-only)

| Metric | Value | Status |
|--------|-------|--------|
| Max Throughput | 585 MiB/s | Excellent |
| Batch Efficiency | 869 K elem/s | Good |
| Memory Overhead | 1.59 ns | Negligible |
| Scaling | O(n) linear | Optimal |

### Expected GPU Performance

| Workload | CPU Time | GPU Time (Projected) | Speedup |
|----------|----------|---------------------|---------|
| Small (<1KB) | 1.16 µs | 0.85 µs | 1.4x |
| Medium (~20KB) | 32.46 µs | 3.25 µs | 10.0x |
| Large (~120KB) | 197.24 µs | 15.78 µs | 12.5x |
| Batch (100 files) | 116.58 µs | 9.33 µs | 12.5x |

## Benchmark Configuration

### Environment
- **Criterion version:** 0.5
- **Sample size:** 100 per benchmark
- **Warm-up time:** 3 seconds
- **Measurement time:** 5-6 seconds
- **Optimization:** Release mode

### Statistical Settings
- **Confidence level:** 95%
- **Outlier detection:** Enabled
- **Noise threshold:** Default
- **Resampling:** Bootstrap (100,000 iterations)

## Adding New Benchmarks

### Template

```rust
fn bench_new_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_name");

    let parser = CudaParser::new().unwrap();
    let test_data = "...";

    group.bench_function("test_case", |b| {
        b.iter(|| {
            parser.parse(black_box(&test_data)).unwrap()
        });
    });

    group.finish();
}

// Add to criterion_group!
criterion_group!(
    benches,
    bench_parsing_by_size,
    bench_new_feature,  // Add here
);
```

### Best Practices

1. **Use black_box()** to prevent compiler optimization
2. **Warm up** expensive operations outside iter()
3. **Set throughput** for bandwidth measurements
4. **Group related tests** for easier comparison
5. **Document methodology** in comments

## Continuous Integration

### Running in CI

```bash
# Quick validation (1 sample, 1s warm-up)
cargo bench --bench cuda_benchmarks -- --test

# Full benchmark suite
cargo bench --bench cuda_benchmarks

# Save baseline
cargo bench --bench cuda_benchmarks -- --save-baseline main

# Compare against baseline
cargo bench --bench cuda_benchmarks -- --baseline main
```

### Performance Regression Detection

```bash
# Compare current vs baseline
cargo bench --bench cuda_benchmarks -- --baseline main

# Fail if >5% regression
cargo bench --bench cuda_benchmarks -- --baseline main --threshold 5
```

## Troubleshooting

### Issue: Benchmarks Run Too Long

**Solution:** Reduce sample size or warm-up time
```rust
group.sample_size(10);
group.warm_up_time(std::time::Duration::from_secs(1));
```

### Issue: High Variance in Results

**Causes:**
- Background processes (close unnecessary apps)
- CPU frequency scaling (disable turbo boost)
- Thermal throttling (ensure cooling)

**Solution:**
```bash
# Linux: Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# Check CPU frequency
watch -n1 "cat /proc/cpuinfo | grep MHz"
```

### Issue: Out of Memory

**Solution:** Reduce batch sizes or file sizes in test data generation

### Issue: Criterion Not Found

```bash
# Add to Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "cuda_benchmarks"
harness = false
```

## Interpreting Performance

### Good Performance Indicators
- Throughput > 500 MiB/s for parsing
- Batch scaling efficiency > 95%
- Memory overhead < 5%
- Low variance (< 10% outliers)

### Performance Degradation Signals
- Throughput decreasing with file size
- Batch scaling < 90% efficiency
- Memory overhead > 10%
- High variance (> 20% outliers)

### Optimization Priorities

1. **Critical:** > 20% performance loss
2. **High:** 10-20% performance loss
3. **Medium:** 5-10% performance loss
4. **Low:** < 5% performance loss

## Related Documentation

- **Performance Report:** `../PERFORMANCE_REPORT.md`
- **Phase 3 Week 25:** `/workspace/portalis/PHASE_3_WEEK_25_PROGRESS.md`
- **CUDA Bridge:** `../src/lib.rs`
- **Criterion Docs:** https://bheisler.github.io/criterion.rs/

## Benchmark Maintenance

### Regular Tasks

- **Weekly:** Run benchmarks and check for regressions
- **Before releases:** Full benchmark suite + baseline comparison
- **After optimizations:** Validate improvements
- **Quarterly:** Review and update test data

### Updating Baselines

```bash
# Save new baseline after verified improvements
cargo bench --bench cuda_benchmarks -- --save-baseline v0.2.0

# List baselines
ls -la target/criterion/*/base/
```

## Support

For issues or questions about benchmarks:
1. Check `PERFORMANCE_REPORT.md` for analysis
2. Review Criterion documentation
3. Check GitHub issues
4. Contact: Performance Engineering Team

---

**Last Updated:** 2025-10-03
**Benchmark Version:** 1.0
**Criterion Version:** 0.5
