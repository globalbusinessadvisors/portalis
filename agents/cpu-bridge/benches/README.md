# CPU Bridge Performance Benchmarks

Comprehensive benchmark suite for validating CPU Bridge performance against architectural targets.

## Performance Targets

Based on the CPU Acceleration Architecture plan (lines 403-409):

| Workload Type | Single Core | 4 Cores | 8 Cores | 16 Cores |
|---------------|-------------|---------|---------|----------|
| Single file (1KB) | 50ms | 45ms | 43ms | 42ms |
| Small batch (10 files) | 500ms | 150ms | 90ms | 70ms |
| Medium batch (100 files) | 5s | 1.5s | 800ms | 500ms |
| Large batch (1000 files) | 50s | 15s | 8s | 5s |

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks with default settings
cargo bench --package portalis-cpu-bridge

# Run specific benchmark group
cargo bench --package portalis-cpu-bridge --bench cpu_benchmarks -- single_file

# Run with HTML reports
cargo bench --package portalis-cpu-bridge -- --save-baseline main

# Compare against baseline
cargo bench --package portalis-cpu-bridge -- --baseline main
```

### Advanced Usage

```bash
# Run with specific thread count
RAYON_NUM_THREADS=4 cargo bench --package portalis-cpu-bridge

# Run with verbose output
cargo bench --package portalis-cpu-bridge -- --verbose

# Run specific test pattern
cargo bench --package portalis-cpu-bridge -- thread_scaling

# Save results for comparison
cargo bench --package portalis-cpu-bridge -- --save-baseline v1.0

# Compare two baselines
critcmp v1.0 v1.1
```

## Benchmark Suite Overview

### 1. Single File Translation (`bench_single_file_translation`)

**Purpose**: Validate single-file translation performance

**Test Cases**:
- Small file (1KB): Basic functions and classes
- Medium file (5KB): Multiple classes with methods
- Large file (20KB): Complex modules with async code

**Target**: < 50ms for 1KB file on 4-core CPU

**Metrics**:
- Throughput (bytes/sec)
- Latency (ms per file)
- CPU utilization

### 2. Small Batch Processing (`bench_small_batch`)

**Purpose**: Measure parallel efficiency for small workloads

**Test Cases**:
- 10 files, sequential processing
- 10 files, parallel processing (default threads)

**Target**: 500ms → 70ms on 16-core CPU

**Metrics**:
- Sequential vs parallel speedup
- Thread utilization
- Overhead measurement

### 3. Medium Batch Processing (`bench_medium_batch`)

**Purpose**: Measure parallel scaling for typical projects

**Test Cases**:
- 100 files, sequential processing
- 100 files, parallel processing

**Target**: 5s → 500ms on 16-core CPU

**Metrics**:
- Scalability factor
- Memory usage per task
- CPU saturation

### 4. Thread Scaling Tests (`bench_thread_scaling`)

**Purpose**: Validate near-linear scaling with CPU cores

**Test Cases**:
- 1, 2, 4, 8, 16 threads processing 100 files

**Target**: Near-linear scaling (90%+ efficiency)

**Metrics**:
- Speedup vs thread count
- Parallel efficiency percentage
- Amdahl's law validation

**Analysis**:
```
Speedup = T(1) / T(N)
Efficiency = Speedup / N * 100%

Ideal: Efficiency > 90%
Good: Efficiency > 80%
Poor: Efficiency < 70%
```

### 5. Workload Complexity Scaling (`bench_workload_complexity`)

**Purpose**: Measure impact of file size on performance

**Test Cases**:
- Single file: small, medium, large
- Batch of 10: small, medium, large

**Metrics**:
- Bytes per second throughput
- Scaling with file size
- Memory per byte processed

### 6. Memory Efficiency (`bench_memory_efficiency`)

**Purpose**: Validate < 50MB per concurrent task target

**Test Cases**:
- Batch sizes: 10, 50, 100, 500 files

**Metrics**:
- Memory per task
- Memory growth rate
- Allocation patterns

### 7. Cache Locality (`bench_cache_locality`)

**Purpose**: Measure data structure efficiency

**Test Cases**:
- Sequential access patterns
- Parallel overhead measurement

**Metrics**:
- Cache hit rate (inferred)
- Memory bandwidth utilization
- SIMD opportunities

### 8. Realistic Workload (`bench_realistic_workload`)

**Purpose**: Simulate real-world project mix

**Test Cases**:
- 100 files: 70% small, 25% medium, 5% large

**Metrics**:
- Real-world throughput
- Mixed workload efficiency
- Resource utilization

### 9. CPU Baseline Comparison (`bench_cpu_baseline_comparison`)

**Purpose**: Measure CPU Bridge overhead

**Test Cases**:
- Raw translation (no wrapper)
- With CPU Bridge (measure overhead)
- Batch overhead

**Target**: < 10% overhead vs direct implementation

## Interpreting Results

### Criterion Output Format

```
single_file_translation/small_1kb
                        time:   [42.123 ms 42.456 ms 42.789 ms]
                        thrpt:  [24.001 KiB/s 24.201 KiB/s 24.401 KiB/s]
```

**Key Metrics**:
- `time`: [lower_bound mean upper_bound] - Lower is better
- `thrpt`: Throughput - Higher is better
- Criterion automatically detects outliers and provides confidence intervals

### Expected Results

**Single File (1KB)**:
```
small_1kb (4 cores):     ~45ms
medium_5kb (4 cores):    ~150ms
large_20kb (4 cores):    ~600ms
```

**Thread Scaling (100 files)**:
```
1 thread:   ~5000ms  (baseline)
2 threads:  ~2600ms  (1.92x speedup, 96% efficiency)
4 threads:  ~1400ms  (3.57x speedup, 89% efficiency)
8 threads:  ~800ms   (6.25x speedup, 78% efficiency)
16 threads: ~500ms   (10.0x speedup, 62% efficiency)
```

Note: Efficiency decreases with more threads due to:
- Synchronization overhead
- Cache coherency traffic
- Memory bandwidth saturation
- Amdahl's law (sequential portions)

### Performance Regression Detection

```bash
# Save baseline before changes
cargo bench --package portalis-cpu-bridge -- --save-baseline before

# Make changes to code...

# Compare after changes
cargo bench --package portalis-cpu-bridge -- --baseline before

# Criterion will highlight regressions:
# - Green: Improvement
# - Red: Regression (>5% slower)
# - Yellow: No significant change
```

## Benchmark Comparison Tools

### Using `critcmp`

Install:
```bash
cargo install critcmp
```

Usage:
```bash
# Compare two benchmark runs
critcmp before after

# Show only regressions
critcmp --threshold 5 before after

# Export to CSV
critcmp --export before after > comparison.csv
```

### Using `cargo-criterion`

```bash
cargo install cargo-criterion

# Run with HTML reports
cargo criterion --package portalis-cpu-bridge

# Open reports
open target/criterion/report/index.html
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Benchmark

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run benchmarks
        run: |
          cargo bench --package portalis-cpu-bridge -- --save-baseline pr

      - name: Compare with main
        run: |
          git fetch origin main
          git checkout origin/main
          cargo bench --package portalis-cpu-bridge -- --save-baseline main
          git checkout -
          critcmp main pr
```

## Profiling Integration

### Using `flamegraph`

```bash
# Install
cargo install flamegraph

# Profile single benchmark
cargo flamegraph --bench cpu_benchmarks -- --bench single_file

# View SVG
firefox flamegraph.svg
```

### Using `perf`

```bash
# Linux only
cargo bench --package portalis-cpu-bridge -- --profile-time=5

# Analyze with perf
perf record -g target/release/deps/cpu_benchmarks-* --bench
perf report
```

### Using `valgrind` (Cache Analysis)

```bash
# Install cachegrind
sudo apt install valgrind

# Run with cache profiling
valgrind --tool=cachegrind target/release/deps/cpu_benchmarks-*

# Analyze results
cg_annotate cachegrind.out.*
```

## Troubleshooting

### Benchmark Takes Too Long

```bash
# Reduce sample size
cargo bench --package portalis-cpu-bridge -- --sample-size 10

# Skip warm-up
cargo bench --package portalis-cpu-bridge -- --warm-up-time 1
```

### Inconsistent Results

```bash
# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Isolate CPUs (Linux)
sudo cset shield --cpu 0-3

# Run on isolated CPUs
sudo cset shield --exec cargo bench --package portalis-cpu-bridge
```

### Out of Memory

```bash
# Reduce parallel jobs
RAYON_NUM_THREADS=2 cargo bench --package portalis-cpu-bridge

# Increase stack size
RUST_MIN_STACK=8388608 cargo bench --package portalis-cpu-bridge
```

## Contributing

When adding new benchmarks:

1. **Follow naming convention**: `bench_<category>_<specific_test>`
2. **Set appropriate sample size**: Use `group.sample_size(N)` for long tests
3. **Add throughput metrics**: Use `Throughput::Bytes` or `Throughput::Elements`
4. **Document expected results**: Add comments with target performance
5. **Use `black_box`**: Prevent compiler optimizations: `black_box(value)`

Example:
```rust
fn bench_new_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_feature");
    group.sample_size(50);
    group.throughput(Throughput::Elements(100));

    group.bench_function("baseline", |b| {
        b.iter(|| {
            // Your benchmark code
            black_box(compute_result())
        });
    });

    group.finish();
}
```

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [CPU Acceleration Architecture Plan](/workspace/Portalis/plans/CPU_ACCELERATION_ARCHITECTURE.md)
- [Rayon Documentation](https://docs.rs/rayon/)
- [Performance Optimization Guide](https://nnethercote.github.io/perf-book/)

## Benchmark Results Archive

Store baseline results in `target/criterion/`:
```
target/criterion/
├── single_file_translation/
│   ├── small_1kb/
│   │   ├── base/
│   │   └── change/
│   └── report/
└── thread_scaling/
    └── report/
```

Commit baseline summaries to track performance over time:
```bash
# Save summary
cargo bench --package portalis-cpu-bridge > benchmarks/results-$(date +%Y%m%d).txt

# Commit to repo
git add benchmarks/
git commit -m "Benchmark results for release v1.0"
```
