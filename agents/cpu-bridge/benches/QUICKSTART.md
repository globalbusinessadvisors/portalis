# CPU Bridge Benchmarks - Quick Start Guide

## Run All Benchmarks

```bash
cd /workspace/Portalis
cargo bench --package portalis-cpu-bridge
```

## Run Specific Benchmark Groups

```bash
# Single file translation
cargo bench --package portalis-cpu-bridge -- single_file

# Small batch (10 files)
cargo bench --package portalis-cpu-bridge -- small_batch

# Medium batch (100 files)
cargo bench --package portalis-cpu-bridge -- medium_batch

# Thread scaling (1, 2, 4, 8, 16 cores)
cargo bench --package portalis-cpu-bridge -- thread_scaling

# Workload complexity
cargo bench --package portalis-cpu-bridge -- workload_complexity

# Memory efficiency
cargo bench --package portalis-cpu-bridge -- memory_efficiency

# Realistic mixed workload
cargo bench --package portalis-cpu-bridge -- realistic_workload

# CPU bridge overhead
cargo bench --package portalis-cpu-bridge -- cpu_bridge_overhead
```

## Control Thread Count

```bash
# Run with 4 threads
RAYON_NUM_THREADS=4 cargo bench --package portalis-cpu-bridge

# Run with 16 threads
RAYON_NUM_THREADS=16 cargo bench --package portalis-cpu-bridge
```

## Save and Compare Baselines

```bash
# Save baseline
cargo bench --package portalis-cpu-bridge -- --save-baseline main

# Make changes...

# Compare against baseline
cargo bench --package portalis-cpu-bridge -- --baseline main
```

## Generate HTML Reports

```bash
# Install cargo-criterion
cargo install cargo-criterion

# Generate reports
cargo criterion --package portalis-cpu-bridge

# View in browser
open target/criterion/report/index.html
```

## Interactive Comparison Tool

```bash
cd /workspace/Portalis/agents/cpu-bridge/benches

# Interactive menu
./compare.sh

# Or direct commands:
./compare.sh run current           # Run and save as 'current'
./compare.sh baseline1 baseline2   # Compare two baselines
```

## Expected Results

### Performance Targets

| Benchmark | Target |
|-----------|--------|
| Single file (1KB, 4-core) | < 50ms |
| Small batch (10 files, 16-core) | < 70ms |
| Medium batch (100 files, 16-core) | < 500ms |
| CPU Bridge overhead | < 10% |

### Thread Scaling Efficiency

| Threads | Expected Efficiency |
|---------|-------------------|
| 2 | > 90% |
| 4 | > 85% |
| 8 | > 75% |
| 16 | > 60% |

Efficiency = (Speedup / Thread Count) * 100%

## Troubleshooting

### Benchmarks too slow

```bash
# Reduce sample size
cargo bench --package portalis-cpu-bridge -- --sample-size 10

# Skip specific slow benchmarks
cargo bench --package portalis-cpu-bridge -- --skip medium_batch
```

### Inconsistent results

```bash
# Close other applications
# Disable CPU frequency scaling (Linux):
sudo cpupower frequency-set --governor performance

# Re-run benchmarks
cargo bench --package portalis-cpu-bridge
```

### Out of memory

```bash
# Reduce parallel tasks
RAYON_NUM_THREADS=2 cargo bench --package portalis-cpu-bridge

# Or skip memory-intensive benchmarks
cargo bench --package portalis-cpu-bridge -- --skip memory_efficiency
```

## CI/CD Integration

Add to `.github/workflows/benchmarks.yml`:

```yaml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmarks
        run: cargo bench --package portalis-cpu-bridge -- --save-baseline ci
```

## References

- Full documentation: [README.md](./README.md)
- CPU Architecture Plan: [/workspace/Portalis/plans/CPU_ACCELERATION_ARCHITECTURE.md](/workspace/Portalis/plans/CPU_ACCELERATION_ARCHITECTURE.md)
- Criterion.rs docs: https://bheisler.github.io/criterion.rs/book/
