# Portalis CPU Bridge

CPU-based parallel processing bridge for the Portalis transpilation platform.

## Overview

The CPU Bridge provides efficient multi-core CPU parallelization as an alternative or complement to GPU acceleration. It leverages Rayon's work-stealing thread pool, optimized data structures, and hardware-aware configuration to maximize throughput across all CPU architectures.

## Features

- **Automatic CPU Detection**: Auto-configures thread count and batch sizes based on hardware
- **Thread Pool Management**: Efficient work-stealing scheduler via Rayon
- **Performance Monitoring**: Real-time metrics tracking and profiling
- **Thread Safety**: Lock-free designs with minimal synchronization overhead
- **Platform Agnostic**: Works on x86_64, ARM64, and other architectures

## Usage

### Basic Example

```rust
use portalis_cpu_bridge::CpuBridge;

// Auto-detect optimal configuration
let bridge = CpuBridge::new();

// Execute parallel translation tasks
let tasks = vec![task1, task2, task3];
let results = bridge.parallel_translate(tasks, |task| {
    // Your translation logic here
    Ok(translated_task)
})?;
```

### Custom Configuration

```rust
use portalis_cpu_bridge::{CpuBridge, CpuConfig};

// Configure custom settings
let config = CpuConfig::builder()
    .num_threads(8)
    .batch_size(64)
    .enable_simd(true)
    .stack_size(4 * 1024 * 1024)
    .build();

let bridge = CpuBridge::with_config(config);
```

### Single Task Execution

```rust
// For low-latency single task execution
let result = bridge.translate_single(task, |t| {
    // Process task
    Ok(processed)
})?;
```

### Performance Metrics

```rust
// Get performance metrics
let metrics = bridge.metrics();
println!("Tasks completed: {}", metrics.tasks_completed());
println!("Avg task time: {:.2}ms", metrics.avg_task_time_ms());
println!("CPU utilization: {:.1}%", metrics.cpu_utilization() * 100.0);
```

## API Reference

### CpuBridge

Core CPU bridge for parallel processing operations.

#### Methods

- `new()` - Creates a new CPU bridge with auto-detected optimal settings
- `with_config(config)` - Creates a new CPU bridge with explicit configuration
- `parallel_translate(tasks, fn)` - Executes parallel translation tasks across multiple CPU cores
- `translate_single(task, fn)` - Executes a single translation task (optimized for latency)
- `metrics()` - Returns a snapshot of current performance metrics
- `config()` - Returns a reference to the CPU configuration
- `num_threads()` - Returns the number of threads in the thread pool

### CpuConfig

Configuration for CPU-based parallel processing.

#### Builder Methods

- `builder()` - Creates a new configuration builder
- `num_threads(n)` - Sets the number of worker threads
- `batch_size(size)` - Sets the task batch size
- `enable_simd(bool)` - Enables or disables SIMD optimizations
- `stack_size(bytes)` - Sets the stack size per thread
- `build()` - Builds the configuration

#### Auto-Detection

- `auto_detect()` - Auto-detects optimal CPU configuration based on hardware

### CpuMetrics

Performance metrics for CPU Bridge operations.

#### Methods

- `tasks_completed()` - Returns the total number of tasks completed
- `avg_task_time_ms()` - Returns the average task time in milliseconds
- `total_time_ms()` - Returns the total execution time in milliseconds
- `cpu_utilization()` - Returns the CPU utilization (0.0 to 1.0)
- `batch_count()` - Returns the total number of batch operations
- `single_task_count()` - Returns the total number of single-task operations
- `success_rate()` - Returns the success rate (0.0 to 1.0)

## Performance Targets

Based on the architecture plan, the CPU Bridge aims to achieve:

| Workload Type | Single Core | 4 Cores | 8 Cores | 16 Cores |
|---------------|-------------|---------|---------|----------|
| Single file (1KB) | 50ms | 45ms | 43ms | 42ms |
| Small batch (10 files) | 500ms | 150ms | 90ms | 70ms |
| Medium batch (100 files) | 5s | 1.5s | 800ms | 500ms |
| Large batch (1000 files) | 50s | 15s | 8s | 5s |

## Architecture

The CPU Bridge is built on top of:

- **Rayon**: Work-stealing thread pool for parallel execution
- **num_cpus**: CPU core detection and system information
- **crossbeam**: Lock-free concurrent data structures
- **parking_lot**: High-performance synchronization primitives

### Design Principles

1. **Thread Safety**: All operations are thread-safe and can be shared via `Arc`
2. **Zero-Cost Abstractions**: Minimal overhead compared to direct Rayon usage (< 10%)
3. **Hardware Awareness**: Auto-detection of CPU cores, cache sizes, and SIMD support
4. **Error Propagation**: Proper error handling with `anyhow::Result`

## Testing

Run the test suite:

```bash
# Unit tests
cargo test -p portalis-cpu-bridge

# Integration tests
cargo test -p portalis-cpu-bridge --test integration_tests

# All tests
cargo test -p portalis-cpu-bridge --all
```

## Benchmarking

Run performance benchmarks:

```bash
# All benchmarks
cargo bench -p portalis-cpu-bridge

# Specific benchmark
cargo bench -p portalis-cpu-bridge --bench cpu_benchmarks

# Generate HTML reports
cargo bench -p portalis-cpu-bridge -- --save-baseline main
```

## Platform Support

| Platform | CPU Support | Status |
|----------|-------------|--------|
| Linux x86_64 | Full + AVX2 | Tier 1 |
| macOS x86_64 | Full + AVX2 | Tier 1 |
| macOS ARM64 | Full + NEON | Tier 1 |
| Windows x86_64 | Full + AVX2 | Tier 1 |
| Linux ARM64 | Full + NEON | Tier 2 |

## License

MIT OR Apache-2.0
