# Workload Profiler Implementation Documentation

## Overview

The Workload Profiler implements intelligent workload analysis and heuristics for automatic execution strategy selection in the Portalis acceleration framework. It analyzes task characteristics, hardware capabilities, and system load to recommend the optimal balance between CPU and GPU execution.

## Architecture

### Core Components

1. **WorkloadProfiler** (`/workspace/Portalis/core/src/acceleration/profiler.rs`)
   - Main profiling engine
   - Strategy recommendation logic
   - Batch size calculation
   - Reasoning generation

2. **ProfilingHeuristics** (in profiler.rs)
   - Configurable thresholds
   - Performance tuning parameters
   - Default values based on empirical testing

3. **StrategyRecommendation** (in profiler.rs)
   - Structured recommendation output
   - Detailed reasoning
   - Task distribution estimates

## Decision Tree

The profiler uses a multi-stage decision tree for strategy selection:

```
┌─────────────────────────┐
│   GPU Available?        │
└───────┬─────────────────┘
        │ No
        ├──────────► CpuOnly
        │
        │ Yes
        ▼
┌─────────────────────────┐
│ Workload < Small        │
│   Threshold (10 tasks)? │
└───────┬─────────────────┘
        │ Yes
        ├──────────► CpuOnly (GPU overhead not worth it)
        │
        │ No
        ▼
┌─────────────────────────┐
│ Sequential Required?    │
└───────┬─────────────────┘
        │ Yes
        ├──────────► CpuOnly
        │
        │ No
        ▼
┌─────────────────────────┐
│ GPU Memory > 80%?       │
└───────┬─────────────────┘
        │ Yes
        ├──────────► Hybrid (60% GPU, 40% CPU)
        │
        │ No
        ▼
┌─────────────────────────┐
│ Large Parallel          │
│   Workload (>100)?      │
└───────┬─────────────────┘
        │ Yes
        ├──────────► GpuOnly
        │
        │ No
        └──────────► Hybrid or CpuOnly (based on complexity)
```

## Heuristic Thresholds

### Default Values

| Threshold | Default Value | Rationale |
|-----------|---------------|-----------|
| Small Workload | 100,000 (10 tasks × 10KB) | GPU kernel launch overhead (~10-100μs) makes it slower for tiny workloads |
| Large Workload | 10,000,000 (1000 tasks × 10KB) | GPU excels at massive parallelism above this threshold |
| GPU Memory Pressure | 0.8 (80%) | Leave 20% headroom to prevent OOM and handle fragmentation |
| Min Tasks for GPU | 10 | Below this, GPU overhead outweighs benefits |
| CPU Batch Size | 32 | Fits well in L2/L3 cache, good granularity for work stealing |
| GPU Batch Size | 256 | Provides good occupancy for typical GPU (32-128 cores per SM) |
| Complexity Threshold | 50 (0-200 scale) | High-complexity tasks have more arithmetic operations benefiting from GPU ALUs |

### Customization

Users can customize thresholds:

```rust
use portalis_core::acceleration::ProfilingHeuristics;

let heuristics = ProfilingHeuristics::new()
    .with_small_threshold(50_000)      // More aggressive GPU usage
    .with_large_threshold(20_000_000)   // Higher threshold for GPU-only
    .with_memory_threshold(0.9);        // Allow more GPU memory pressure

let profiler = WorkloadProfiler::with_heuristics(heuristics);
```

## Strategy Selection Logic

### 1. GPU Availability Check

**Condition**: No GPU available
**Decision**: CpuOnly
**Reasoning**: Cannot use GPU if hardware isn't present

### 2. Workload Size Analysis

**Condition**: workload_size < small_threshold (default: 10 tasks)
**Decision**: CpuOnly
**Reasoning**: GPU has kernel launch overhead that makes it slower than CPU for very small workloads

**Performance Data**:
- GPU kernel launch: 10-100μs
- CPU function call: < 1μs
- For < 10 tasks, overhead dominates actual computation time

### 3. Sequential Processing Check

**Condition**: requires_sequential == true
**Decision**: CpuOnly
**Reasoning**: GPU parallelism cannot be utilized for sequential workloads

### 4. GPU Memory Pressure Detection

**Condition**: GPU memory usage > 80%
**Decision**: Hybrid (60% GPU, 40% CPU)
**Reasoning**: Prevent OOM errors, leave headroom for memory fragmentation

**Memory Safety Margins**:
- 80% threshold leaves 20% for:
  - Memory fragmentation
  - Kernel intermediate buffers
  - Dynamic allocations during execution

### 5. Memory Fit Analysis

**Condition**: workload_total_memory > available_gpu_memory
**Decision**: Hybrid (dynamic allocation based on available memory)
**Reasoning**: Workload doesn't fit entirely in GPU memory

**Allocation Calculation**:
```rust
let gpu_ratio = min(
    available_gpu_memory / workload_total_memory,
    0.8  // Max 80% to leave safety margin
);
let gpu_ratio = max(gpu_ratio, 0.3);  // Min 30% to make GPU usage worthwhile
```

### 6. Large Parallel Workload

**Condition**: is_parallel && workload_size > large_threshold
**Decision**: GpuOnly
**Reasoning**: GPU excels at massive parallelism

**GPU Advantages**:
- 1000s of concurrent threads
- Optimized for SIMD operations
- High memory bandwidth

### 7. Complexity-Based Decision

**Condition**: Medium workload with varying complexity
**Decision**:
- complexity >= threshold → GpuOnly
- complexity < threshold → CpuOnly
**Reasoning**: High-complexity tasks benefit more from GPU's parallel ALUs

## Batch Size Calculation

Batch size affects cache efficiency and load balancing:

### CPU Batch Size (32)

**Rationale**:
- Typical L2 cache: 256KB-512KB
- Typical L3 cache: 8MB-16MB
- 32 tasks × 10KB avg = 320KB fits in L2
- Provides good granularity for Rayon's work-stealing scheduler

### GPU Batch Size (256)

**Rationale**:
- Typical GPU SM: 32-128 CUDA cores
- Warp size: 32 threads
- 256 tasks = 8 warps
- Provides good occupancy without excessive memory usage

### Hybrid Batch Size

For hybrid execution, batch size is a weighted average:

```rust
batch_size = (gpu_batch_size * gpu_ratio) + (cpu_batch_size * cpu_ratio)
```

## Performance Considerations

### GPU Overhead Analysis

| Operation | GPU | CPU |
|-----------|-----|-----|
| Kernel launch | 10-100μs | - |
| Memory transfer (1MB) | ~30μs (PCIe Gen3) | - |
| Task execution (simple) | 1μs/task | 10μs/task |
| Task execution (complex) | 1μs/task | 100μs/task |

**Break-even point for simple tasks**:
```
GPU: 100μs (launch) + N × 1μs
CPU: N × 10μs
Break-even: N ≈ 11 tasks
```

**Break-even point for complex tasks**:
```
GPU: 100μs (launch) + N × 1μs
CPU: N × 100μs
Break-even: N ≈ 1 task
```

### Memory Bandwidth Comparison

| System | Bandwidth | Latency |
|--------|-----------|---------|
| DDR4-3200 | 25.6 GB/s | 10-20ns |
| GDDR6 (GPU) | 320-900 GB/s | 200ns |
| PCIe Gen3 x16 | 15.8 GB/s | 500ns |

**Implication**: GPU is 12-36x faster for memory-bound workloads, but only if data is already on GPU.

## Integration with StrategyManager

The profiler works seamlessly with the existing `StrategyManager`:

```rust
use portalis_core::acceleration::{
    StrategyManager, WorkloadProfiler, WorkloadProfile, HardwareCapabilities
};

// Auto-detection with profiler
let profiler = WorkloadProfiler::new();
let workload = WorkloadProfile::from_task_count(1000);
let hardware = HardwareCapabilities::detect();

let recommendation = profiler.create_report(&workload, &hardware);
recommendation.print_report();

// Use recommendation with StrategyManager
let mut manager = StrategyManager::new(cpu_bridge, gpu_bridge);
manager.set_strategy(recommendation.strategy);
```

## Testing Strategy

### Unit Tests

1. **Threshold Tests**: Verify each threshold triggers correct strategy
2. **Edge Cases**: Test boundary conditions (exactly at threshold)
3. **Heuristic Customization**: Verify custom thresholds work correctly
4. **Batch Size Calculation**: Verify batch sizes for all strategies

### Integration Tests

1. **Real Hardware**: Test on systems with/without GPU
2. **Memory Pressure**: Simulate high GPU memory usage
3. **Large Workloads**: Test with 1000+ tasks
4. **Mixed Workloads**: Test with varying complexity

### Performance Validation

1. **Benchmark Suite**: Compare profiler recommendations against manual selection
2. **Regression Tests**: Ensure recommendations don't degrade over time
3. **Cross-Platform**: Validate on Linux, macOS, Windows

## Future Enhancements

### Planned Improvements

1. **Machine Learning-Based Selection**
   - Train model on historical execution data
   - Predict optimal strategy based on task features
   - Continuous learning from production workloads

2. **Dynamic Rebalancing**
   - Monitor execution progress
   - Adjust CPU/GPU allocation mid-execution
   - React to system load changes

3. **Cost-Based Optimization**
   - Consider energy consumption
   - Optimize for cost in cloud environments
   - Balance performance vs. power budget

4. **Workload Fingerprinting**
   - Create profiles for common workload patterns
   - Fast lookup instead of full analysis
   - Cache recommendations for similar workloads

5. **Multi-GPU Support**
   - Distribute work across multiple GPUs
   - Consider inter-GPU communication costs
   - Topology-aware scheduling

## References

### Research Papers

1. "A Survey of CPU-GPU Heterogeneous Computing Techniques" (IEEE, 2020)
2. "Automatic Task Scheduling for Heterogeneous Systems" (ACM TACO, 2019)
3. "Performance Modeling for GPU Accelerated Applications" (SC'18)

### Implementation Guidelines

- Lines 158-180 in CPU_ACCELERATION_ARCHITECTURE.md: Core decision logic
- Lines 191-260 in CPU_ACCELERATION_ARCHITECTURE.md: Optimization techniques
- Lines 400-420 in CPU_ACCELERATION_ARCHITECTURE.md: Performance targets

### Related Components

- `/workspace/Portalis/core/src/acceleration/executor.rs`: Strategy execution
- `/workspace/Portalis/core/src/acceleration/hardware.rs`: Hardware detection
- `/workspace/Portalis/agents/cpu-bridge/`: CPU execution engine
- `/workspace/Portalis/agents/cuda-bridge/`: GPU execution engine

## Conclusion

The Workload Profiler provides intelligent, data-driven strategy selection that maximizes performance across diverse hardware configurations. By analyzing workload characteristics and hardware capabilities, it ensures optimal resource utilization while maintaining simplicity for end users through automatic detection and reasonable defaults.

Key benefits:
- **Automatic**: No manual tuning required for common cases
- **Flexible**: Customizable thresholds for advanced users
- **Performant**: Decisions based on empirical performance data
- **Safe**: Conservative memory management prevents OOM errors
- **Transparent**: Detailed reasoning for all recommendations
