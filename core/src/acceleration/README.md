# Acceleration Module

Intelligent CPU/GPU execution strategy selection and workload profiling.

## Components

### 1. Executor (`executor.rs`)
- `StrategyManager`: Manages execution across CPU/GPU
- `ExecutionStrategy`: CPU-only, GPU-only, Hybrid, Auto
- `HardwareCapabilities`: Hardware detection
- `WorkloadProfile`: Task characteristics

### 2. Profiler (`profiler.rs`)
- `WorkloadProfiler`: Intelligent strategy recommendation
- `ProfilingHeuristics`: Configurable thresholds
- `StrategyRecommendation`: Detailed recommendations with reasoning

### 3. Hardware (`hardware.rs`)
- `HardwareDetector`: Cross-platform hardware detection
- `CpuCapabilities`: CPU detection (cores, SIMD, memory)
- `GpuCapabilities`: CUDA GPU detection

## Quick Start

```rust
use portalis_core::acceleration::*;

// Auto-detect and execute
let profiler = WorkloadProfiler::new();
let workload = WorkloadProfile::from_task_count(1000);
let hardware = HardwareCapabilities::detect();

let recommendation = profiler.create_report(&workload, &hardware);
recommendation.print_report();
```

## Documentation

- **Profiler**: See `PROFILER_DOCUMENTATION.md` for detailed technical docs
- **Implementation**: See `PROFILER_IMPLEMENTATION_SUMMARY.md` for integration details
- **Deliverables**: See `/workspace/Portalis/WORKLOAD_PROFILING_DELIVERABLES.md` for complete overview

## Files

- `executor.rs` - Execution strategy management
- `profiler.rs` - Workload profiling and recommendations  
- `hardware.rs` - Hardware capability detection
- `mod.rs` - Module exports
- `PROFILER_DOCUMENTATION.md` - Technical documentation
- `PROFILER_IMPLEMENTATION_SUMMARY.md` - Implementation guide
- `README.md` - This file
