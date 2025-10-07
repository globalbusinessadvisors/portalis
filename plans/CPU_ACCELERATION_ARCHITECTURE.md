# CPU Acceleration Architecture Plan

**Date:** 2025-10-07
**Version:** 1.0
**Status:** Planning

## Executive Summary

This plan outlines the architecture and implementation strategy for adding comprehensive CPU-based parallel processing capabilities to the Portalis platform. While the platform currently supports GPU acceleration through CUDA, many users don't have access to GPUs or may be running in CPU-only environments (cloud containers, edge devices, CI/CD pipelines). This plan ensures Portalis provides excellent performance across all hardware configurations.

## Current State Analysis

### Existing GPU Capabilities
- **CUDA Bridge** (`agents/cuda-bridge`): GPU acceleration for parallel processing
- **NeMo Bridge** (`agents/nemo-bridge`): NVIDIA NeMo LLM integration
- GPU-optimized transpilation pipelines

### Gaps for CPU Users
1. **No CPU fallback** when GPU is unavailable
2. **No CPU-optimized parallel processing** for translation workloads
3. **No multi-core utilization** in non-GPU environments
4. **Performance degradation** when CUDA is disabled
5. **Limited deployment flexibility** (requires GPU infrastructure)

## Objectives

### Primary Goals
1. **Universal Compatibility**: Run efficiently on any CPU architecture (x86_64, ARM, RISC-V)
2. **Multi-core Parallelism**: Utilize all available CPU cores for translation tasks
3. **Graceful Degradation**: Automatically fall back from GPU → CPU when needed
4. **Optimized Performance**: Achieve near-linear scaling with CPU core count
5. **Platform Parity**: Maintain feature parity between GPU and CPU execution paths

### Performance Targets
- **Single File Translation**: < 100ms on modern 4-core CPU
- **Batch Processing**: Linear scaling up to available CPU cores
- **Memory Efficiency**: < 50MB per concurrent translation task
- **CPU Utilization**: > 90% on multi-file batch operations

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Portalis Platform                             │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Acceleration Strategy Manager                       │ │
│  │  - Auto-detect hardware capabilities                       │ │
│  │  - Select optimal execution strategy                       │ │
│  │  - Load balance across resources                           │ │
│  └────────────┬───────────────────────────┬───────────────────┘ │
│               │                           │                      │
│    ┌──────────▼──────────┐    ┌──────────▼──────────┐          │
│    │   GPU Acceleration  │    │   CPU Acceleration  │          │
│    │   (CUDA Bridge)     │    │   (NEW: CPU Bridge) │          │
│    └─────────────────────┘    └─────────────────────┘          │
│                                                                   │
│  Execution Paths:                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. GPU-Only:     CUDA Kernels (existing)                 │  │
│  │ 2. CPU-Only:     Rayon Thread Pool (NEW)                 │  │
│  │ 3. Hybrid:       GPU for heavy tasks, CPU for light      │  │
│  │ 4. Distributed:  CPU cluster coordination (future)       │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. CPU Bridge Agent (`agents/cpu-bridge`)

**Purpose**: Provide CPU-based parallel processing equivalent to CUDA bridge

**Core Responsibilities**:
- Thread pool management using Rayon
- Work-stealing scheduler for load balancing
- SIMD vectorization for data-parallel operations
- Memory-efficient task batching
- Performance monitoring and profiling

**API Design**:
```rust
pub struct CpuBridge {
    thread_pool: rayon::ThreadPool,
    config: CpuConfig,
    metrics: Arc<RwLock<CpuMetrics>>,
}

impl CpuBridge {
    /// Create CPU bridge with auto-detected optimal settings
    pub fn new() -> Self;

    /// Create with explicit configuration
    pub fn with_config(config: CpuConfig) -> Self;

    /// Execute parallel translation tasks
    pub fn parallel_translate(
        &self,
        tasks: Vec<TranslationTask>
    ) -> Result<Vec<TranslationOutput>>;

    /// Execute single task (optimized for latency)
    pub fn translate_single(
        &self,
        task: TranslationTask
    ) -> Result<TranslationOutput>;

    /// Get performance metrics
    pub fn metrics(&self) -> CpuMetrics;
}

pub struct CpuConfig {
    /// Number of worker threads (default: num_cpus)
    pub num_threads: usize,

    /// Task batch size for optimal cache usage
    pub batch_size: usize,

    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Stack size per thread
    pub stack_size: usize,
}
```

### 2. Acceleration Strategy Manager

**Purpose**: Intelligently select and coordinate between GPU and CPU execution

**Decision Logic**:
```rust
pub enum ExecutionStrategy {
    /// Use GPU exclusively (CUDA available, large workload)
    GpuOnly,

    /// Use CPU exclusively (no GPU, or small workload)
    CpuOnly,

    /// Use both GPU and CPU (hybrid workload)
    Hybrid {
        gpu_allocation: f32,  // 0.0 to 1.0
        cpu_allocation: f32,
    },

    /// Automatic selection based on runtime conditions
    Auto,
}

pub struct StrategyManager {
    gpu_bridge: Option<CudaBridge>,
    cpu_bridge: CpuBridge,
    strategy: ExecutionStrategy,
}

impl StrategyManager {
    /// Auto-detect optimal strategy
    pub fn detect_strategy(&mut self) -> ExecutionStrategy {
        // 1. Check GPU availability
        if !self.has_gpu() {
            return ExecutionStrategy::CpuOnly;
        }

        // 2. Check workload size
        if workload_size < SMALL_THRESHOLD {
            // CPU faster for small tasks (no GPU overhead)
            return ExecutionStrategy::CpuOnly;
        }

        // 3. Check GPU memory availability
        if gpu_memory_pressure() > 0.8 {
            return ExecutionStrategy::Hybrid {
                gpu_allocation: 0.6,
                cpu_allocation: 0.4,
            };
        }

        // 4. Use GPU for large parallel workloads
        ExecutionStrategy::GpuOnly
    }

    /// Execute with selected strategy
    pub fn execute(
        &self,
        tasks: Vec<TranslationTask>
    ) -> Result<Vec<TranslationOutput>>;
}
```

### 3. CPU Optimization Techniques

#### A. Rayon Thread Pool
```rust
use rayon::prelude::*;

impl CpuBridge {
    /// Parallel batch translation using work-stealing
    fn parallel_translate_batch(
        &self,
        files: Vec<PathBuf>
    ) -> Result<Vec<TranslationOutput>> {
        files
            .par_iter()
            .map(|file| self.translate_file(file))
            .collect()
    }
}
```

#### B. SIMD Vectorization
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl CpuBridge {
    /// Vectorized AST node processing
    #[target_feature(enable = "avx2")]
    unsafe fn process_nodes_simd(
        &self,
        nodes: &[AstNode]
    ) -> Vec<ProcessedNode> {
        // Use SIMD instructions for parallel node analysis
        // Process 8 nodes at once with AVX2
    }
}
```

#### C. Cache-Friendly Data Structures
```rust
/// Structure of Arrays (SoA) for better cache locality
pub struct TranslationBatch {
    /// All source code strings together
    sources: Vec<String>,

    /// All file paths together
    paths: Vec<PathBuf>,

    /// All configurations together
    configs: Vec<TranslationConfig>,
}
```

#### D. Async I/O with Tokio
```rust
impl CpuBridge {
    /// Non-blocking file I/O during translation
    async fn translate_files_async(
        &self,
        files: Vec<PathBuf>
    ) -> Result<Vec<TranslationOutput>> {
        // Read files asynchronously
        let contents = join_all(
            files.iter().map(|f| tokio::fs::read_to_string(f))
        ).await;

        // Translate in parallel on CPU
        self.parallel_translate(contents)
    }
}
```

## Implementation Phases

### Phase 1: CPU Bridge Foundation (Week 1-2)

**Deliverables**:
- [ ] Create `agents/cpu-bridge` crate
- [ ] Implement basic Rayon thread pool wrapper
- [ ] Add CPU configuration and auto-detection
- [ ] Integrate with core translation pipeline
- [ ] Basic benchmarking suite

**Files to Create**:
```
agents/cpu-bridge/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── thread_pool.rs      # Rayon integration
│   ├── config.rs           # CPU configuration
│   ├── metrics.rs          # Performance tracking
│   └── simd.rs             # SIMD optimizations
├── benches/
│   └── cpu_benchmarks.rs
└── tests/
    └── integration_tests.rs
```

**Dependencies**:
```toml
[dependencies]
rayon = "1.8"
num_cpus = "1.16"
crossbeam = "0.8"
parking_lot = "0.12"
```

### Phase 2: Strategy Manager (Week 3)

**Deliverables**:
- [ ] Implement `StrategyManager` in `portalis-core`
- [ ] Hardware capability detection
- [ ] Workload profiling and heuristics
- [ ] Auto-selection logic
- [ ] Graceful GPU → CPU fallback

**Integration Points**:
```rust
// In portalis-core/src/lib.rs
pub struct AccelerationConfig {
    pub strategy: ExecutionStrategy,
    pub cpu_config: CpuConfig,
    pub gpu_config: Option<CudaConfig>,
}

// Update TranspilerAgent
impl TranspilerAgent {
    pub fn with_acceleration(
        config: AccelerationConfig
    ) -> Self {
        // Auto-configure based on hardware
    }
}
```

### Phase 3: SIMD Optimizations (Week 4)

**Deliverables**:
- [ ] AVX2/SSE4 vectorized operations for x86_64
- [ ] NEON optimizations for ARM
- [ ] Runtime CPU feature detection
- [ ] Fallback implementations for unsupported CPUs
- [ ] Performance benchmarks showing speedup

**Target Operations**:
- AST node traversal and analysis
- String pattern matching (imports, identifiers)
- Type inference batch operations
- Parallel syntax validation

### Phase 4: Memory & Cache Optimizations (Week 5)

**Deliverables**:
- [ ] Implement cache-friendly data structures
- [ ] Memory pool for AST nodes
- [ ] Reduce allocations in hot paths
- [ ] Profile-guided optimization
- [ ] Memory usage benchmarks

**Techniques**:
- Structure of Arrays (SoA) layouts
- Arena allocation for AST nodes
- Object pooling for frequent allocations
- Cache line alignment for critical structures

### Phase 5: CLI & Configuration (Week 6)

**Deliverables**:
- [ ] CLI flags for execution strategy
- [ ] Configuration file support
- [ ] Performance profiling mode
- [ ] Benchmark comparison tools
- [ ] Documentation and examples

**CLI Interface**:
```bash
# Auto-detect (default)
portalis convert script.py

# Force CPU-only
portalis convert script.py --cpu-only

# Force GPU-only
portalis convert script.py --gpu-only

# Hybrid with custom allocation
portalis convert script.py --hybrid --gpu-ratio 0.7

# Show performance stats
portalis convert script.py --profile

# Benchmark mode
portalis benchmark --compare-strategies
```

### Phase 6: Testing & Validation (Week 7)

**Deliverables**:
- [ ] Unit tests for CPU bridge
- [ ] Integration tests for strategy manager
- [ ] Performance regression tests
- [ ] Cross-platform validation (Linux, macOS, Windows)
- [ ] ARM compatibility testing

**Test Coverage Goals**:
- Unit tests: > 90%
- Integration tests: All execution paths
- Benchmark suite: CPU vs GPU vs Hybrid

## Performance Benchmarks

### Target Metrics

| Workload Type | Single Core | 4 Cores | 8 Cores | 16 Cores |
|---------------|-------------|---------|---------|----------|
| Single file (1KB) | 50ms | 45ms | 43ms | 42ms |
| Small batch (10 files) | 500ms | 150ms | 90ms | 70ms |
| Medium batch (100 files) | 5s | 1.5s | 800ms | 500ms |
| Large batch (1000 files) | 50s | 15s | 8s | 5s |

### Comparison vs GPU

| Workload | CPU (16-core) | GPU (CUDA) | Speedup |
|----------|---------------|------------|---------|
| Single file | 42ms | 25ms | 1.68x |
| Small batch | 70ms | 35ms | 2.0x |
| Medium batch | 500ms | 150ms | 3.33x |
| Large batch | 5s | 800ms | 6.25x |

**Interpretation**: GPUs excel at large parallel workloads, but CPU is competitive for small tasks and more universally available.

## Configuration Examples

### Default Auto-Detection
```toml
# portalis.toml
[acceleration]
strategy = "auto"

[acceleration.cpu]
# Auto-detect optimal settings
auto_configure = true

[acceleration.gpu]
# Use GPU if available
enabled = true
```

### CPU-Only Production Environment
```toml
[acceleration]
strategy = "cpu-only"

[acceleration.cpu]
num_threads = 16  # Explicit thread count
batch_size = 32   # Optimize for throughput
enable_simd = true
```

### Development Laptop (Hybrid)
```toml
[acceleration]
strategy = "hybrid"

[acceleration.hybrid]
gpu_allocation = 0.6
cpu_allocation = 0.4

[acceleration.cpu]
num_threads = 4
enable_simd = true
```

## Platform Compatibility

### Supported Platforms

| Platform | CPU Support | GPU Support | Status |
|----------|-------------|-------------|--------|
| Linux x86_64 | ✅ Full | ✅ CUDA | Tier 1 |
| macOS x86_64 | ✅ Full | ❌ N/A | Tier 1 |
| macOS ARM64 | ✅ Full + NEON | ❌ N/A | Tier 1 |
| Windows x86_64 | ✅ Full | ✅ CUDA | Tier 1 |
| Linux ARM64 | ✅ Full + NEON | ⚠️  Limited | Tier 2 |
| WebAssembly | ⚠️  Limited | ❌ N/A | Tier 2 |

### Minimum Requirements

**CPU-Only Mode**:
- 2+ CPU cores recommended
- 4GB RAM minimum
- x86_64, ARM64, or RISC-V architecture

**Hybrid Mode**:
- 4+ CPU cores recommended
- 8GB RAM minimum
- CUDA-capable GPU (optional)

## Monitoring & Observability

### Metrics to Track

```rust
pub struct CpuMetrics {
    /// Total tasks executed
    pub tasks_completed: u64,

    /// Average execution time per task
    pub avg_task_time_ms: f64,

    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,

    /// Memory usage in bytes
    pub memory_usage: usize,

    /// Thread pool statistics
    pub active_threads: usize,
    pub idle_threads: usize,

    /// Cache hit rate for optimizations
    pub cache_hit_rate: f64,
}
```

### Prometheus Integration

```rust
// Export metrics for monitoring
impl CpuBridge {
    pub fn export_prometheus_metrics(&self) -> String {
        format!(
            "portalis_cpu_tasks_total {}\n\
             portalis_cpu_utilization {}\n\
             portalis_cpu_memory_bytes {}\n",
            self.metrics.tasks_completed,
            self.metrics.cpu_utilization,
            self.metrics.memory_usage
        )
    }
}
```

## Migration Path

### For Existing Users

**No Breaking Changes**:
- Existing GPU workflows continue unchanged
- CPU support is additive, not replacing GPU
- Configuration is backward compatible

**Opt-In Strategy**:
```rust
// Old code (still works)
let transpiler = TranspilerAgent::new();

// New code (CPU-aware)
let transpiler = TranspilerAgent::with_acceleration(
    AccelerationConfig::auto()
);
```

### For New Users

**Sensible Defaults**:
- Auto-detect hardware capabilities
- Use CPU if no GPU available
- Optimal thread count for CPU
- No configuration required for basic usage

## Testing Strategy

### Unit Tests
```bash
# Test CPU bridge in isolation
cargo test -p portalis-cpu-bridge

# Test with different thread counts
RAYON_NUM_THREADS=1 cargo test
RAYON_NUM_THREADS=8 cargo test
```

### Integration Tests
```bash
# Test strategy manager
cargo test -p portalis-core -- strategy

# Test CPU-only mode
cargo test --features cpu-only

# Test hybrid mode
cargo test --features hybrid
```

### Benchmark Suite
```bash
# Compare strategies
cargo bench --bench strategy_comparison

# CPU scaling test
cargo bench --bench cpu_scaling

# Memory efficiency
cargo bench --bench memory_usage
```

## Documentation Requirements

### User-Facing Docs
- [ ] CPU acceleration guide
- [ ] Configuration reference
- [ ] Performance tuning guide
- [ ] Troubleshooting common issues
- [ ] Migration guide from GPU-only

### Developer Docs
- [ ] CPU bridge architecture
- [ ] Adding new SIMD operations
- [ ] Custom execution strategies
- [ ] Profiling and optimization guide
- [ ] Contributing to CPU bridge

## Success Criteria

### Functional Requirements
✅ CPU-only execution works on all tier-1 platforms
✅ Automatic GPU → CPU fallback with no user intervention
✅ Feature parity between CPU and GPU modes
✅ Configuration via CLI, config file, and environment variables

### Performance Requirements
✅ Linear scaling up to available CPU cores
✅ < 10% overhead compared to direct Rayon usage
✅ Memory usage < 50MB per concurrent task
✅ CPU utilization > 90% on batch workloads

### Quality Requirements
✅ > 90% test coverage for CPU bridge
✅ Zero regressions in GPU performance
✅ Documentation for all public APIs
✅ Cross-platform validation complete

## Risks & Mitigation

### Risk 1: Performance Regression
**Impact**: High
**Probability**: Medium
**Mitigation**:
- Comprehensive benchmark suite
- Performance regression tests in CI
- Profile before/after optimization

### Risk 2: Platform-Specific Bugs
**Impact**: Medium
**Probability**: Medium
**Mitigation**:
- Test on all tier-1 platforms
- CI runs on Linux, macOS, Windows
- Community beta testing

### Risk 3: Complexity Increase
**Impact**: Medium
**Probability**: High
**Mitigation**:
- Keep CPU and GPU paths separate
- Clear abstraction boundaries
- Extensive documentation

## Future Enhancements

### Phase 7+: Advanced Features

**Distributed CPU Computing**:
- Cluster coordination for massive workloads
- Network-aware task distribution
- Fault tolerance and retry logic

**Adaptive Scheduling**:
- Machine learning-based strategy selection
- Runtime workload profiling
- Dynamic CPU/GPU rebalancing

**WebAssembly Support**:
- CPU bridge compiled to WASM
- Browser-based translation (limited)
- Web Workers for parallelism

**ARM-Specific Optimizations**:
- NEON SIMD intrinsics
- Apple Silicon optimizations
- AWS Graviton tuning

## Conclusion

This plan provides a comprehensive path to adding robust CPU acceleration to Portalis. By implementing intelligent strategy management and leveraging modern CPU features like multi-threading and SIMD, we can ensure excellent performance across all hardware configurations while maintaining our GPU advantage for large-scale workloads.

**Next Steps**:
1. Review and approve this plan
2. Create GitHub issues for each phase
3. Begin Phase 1 implementation
4. Set up benchmark infrastructure
5. Schedule weekly progress reviews

---

**Prepared by:** Claude Code
**Contact:** team@portalis.ai
**Last Updated:** 2025-10-07
