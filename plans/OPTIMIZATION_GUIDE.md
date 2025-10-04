# Portalis Optimization Guide

## How to Achieve Maximum Performance

This guide explains the optimization techniques used in Portalis and how to configure them for your specific use case.

---

## 1. NeMo Model Optimization

### Quick Start

```python
from optimizations.nemo_optimizations import NeMoOptimizer, OptimizationConfig

# Create configuration
config = OptimizationConfig(
    use_tensorrt=True,
    trt_precision="fp16",
    use_quantization=True,
    quantization_bits=8,
    enable_flash_attention=True,
    optimal_batch_size=32
)

# Initialize optimizer
optimizer = NeMoOptimizer(config)

# Optimize your model
optimized_model = optimizer.optimize_model(
    model=your_nemo_model,
    sample_inputs=calibration_data
)

# Get optimization report
report = optimizer.get_optimization_report()
print(report)
```

### Configuration Options

#### TensorRT Settings

```python
# For maximum speed (slight accuracy trade-off)
config.use_tensorrt = True
config.trt_precision = "fp16"  # 2-3x faster than FP32

# For balanced performance
config.trt_precision = "fp16"
config.trt_max_batch_size = 64

# For maximum accuracy
config.trt_precision = "fp32"  # Slower but most accurate
```

#### Quantization Settings

```python
# INT8 quantization (50% memory reduction)
config.use_quantization = True
config.quantization_bits = 8

# INT4 quantization (75% memory reduction, requires bitsandbytes)
config.quantization_bits = 4

# Disable quantization for maximum accuracy
config.use_quantization = False
```

#### Batch Size Optimization

```python
# For latency-sensitive applications
config.optimal_batch_size = 8
config.batch_timeout_ms = 50

# For throughput-sensitive applications
config.optimal_batch_size = 64
config.batch_timeout_ms = 200

# For balanced performance
config.optimal_batch_size = 32
config.batch_timeout_ms = 100
```

### Expected Performance Gains

| Optimization | Speedup | Memory Reduction | Accuracy Impact |
|-------------|---------|------------------|-----------------|
| TensorRT FP16 | 2.5x | 40% | <1% loss |
| INT8 Quantization | 1.3x | 50% | <2% loss |
| INT4 Quantization | 1.5x | 75% | 2-5% loss |
| Flash Attention | 2-4x | 0% | None |
| Batching (32) | 1.5x | 0% | None |

---

## 2. CUDA Kernel Optimization

### Compilation

```bash
# Compile optimized CUDA kernels
nvcc -O3 -arch=sm_80 \
  -I/usr/local/cuda/include \
  -c optimizations/cuda_optimizations.cu \
  -o cuda_optimizations.o

# Link with your application
nvcc cuda_optimizations.o your_app.o -o your_app
```

### Usage

```cpp
#include "cuda_optimizations.cuh"

// Initialize
ParserConfig config;
config.max_tokens = 100000;
config.max_nodes = 50000;
initializeASTParser(&config);

// Use optimized kernels
cudaStream_t stream;
cudaStreamCreate(&stream);

launchFusedTokenizeEmbed(
    source_code,
    code_length,
    embedding_table,
    vocab_size,
    embedding_dim,
    embeddings_out,
    tokens_out,
    token_count,
    stream
);

cudaStreamSynchronize(stream);
```

### Tuning Parameters

```cpp
// For small code snippets (<1KB)
const int BLOCK_SIZE = 128;
const int GRID_SIZE = 4;

// For medium code files (1KB-100KB)
const int BLOCK_SIZE = 256;
const int GRID_SIZE = (code_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

// For large code files (>100KB)
const int BLOCK_SIZE = 512;
const int GRID_SIZE = min(512, (code_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
```

### Performance Optimization Checklist

✅ Use kernel fusion to reduce memory transfers
✅ Ensure memory coalescing (aligned, sequential access)
✅ Utilize shared memory (48KB per SM)
✅ Minimize bank conflicts (stride by 33 for padding)
✅ Use warp-level primitives for reductions
✅ Leverage Tensor Cores for matrix operations
✅ Stream concurrent kernels
✅ Use async memory copies

---

## 3. Triton Inference Server Optimization

### Model Configuration

Edit `/workspace/portalis/optimizations/triton_optimizations.yaml`:

```yaml
# For latency-critical applications
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: 50000  # 50ms
}

# For throughput-critical applications
dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 200000  # 200ms
}

# For balanced performance
dynamic_batching {
  preferred_batch_size: [ 1, 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100000  # 100ms
}
```

### Instance Configuration

```yaml
# For high availability (more instances, lower throughput)
instance_group [
  {
    count: 4  # 4 instances per GPU
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]

# For high throughput (fewer instances, higher throughput)
instance_group [
  {
    count: 1  # 1 instance per GPU
    kind: KIND_GPU
    gpus: [ 0, 1, 2, 3 ]
  }
]
```

### CUDA Graphs

```yaml
optimization {
  cuda {
    graphs: true
    graph_spec {
      batch_size: 1
    }
    graph_spec {
      batch_size: 8
    }
    graph_spec {
      batch_size: 32
    }
  }
}
```

**Impact:** 70% reduction in kernel launch overhead

---

## 4. NIM Microservice Optimization

### Connection Pooling

```python
from optimizations.nim_optimizations import NIMOptimizer, NIMOptimizationConfig

config = NIMOptimizationConfig(
    max_connections=100,      # Max connections in pool
    min_connections=10,       # Keep-alive connections
    connection_timeout_seconds=30,
    connection_ttl_seconds=300  # Recycle connections every 5 min
)

optimizer = NIMOptimizer(config)
await optimizer.initialize()
```

### Caching Strategy

```python
# For static code translation (high cache hit rate)
config.enable_cache = True
config.cache_size_mb = 2048  # 2GB cache
config.cache_ttl_seconds = 7200  # 2 hours

# For dynamic code translation (low cache hit rate)
config.enable_cache = True
config.cache_size_mb = 512   # 512MB cache
config.cache_ttl_seconds = 1800  # 30 minutes

# Disable caching
config.enable_cache = False
```

### Response Compression

```python
# Enable compression for bandwidth savings
config.enable_compression = True
config.compression_level = 6  # Balanced (1-9)
config.compression_min_size_bytes = 1024  # Only compress >1KB

# Maximum compression (slower)
config.compression_level = 9

# Disable compression for low-latency
config.enable_compression = False
```

### Batch Processing

```python
# Enable batching for throughput
config.enable_batching = True
config.max_batch_size = 32
config.batch_timeout_ms = 50

# Disable batching for latency
config.enable_batching = False
```

---

## 5. DGX Cloud Optimization

### Job Scheduling

```python
from optimizations.dgx_optimizations import DGXOptimizer, DGXOptimizationConfig, JobPriority

config = DGXOptimizationConfig(
    # Cost optimization
    target_cost_per_translation=0.08,
    spot_instance_ratio=0.75,  # 75% spot, 25% on-demand

    # Resource utilization
    target_gpu_utilization=0.85,
    min_gpu_utilization=0.60,

    # Auto-scaling
    enable_autoscaling=True,
    scale_up_threshold=0.90,
    scale_down_threshold=0.30,
    min_nodes=1,
    max_nodes=10
)

optimizer = DGXOptimizer(config)

# Submit jobs with priorities
high_priority_job = optimizer.submit_translation_job(
    code_size_loc=50000,
    priority=JobPriority.HIGH,
    deadline=datetime.now() + timedelta(hours=1)
)
```

### Cost Optimization

```python
# For minimum cost (use more spot instances)
config.spot_instance_ratio = 0.85
config.enable_spot_instances = True
config.spot_fallback_on_demand = True

# For reliability (use more on-demand)
config.spot_instance_ratio = 0.30
config.enable_spot_instances = True

# For maximum reliability (on-demand only)
config.enable_spot_instances = False
```

### Cluster Sizing

```python
# For variable workload
config.enable_autoscaling = True
config.min_nodes = 2
config.max_nodes = 20

# For predictable workload
config.enable_autoscaling = False
config.min_nodes = 5
config.max_nodes = 5

# For burst workload
config.enable_autoscaling = True
config.scale_up_threshold = 0.70  # Scale up earlier
config.min_nodes = 1
config.max_nodes = 50
```

---

## 6. End-to-End Pipeline Optimization

### Configuration

```python
from optimizations.pipeline_optimizations import PipelineOptimizer, PipelineConfig

# For maximum throughput
config = PipelineConfig(
    max_parallel_stages=8,
    enable_stage_fusion=True,
    enable_intermediate_cache=True,
    enable_memory_pooling=True,
    skip_validation_on_high_confidence=True,
    confidence_threshold=0.95
)

# For minimum latency
config = PipelineConfig(
    max_parallel_stages=16,
    enable_stage_fusion=True,
    enable_intermediate_cache=True,
    skip_validation_on_high_confidence=True,
    confidence_threshold=0.90
)

# For maximum accuracy
config = PipelineConfig(
    max_parallel_stages=4,
    enable_stage_fusion=False,  # Separate stages
    skip_validation_on_high_confidence=False  # Always validate
)

optimizer = PipelineOptimizer(config)
```

### Batch Processing

```python
# Process multiple files efficiently
batch_data = [
    PipelineData(job_id=f"job-{i}", python_code=code)
    for i, code in enumerate(code_files)
]

tasks = optimizer.optimize_batch(batch_data)
results = await asyncio.gather(*tasks)
```

---

## 7. Performance Tuning Workflow

### Step 1: Measure Baseline

```bash
# Run benchmarks
python benchmarks/benchmark_e2e.py

# Save baseline results
cp benchmarks/e2e_results.json benchmarks/baseline_results.json
```

### Step 2: Apply Optimizations

Start with high-impact, low-risk optimizations:

1. ✅ Enable TensorRT (2-3x speedup)
2. ✅ Enable Flash Attention (2-4x speedup)
3. ✅ Optimize batch size (1.5-2x speedup)
4. ✅ Enable caching (variable speedup)
5. ✅ Apply INT8 quantization (1.3x speedup, 50% memory)

### Step 3: Validate Performance

```bash
# Re-run benchmarks
python benchmarks/benchmark_e2e.py

# Compare results
python scripts/compare_results.py baseline_results.json e2e_results.json
```

### Step 4: Monitor in Production

```bash
# Start monitoring
docker-compose up -d prometheus grafana

# View dashboard
open http://localhost:3000/d/portalis-performance
```

### Step 5: Iterate

Monitor metrics and identify bottlenecks:
- High latency → Increase parallelism, reduce batch timeout
- Low GPU utilization → Increase batch size
- High cost → Increase spot instance ratio
- Cache misses → Increase cache size, longer TTL

---

## 8. Troubleshooting

### High Latency

**Symptom:** P95 latency >500ms for 100 LOC

**Solutions:**
1. Reduce batch timeout: `config.batch_timeout_ms = 50`
2. Increase parallelism: `config.max_parallel_stages = 8`
3. Enable stage fusion: `config.enable_stage_fusion = True`
4. Check GPU utilization (should be >60%)

### Low Throughput

**Symptom:** QPS <100

**Solutions:**
1. Increase batch size: `config.optimal_batch_size = 64`
2. Add more GPU instances
3. Enable batching: `config.enable_batching = True`
4. Check for pipeline bottlenecks

### High Cost

**Symptom:** Cost >$0.10 per translation

**Solutions:**
1. Increase spot instance ratio: `config.spot_instance_ratio = 0.85`
2. Enable auto-scaling: `config.enable_autoscaling = True`
3. Optimize batch processing
4. Enable INT8 quantization

### Low GPU Utilization

**Symptom:** GPU utilization <60%

**Solutions:**
1. Increase batch size
2. Reduce model instance count
3. Enable dynamic batching
4. Check for CPU bottlenecks

### OOM (Out of Memory)

**Symptom:** CUDA out of memory errors

**Solutions:**
1. Enable INT8 quantization: `config.use_quantization = True`
2. Reduce batch size: `config.optimal_batch_size = 16`
3. Enable gradient checkpointing
4. Use INT4 quantization for extreme cases

---

## 9. Best Practices

### Development

✅ Start with default configs
✅ Enable all optimizations in testing
✅ Benchmark before and after changes
✅ Use version control for configs
✅ Document optimization decisions

### Production

✅ Monitor all metrics continuously
✅ Set up alerts for SLA violations
✅ Use auto-scaling for variable load
✅ Keep 20% capacity headroom
✅ Test optimizations in staging first

### Cost Management

✅ Use spot instances for >70% of workload
✅ Enable auto-scaling down during off-peak
✅ Monitor cost per translation metric
✅ Set budget alerts
✅ Review cost reports weekly

---

## 10. Quick Reference

### Performance Targets

| Metric | Target | Config |
|--------|--------|--------|
| P95 Latency (100 LOC) | <500ms | `batch_timeout_ms=100` |
| Throughput | >100 QPS | `batch_size=32` |
| GPU Utilization | >70% | `instances_per_gpu=2` |
| Cost per Translation | <$0.10 | `spot_ratio=0.75` |

### Configuration Templates

```python
# Template: Low Latency
LOW_LATENCY_CONFIG = {
    'batch_size': 8,
    'batch_timeout_ms': 50,
    'tensorrt': True,
    'precision': 'fp16',
    'quantization': False
}

# Template: High Throughput
HIGH_THROUGHPUT_CONFIG = {
    'batch_size': 64,
    'batch_timeout_ms': 200,
    'tensorrt': True,
    'precision': 'fp16',
    'quantization': True
}

# Template: Low Cost
LOW_COST_CONFIG = {
    'batch_size': 32,
    'spot_ratio': 0.85,
    'quantization': True,
    'quantization_bits': 8,
    'autoscaling': True
}
```

---

**Last Updated:** 2025-10-03
**Version:** 1.0
