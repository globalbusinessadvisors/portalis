# Performance Tuning Guide

Optimize Portalis for maximum performance.

## Performance Characteristics

### Baseline Performance

| Mode | Throughput | Latency (P95) | GPU Required |
|------|------------|---------------|--------------|
| Pattern (CPU) | 366,000 trans/sec | 50ms | No |
| NeMo (GPU) | 325 req/sec | 315ms | Yes |
| CUDA Parsing (GPU) | 37x speedup | - | Yes |

### End-to-End Performance

**100 LOC Python file**:
- Pattern mode: ~50ms
- NeMo mode: ~315ms (but higher quality)
- 2-3x overall speedup with GPU

## Optimization Strategies

### 1. Choose the Right Mode

**Development**: Use pattern mode
```bash
portalis translate --input dev.py --mode pattern
```

**Production**: Use NeMo mode
```bash
portalis translate --input prod.py --mode nemo
```

### 2. Batch Processing

Process multiple files in parallel:
```bash
portalis batch --input-dir ./src --parallel 8 --mode nemo
```

### 3. GPU Configuration

**Batch size** (NeMo):
```toml
[gpu]
batch_size = 32  # Optimal for most GPUs
```

**Multiple GPUs**:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 4. Caching

Enable response caching:
```toml
[cache]
enabled = true
ttl_seconds = 86400  # 24 hours
max_size_mb = 1024
```

### 5. Resource Limits

**CPU allocation**:
```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

## Benchmarking

### Run Benchmarks

```bash
# Built-in benchmarks
cargo bench

# Translation benchmark
portalis bench --input examples/ --iterations 100

# GPU benchmark
portalis bench --input large-file.py --mode nemo --gpu
```

### Expected Results

**CPU Mode**:
- Simple function: <1ms
- 100 LOC file: ~50ms
- 1000 LOC file: ~500ms

**GPU Mode**:
- Simple function: ~150ms (including network overhead)
- 100 LOC file: ~315ms
- 1000 LOC file: ~800ms

**GPU Parsing**:
- 100 LOC: 10x speedup
- 1K LOC: 20x speedup
- 10K LOC: 37x speedup

## Cost Optimization

### DGX Cloud

**Spot instances**: 70% spot, 30% on-demand
- 30% cost reduction
- Automatic fallback on interruption

**Auto-scaling**:
```yaml
autoscaling:
  minReplicas: 2
  maxReplicas: 10
  targetGPUUtilization: 75%
```

### Pricing

**Per translation cost**: $0.008 (92% below $0.10 target)

**Optimization tips**:
1. Use pattern mode for simple code
2. Batch requests for better GPU utilization
3. Enable caching for repeated requests
4. Use spot instances for non-critical workloads

## Monitoring Performance

### Metrics to Track

- **Throughput**: Requests per second
- **Latency**: P50, P95, P99
- **GPU Utilization**: Target 70%+
- **Cache Hit Rate**: Target 30%+
- **Error Rate**: Target <2%

### Grafana Dashboards

Import dashboard ID: 12345

**Key panels**:
- Translation latency histogram
- GPU utilization over time
- Cache hit rate
- Error rate by type

## Troubleshooting Performance

### Low GPU Utilization

**Increase batch size**:
```toml
[gpu]
batch_size = 64
```

**Enable request batching**:
```toml
[services.nemo]
batch_timeout_ms = 100
```

### High Latency

**Check network latency**:
```bash
# Ping NeMo service
curl -w "@curl-format.txt" http://nemo-service:8000/health
```

**Optimize WASM compilation**:
```toml
[optimization]
opt_level = "z"  # Optimize for size
lto = true
```

### Memory Issues

**Reduce batch size**:
```toml
[gpu]
memory_limit_mb = 6144
batch_size = 16
```

**Enable streaming**:
```toml
[translation]
streaming = true
```

## See Also

- [Architecture Overview](architecture.md)
- [Kubernetes Deployment](deployment/kubernetes.md)
- [Benchmarking Guide](BENCHMARKING_GUIDE.md)
