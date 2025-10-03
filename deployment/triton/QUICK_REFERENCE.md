# Triton Deployment Quick Reference Card

## Essential Commands

### Deployment
```bash
# Deploy everything
cd deployment/triton/scripts && ./deploy-triton.sh

# Deploy with custom settings
REPLICAS=5 GPU_PER_POD=4 ./deploy-triton.sh
```

### Status Checks
```bash
# Check pods
kubectl get pods -n portalis-deployment

# Check services
kubectl get svc -n portalis-deployment

# Check HPA
kubectl get hpa -n portalis-deployment

# Check logs
kubectl logs -f triton-server-0 -n portalis-deployment
```

### Access Services
```bash
# Port-forward Triton HTTP
kubectl port-forward -n portalis-deployment svc/triton-service 8000:8000

# Port-forward Grafana
kubectl port-forward -n portalis-monitoring svc/grafana 3000:3000

# Port-forward Prometheus
kubectl port-forward -n portalis-monitoring svc/prometheus 9090:9090
```

### Scaling
```bash
# Manual scale
kubectl scale statefulset triton-server --replicas=5 -n portalis-deployment

# Check auto-scaling
kubectl describe hpa triton-server-hpa -n portalis-deployment
```

### Troubleshooting
```bash
# Describe pod
kubectl describe pod triton-server-0 -n portalis-deployment

# Check events
kubectl get events -n portalis-deployment --sort-by='.lastTimestamp'

# Execute in pod
kubectl exec -it triton-server-0 -n portalis-deployment -- bash

# Check GPU
kubectl exec -n portalis-deployment triton-server-0 -- nvidia-smi

# Check model status
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8000/v2/models
```

### Model Management
```bash
# List models
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8000/v2/models

# Reload models
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl -X POST http://localhost:8000/v2/repository/index

# Load specific model
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl -X POST http://localhost:8000/v2/repository/models/translation_model/load
```

## Python Client Usage

### Basic Translation
```python
from configs.triton_client import create_client

client = create_client(url="localhost:8000", protocol="http")

result = client.translate_code("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""")

print(result.rust_code)
print(f"Confidence: {result.confidence}")
```

### Interactive Translation
```python
result = client.translate_interactive(
    code_snippet="x = [i**2 for i in range(10)]",
    target_language="rust",
    context=["Previous code context here"]
)

print(result.translated_code)
print(f"Suggestions: {result.suggestions}")
print(f"Warnings: {result.warnings}")
```

### Batch Translation
```python
result = client.translate_batch(
    source_files=["file1.py", "file2.py"],
    project_config={"name": "myproject", "version": "1.0"},
    optimization_level="release"
)

for i, translated in enumerate(result.translated_files):
    print(f"File {i}: {len(translated)} chars")
```

## Monitoring URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Triton HTTP | http://localhost:8000 | - |
| Triton gRPC | grpc://localhost:8001 | - |
| Triton Metrics | http://localhost:8002/metrics | - |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |

## Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `nv_inference_request_success` | Successful requests | - |
| `nv_inference_request_failure` | Failed requests | >5% error rate |
| `DCGM_FI_DEV_GPU_UTIL` | GPU utilization | >70% (scale up) |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature | >85°C |
| `nv_inference_pending_request_count` | Queue depth | >100 requests |

## Configuration Files

| File | Purpose |
|------|---------|
| `models/*/config.pbtxt` | Model configuration |
| `configs/load_balancer.yaml` | NGINX load balancer |
| `configs/autoscaling.yaml` | HPA/VPA settings |
| `monitoring/prometheus-config.yaml` | Metrics collection |
| `monitoring/grafana-dashboards.json` | Visualization |

## Common Issues

| Issue | Quick Fix |
|-------|-----------|
| Pods pending | Check GPU availability: `kubectl describe nodes \| grep nvidia` |
| Models not loading | Re-copy models: `kubectl cp models triton-server-0:/` |
| High latency | Scale up: `kubectl scale sts triton-server --replicas=5` |
| Out of memory | Increase shm: Edit StatefulSet shm volume size |
| GPU not available | Check device plugin: `kubectl get ds -n kube-system \| grep nvidia` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NAMESPACE` | portalis-deployment | K8s namespace |
| `REPLICAS` | 3 | Number of pods |
| `GPU_PER_POD` | 2 | GPUs per pod |
| `TRITON_URL` | localhost:8000 | Client URL |

## File Locations

```
deployment/triton/
├── models/              # Model repository
├── configs/             # Configuration files
├── scripts/             # Deployment scripts
├── monitoring/          # Prometheus/Grafana
├── tests/               # Integration tests
└── docs/                # Documentation
```

## Support Resources

- **Documentation**: /workspace/portalis/deployment/triton/README.md
- **Deployment Guide**: /workspace/portalis/deployment/triton/DEPLOYMENT_GUIDE.md
- **Triton Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **GitHub Issues**: https://github.com/portalis/portalis/issues

---

**Keep this card handy for quick operations!**
