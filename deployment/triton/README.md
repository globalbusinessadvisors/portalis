# Triton Inference Server Deployment for Portalis

## Overview

This directory contains the complete Triton Inference Server deployment infrastructure for the Portalis Python-to-Rust translation pipeline. The deployment supports:

- **Batch Translation**: Large-scale translation of entire Python projects
- **Interactive Translation**: Real-time, context-aware translation API
- **GPU Acceleration**: NeMo-powered translation with CUDA optimization
- **Auto-scaling**: Kubernetes HPA based on GPU utilization and queue depth
- **Monitoring**: Prometheus + Grafana dashboards with custom metrics
- **Load Balancing**: NGINX-based load balancing across multiple Triton instances

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Load Balancer (NGINX)                  │
│              HTTP (8000) / gRPC (8001)                  │
└────────────────────┬────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼─────┐   ┌────▼─────┐   ┌────▼─────┐
│ Triton-0 │   │ Triton-1 │   │ Triton-2 │
│ 2x GPUs  │   │ 2x GPUs  │   │ 2x GPUs  │
└──────────┘   └──────────┘   └──────────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
         ┌───────────▼───────────┐
         │ Model Repository (PVC)│
         │  - translation_model  │
         │  - interactive_api    │
         │  - batch_processor    │
         └───────────────────────┘
```

## Directory Structure

```
deployment/triton/
├── models/                          # Triton model repository
│   ├── translation_model/          # Main translation service
│   │   ├── config.pbtxt            # Model configuration
│   │   └── 1/
│   │       └── model.py            # Python backend implementation
│   ├── interactive_api/            # Interactive translation API
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.py
│   └── batch_processor/            # Batch processing pipeline
│       └── config.pbtxt
├── configs/                         # Configuration files
│   ├── triton_client.py            # Python client library
│   ├── load_balancer.yaml          # NGINX load balancer config
│   └── autoscaling.yaml            # HPA and VPA configs
├── monitoring/                      # Monitoring infrastructure
│   ├── prometheus-config.yaml      # Prometheus configuration
│   └── grafana-dashboards.json     # Grafana dashboards
├── scripts/                         # Deployment scripts
│   └── deploy-triton.sh            # Main deployment script
├── tests/                           # Integration tests
│   └── test_triton_integration.py  # Test suite
├── Dockerfile                       # Custom Triton image
└── README.md                        # This file
```

## Prerequisites

### Required

- Kubernetes cluster (1.24+)
- NVIDIA GPU nodes with:
  - NVIDIA GPU Operator installed
  - GPU device plugin running
  - At least 2 GPUs per node (A100/V100 recommended)
- kubectl configured with cluster access
- Persistent Volume support (ReadWriteMany for model repository)

### Optional

- Helm 3.x (for easier deployment)
- Docker (for building custom images)
- NVIDIA NGC account (for accessing base images)

## Quick Start

### 1. Deploy Triton Inference Server

```bash
# Set environment variables (optional)
export NAMESPACE=portalis-deployment
export TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3
export REPLICAS=3
export GPU_PER_POD=2

# Run deployment script
cd deployment/triton/scripts
./deploy-triton.sh
```

### 2. Verify Deployment

```bash
# Check pod status
kubectl get pods -n portalis-deployment

# Check services
kubectl get svc -n portalis-deployment

# Check HPA
kubectl get hpa -n portalis-deployment

# View logs
kubectl logs -n portalis-deployment triton-server-0 -f
```

### 3. Access Services

```bash
# Port-forward Triton HTTP API
kubectl port-forward -n portalis-deployment svc/triton-service 8000:8000

# Port-forward Triton gRPC API
kubectl port-forward -n portalis-deployment svc/triton-service 8001:8001

# Port-forward Grafana
kubectl port-forward -n portalis-monitoring svc/grafana 3000:3000
```

### 4. Test Translation

```python
from configs.triton_client import create_client

# Create client
client = create_client(url="localhost:8000", protocol="http")

# Translate code
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

result = client.translate_code(python_code)
print(f"Translated code:\n{result.rust_code}")
print(f"Confidence: {result.confidence}")
```

## Models

### Translation Model

**Purpose**: Core Python-to-Rust translation using NeMo

**Endpoints**:
- Input: `python_code` (string), `translation_options` (optional JSON)
- Output: `rust_code` (string), `confidence_score` (float), `metadata` (JSON)

**Configuration**:
- Max batch size: 32
- Dynamic batching with preferred sizes: [8, 16, 32]
- GPU instances: 3 (2 on GPU 0, 1 on GPU 1)

**Usage**:
```python
result = client.translate_code(
    python_code="def hello(): pass",
    options={"optimization_level": "release"}
)
```

### Interactive API

**Purpose**: Real-time, context-aware translation for IDE integration

**Endpoints**:
- Input: `code_snippet`, `target_language`, `context` (optional)
- Output: `translated_code`, `confidence`, `suggestions`, `warnings`

**Features**:
- Sequence batching for multi-turn conversations
- Session management with context caching
- Low-latency responses (< 50ms queue delay)

**Usage**:
```python
result = client.translate_interactive(
    code_snippet="x = [i**2 for i in range(10)]",
    target_language="rust",
    context=["Previous context here"]
)
```

### Batch Processor

**Purpose**: End-to-end translation of entire projects

**Endpoints**:
- Input: `source_files[]`, `project_config`, `optimization_level`
- Output: `translated_files[]`, `compilation_status[]`, `performance_metrics`, `wasm_binaries[]`

**Pipeline Stages**:
1. Analysis Agent (API extraction)
2. Spec Generator (Rust spec generation)
3. Translation Model (code translation)
4. Build Agent (WASM compilation)
5. Test Validator (conformance testing)

**Usage**:
```python
result = client.translate_batch(
    source_files=["file1.py", "file2.py"],
    project_config={"name": "myproject", "version": "1.0"},
    optimization_level="release"
)
```

## Monitoring

### Prometheus Metrics

**Triton Metrics**:
- `nv_inference_request_success` - Successful inference requests
- `nv_inference_request_failure` - Failed inference requests
- `nv_inference_request_duration_us` - Request latency histogram
- `nv_inference_queue_duration_us` - Queue time
- `nv_inference_pending_request_count` - Queue depth

**GPU Metrics** (via DCGM):
- `DCGM_FI_DEV_GPU_UTIL` - GPU utilization %
- `DCGM_FI_DEV_FB_USED` - GPU memory used
- `DCGM_FI_DEV_GPU_TEMP` - GPU temperature

### Grafana Dashboards

Access Grafana at `http://<GRAFANA_IP>:3000` (default: admin/admin)

**Dashboards**:
- Translation Request Rate
- GPU Utilization
- Request Latency (P50/P95/P99)
- Model Queue Depth
- Error Rate and Types
- Cache Hit Rate

### Alerts

**Critical Alerts**:
- Triton server down (>2 min)
- GPU temperature > 85°C
- High error rate (>5% for 5 min)

**Warning Alerts**:
- GPU memory pressure (>90% for 10 min)
- High request latency (P95 > 5s)
- Model queue backlog (>100 requests)

## Auto-scaling

### Horizontal Pod Autoscaler (HPA)

**Metrics**:
- GPU utilization: Scale when >70%
- Queue duration: Scale when >100ms
- CPU utilization: Scale when >75%
- Memory utilization: Scale when >80%

**Scaling Behavior**:
- Scale up: 100% increase (or +4 pods) every 30s
- Scale down: 50% decrease (or -2 pods) every 60s
- Min replicas: 2
- Max replicas: 10

**Stability**:
- Scale-up stabilization: 60s
- Scale-down stabilization: 300s

### Vertical Pod Autoscaler (VPA)

**Resource Limits**:
- Min: 2 CPU, 8Gi RAM, 1 GPU
- Max: 16 CPU, 64Gi RAM, 4 GPUs

**Update Mode**: Auto (applies recommendations automatically)

## Load Balancing

### NGINX Configuration

**Algorithm**: Least connections

**Health Checks**:
- Max failures: 3
- Fail timeout: 30s
- Keepalive connections: 32

**Endpoints**:
- HTTP/REST: Port 80 → Triton 8000
- gRPC: Port 8001 → Triton 8001
- Metrics: `/metrics` → Triton metrics endpoint

**Session Affinity**: ClientIP with 1-hour timeout

## Performance Tuning

### Triton Server Options

```bash
--backend-config=python,shm-default-byte-size=16777216  # 16MB shared memory
--allow-gpu-metrics=true                                 # Enable GPU metrics
--metrics-port=8002                                      # Metrics endpoint
--log-verbose=1                                          # Verbose logging
```

### Dynamic Batching

**Translation Model**:
- Preferred batch sizes: 8, 16, 32
- Max queue delay: 100ms
- Priority levels: 2 (interactive=high, batch=normal)

**Interactive API**:
- Preferred batch sizes: 1, 2, 4, 8
- Max queue delay: 50ms (low latency)
- Priority levels: 3

### GPU Memory Optimization

- Shared memory: 8Gi per pod
- CUDA graphs enabled for common batch sizes
- Model warmup on startup
- Pinned memory for inputs/outputs

## Testing

### Run Integration Tests

```bash
# Install dependencies
pip install pytest tritonclient[all]

# Set Triton URL
export TRITON_URL=localhost:8000

# Run all tests
cd tests
pytest test_triton_integration.py -v

# Run specific test class
pytest test_triton_integration.py::TestSingleTranslation -v

# Run with coverage
pytest test_triton_integration.py --cov=configs --cov-report=html
```

### Test Categories

1. **Single Translation**: Basic code translation tests
2. **Interactive Translation**: Context-aware translation
3. **Batch Translation**: Multi-file project translation
4. **Performance**: Latency and throughput benchmarks
5. **Error Handling**: Edge cases and invalid inputs

## Troubleshooting

### Common Issues

**1. Pods not starting**
```bash
# Check events
kubectl describe pod triton-server-0 -n portalis-deployment

# Check GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
```

**2. Models not loading**
```bash
# Check model repository
kubectl exec -n portalis-deployment triton-server-0 -- ls -la /models

# Check Triton logs
kubectl logs -n portalis-deployment triton-server-0 --tail=100

# Reload models
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl -X POST http://localhost:8000/v2/repository/index
```

**3. High latency**
```bash
# Check queue depth
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8002/metrics | grep queue

# Check GPU utilization
kubectl exec -n portalis-deployment triton-server-0 -- nvidia-smi

# Scale up if needed
kubectl scale statefulset triton-server -n portalis-deployment --replicas=5
```

**4. Memory issues**
```bash
# Check memory usage
kubectl top pods -n portalis-deployment

# Increase shared memory
# Edit StatefulSet and increase shm volume size
kubectl edit statefulset triton-server -n portalis-deployment
```

## Security

### Network Policies

```yaml
# Restrict traffic to Triton pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: triton-network-policy
  namespace: portalis-deployment
spec:
  podSelector:
    matchLabels:
      app: triton-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: portalis-deployment
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8001
```

### RBAC

Triton pods use ServiceAccount `prometheus` for metrics scraping:

```bash
# View permissions
kubectl get clusterrole prometheus -o yaml
```

## Maintenance

### Updating Models

```bash
# Copy new model version
kubectl cp new_model.py portalis-deployment/triton-server-0:/models/translation_model/2/

# Load new version
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl -X POST http://localhost:8000/v2/repository/models/translation_model/load

# Unload old version
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl -X POST http://localhost:8000/v2/repository/models/translation_model/unload?version=1
```

### Backup and Restore

```bash
# Backup model repository
kubectl exec -n portalis-deployment triton-server-0 -- tar czf /tmp/models-backup.tar.gz /models
kubectl cp portalis-deployment/triton-server-0:/tmp/models-backup.tar.gz ./models-backup.tar.gz

# Restore
kubectl cp ./models-backup.tar.gz portalis-deployment/triton-server-0:/tmp/
kubectl exec -n portalis-deployment triton-server-0 -- tar xzf /tmp/models-backup.tar.gz -C /
```

## Production Checklist

- [ ] GPU nodes labeled and tainted appropriately
- [ ] Resource quotas configured
- [ ] Persistent volumes provisioned (ReadWriteMany)
- [ ] Monitoring stack deployed (Prometheus + Grafana)
- [ ] Alerting rules configured
- [ ] Load balancer deployed and tested
- [ ] HPA and VPA configured
- [ ] Network policies applied
- [ ] TLS certificates configured (if external)
- [ ] Backup strategy implemented
- [ ] Integration tests passing
- [ ] Performance benchmarks validated
- [ ] Documentation reviewed

## Support

For issues and questions:
- GitHub Issues: [Portalis Repository](https://github.com/portalis/portalis)
- Documentation: [Portalis Docs](https://portalis.dev/docs)
- NVIDIA Triton: [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)

## License

Copyright (c) 2025 Portalis Team
Licensed under the MIT License
