# Portalis NIM Microservice

Enterprise-grade NVIDIA NIM (NVIDIA Inference Microservices) implementation for portable, scalable Python-to-Rust code translation.

## Overview

The Portalis NIM microservice provides a production-ready deployment of the Portalis translation platform using NVIDIA NIM architecture. It offers:

- **High Performance**: GPU-accelerated translation with NVIDIA NeMo and Triton
- **Scalability**: Auto-scaling from 3 to 20 pods based on load
- **Enterprise Features**: Authentication, rate limiting, monitoring, and observability
- **Multiple Interfaces**: REST API, gRPC, and WebSocket streaming
- **Cloud Native**: Kubernetes-native with Helm charts for easy deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ingress (NGINX)                          │
│              SSL/TLS, Rate Limiting, CORS                   │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼─────────┐         ┌────────▼────────┐
│   REST API        │         │   gRPC API      │
│   (FastAPI)       │         │   (Port 50051)  │
│   (Port 8000)     │         │                 │
└─────────┬─────────┘         └────────┬────────┘
          │                            │
          └──────────┬─────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Translation Core   │
          │  - NeMo Service     │
          │  - Triton Client    │
          │  - CUDA Runtime     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Triton Inference   │
          │  Server             │
          │  (Model Serving)    │
          └─────────────────────┘
```

## Quick Start

### Prerequisites

- Kubernetes cluster with GPU support
- NVIDIA GPU Operator installed
- kubectl configured
- Helm 3.x installed

### Deploy with Helm

```bash
# Add Helm repository (if available)
helm repo add portalis https://charts.portalis.dev
helm repo update

# Install with default values
helm install portalis-nim portalis/portalis-nim

# Or install from local chart
helm install portalis-nim ./helm \
  --set image.tag=latest \
  --set autoscaling.enabled=true \
  --set ingress.hosts[0].host=portalis-nim.yourdomain.com
```

### Deploy with kubectl

```bash
# Apply all manifests
kubectl apply -f k8s/base/

# Check deployment status
kubectl get pods -l app=portalis-nim
kubectl get svc -l app=portalis-nim
```

### Local Development with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f portalis-nim

# Stop services
docker-compose down
```

## API Documentation

### REST API

The REST API is available at `http://localhost:8000` (or your configured ingress).

#### Endpoints

- **POST** `/api/v1/translation/translate` - Translate Python code to Rust
- **POST** `/api/v1/translation/translate/batch` - Batch translation
- **POST** `/api/v1/translation/translate/stream` - Streaming translation
- **GET** `/api/v1/translation/models` - List available models
- **GET** `/health` - Health check
- **GET** `/ready` - Readiness check
- **GET** `/live` - Liveness check
- **GET** `/metrics` - Prometheus metrics
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation (ReDoc)

#### Example: Translate Code

```bash
curl -X POST http://localhost:8000/api/v1/translation/translate \
  -H "Content-Type: application/json" \
  -d '{
    "python_code": "def add(a, b):\n    return a + b",
    "mode": "standard",
    "temperature": 0.2,
    "max_length": 512
  }'
```

Response:
```json
{
  "rust_code": "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}",
  "confidence": 0.95,
  "alternatives": [],
  "metadata": {
    "model": "translation_model",
    "prompt_length": 125,
    "output_length": 45
  },
  "warnings": [],
  "suggestions": ["Consider using generics for type flexibility"],
  "processing_time_ms": 123.45
}
```

### gRPC API

The gRPC API is available on port `50051`.

Protocol buffer definitions are in `grpc/translation.proto`.

#### Example: Python Client

```python
import grpc
from grpc import translation_pb2, translation_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = translation_pb2_grpc.TranslationServiceStub(channel)

# Translate code
request = translation_pb2.TranslateRequest(
    python_code="def multiply(a, b): return a * b",
    mode=translation_pb2.TRANSLATION_MODE_FAST
)

response = stub.TranslateCode(request)
print(f"Rust code: {response.rust_code}")
print(f"Confidence: {response.confidence}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVICE_NAME` | Service name | `portalis-nim` |
| `SERVICE_VERSION` | Service version | `1.0.0` |
| `ENVIRONMENT` | Environment (production/development) | `production` |
| `LOG_LEVEL` | Logging level | `info` |
| `HOST` | API host | `0.0.0.0` |
| `PORT` | API port | `8000` |
| `WORKERS` | Number of workers | `2` |
| `MODEL_PATH` | Path to NeMo model | `/models/nemo_translation.nemo` |
| `ENABLE_CUDA` | Enable CUDA acceleration | `true` |
| `GPU_ID` | GPU device ID | `0` |
| `TRITON_URL` | Triton server URL | `localhost:8000` |
| `TRITON_PROTOCOL` | Triton protocol (http/grpc) | `http` |
| `BATCH_SIZE` | Batch size for inference | `32` |
| `ENABLE_AUTH` | Enable authentication | `false` |
| `API_KEYS` | API keys (comma-separated) | - |
| `RATE_LIMIT_PER_MINUTE` | Rate limit per minute | `60` |
| `ENABLE_METRICS` | Enable Prometheus metrics | `true` |
| `CORS_ORIGINS` | CORS allowed origins | `*` |

### Configuration Files

Place configuration files in `/app/config`:

- `service.yaml` - Service configuration
- `logging.yaml` - Logging configuration
- `prometheus.yaml` - Prometheus scraping config

## Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `nim_requests_total` - Total number of requests
- `nim_request_duration_seconds` - Request duration histogram
- `nim_requests_in_progress` - In-progress requests gauge
- `nim_translation_duration_seconds` - Translation duration histogram
- `nim_translations_total` - Total translations counter
- `nim_gpu_memory_bytes` - GPU memory usage
- `nim_gpu_utilization_percent` - GPU utilization percentage

### Grafana Dashboards

Import pre-built dashboards from `k8s/monitoring/grafana-dashboards/`:

1. NIM Overview Dashboard
2. Performance Metrics Dashboard
3. GPU Utilization Dashboard

## Scaling

### Horizontal Pod Autoscaling (HPA)

The service automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Custom metrics (GPU utilization, requests/second)

Configuration:
```yaml
autoscaling:
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
```

### Manual Scaling

```bash
# Scale to specific number of replicas
kubectl scale deployment portalis-nim --replicas=5

# Using Helm
helm upgrade portalis-nim ./helm --set replicaCount=5
```

## Security

### Authentication

Enable API key authentication:

```yaml
env:
  ENABLE_AUTH: "true"
  API_KEYS: "client1:key1,client2:key2"
```

Use API key in requests:
```bash
curl -H "Authorization: Bearer key1" \
  http://localhost:8000/api/v1/translation/translate
```

### Rate Limiting

Rate limiting is configured per client:
- Default: 60 requests/minute
- Burst: 10 requests

### TLS/SSL

Configure TLS in ingress:
```yaml
tls:
  - secretName: portalis-nim-tls
    hosts:
      - portalis-nim.yourdomain.com
```

## Troubleshooting

### Pod not starting

```bash
# Check pod status
kubectl describe pod -l app=portalis-nim

# Check logs
kubectl logs -l app=portalis-nim --tail=100

# Common issues:
# - Model file not found: Check PVC mount
# - GPU not available: Verify GPU operator installation
# - OOM: Increase memory limits
```

### High latency

```bash
# Check GPU utilization
kubectl exec -it <pod-name> -- nvidia-smi

# Check metrics
curl http://localhost:8000/metrics

# Possible solutions:
# - Increase replicas
# - Adjust batch size
# - Enable GPU caching
```

### Connection issues

```bash
# Check service endpoints
kubectl get endpoints portalis-nim-api

# Check ingress
kubectl describe ingress portalis-nim-ingress

# Test internal connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://portalis-nim-api:8000/health
```

## Performance Tuning

### GPU Optimization

```yaml
env:
  GPU_MEMORY_FRACTION: "0.9"  # Use 90% of GPU memory
  BATCH_SIZE: "32"            # Optimize batch size for GPU
```

### Worker Configuration

```yaml
env:
  WORKERS: "2"  # Number of Uvicorn workers
```

### Resource Limits

```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "4"
    memory: "8Gi"
    nvidia.com/gpu: "1"
```

## Development

### Build Docker Image

```bash
# Standard build
docker build -t portalis-nim:latest .

# Optimized build (smaller size)
docker build -f Dockerfile.optimized -t portalis-nim:optimized .

# Multi-arch build
docker buildx build --platform linux/amd64,linux/arm64 \
  -t portalis-nim:multi-arch .
```

### Run Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=api --cov-report=html
```

### Generate gRPC Code

```bash
python -m grpc_tools.protoc \
  -I./grpc \
  --python_out=./grpc \
  --grpc_python_out=./grpc \
  ./grpc/translation.proto
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

## Support

- Documentation: https://docs.portalis.dev
- Issues: https://github.com/portalis/portalis/issues
- Discussions: https://github.com/portalis/portalis/discussions
