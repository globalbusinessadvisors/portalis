# Docker Compose Deployment Guide

Quick start guide for deploying Portalis locally with Docker Compose.

## Quick Start

```bash
# Clone repository
git clone https://github.com/portalis/portalis.git
cd portalis

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Access API
curl http://localhost:8080/health
```

## Architecture

```
┌──────────────────┐
│   NGINX (80)     │ → Load balancer
└──────────────────┘
         ↓
┌──────────────────┬──────────────────┬──────────────────┐
│  Translation     │  NeMo Service    │  CUDA Parser     │
│  API (3x)        │  (GPU)           │  (GPU)           │
│  Port: 8080-8082 │  Port: 8000      │  Port: 8001      │
└──────────────────┴──────────────────┴──────────────────┘
         ↓                   ↓                   ↓
┌──────────────────┬──────────────────┬──────────────────┐
│   Redis Cache    │  Prometheus      │  Grafana         │
│   Port: 6379     │  Port: 9090      │  Port: 3000      │
└──────────────────┴──────────────────┴──────────────────┘
```

## Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  # Translation API (CPU)
  portalis-api:
    image: portalis/portalis:latest
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - NEMO_SERVICE_URL=http://nemo:8000
      - CUDA_SERVICE_URL=http://cuda:8001
      - REDIS_URL=redis://redis:6379
    depends_on:
      - nemo
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # NeMo Translation Service (GPU)
  nemo:
    image: portalis/nemo-service:latest
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/models/nemo-translation
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # CUDA Parsing Service (GPU)
  cuda:
    image: portalis/cuda-parser:latest
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Redis Cache
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru

  # NGINX Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - portalis-api

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

## GPU Configuration

### Prerequisites

```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### GPU-Enabled Compose

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  portalis-api:
    extends:
      file: docker-compose.yml
      service: portalis-api

  nemo:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Common Operations

### Start Services

```bash
# CPU-only mode
docker-compose up -d

# GPU-enabled mode
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Scale API replicas
docker-compose up -d --scale portalis-api=5
```

### Monitoring

```bash
# View logs
docker-compose logs -f portalis-api
docker-compose logs -f nemo

# Check resource usage
docker stats

# Access Grafana
open http://localhost:3000  # admin/admin
```

### Maintenance

```bash
# Update images
docker-compose pull
docker-compose up -d

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Remove volumes
docker-compose down -v
```

## Environment Variables

Create `.env` file:

```bash
# .env
PORTALIS_VERSION=latest
RUST_LOG=info
NEMO_MODEL_PATH=./models/nemo
REDIS_MAX_MEMORY=2gb
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=changeme
```

## Troubleshooting

**Services not starting**:
```bash
docker-compose logs
docker-compose ps
```

**GPU not detected**:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Out of memory**:
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or adjust service limits in docker-compose.yml
```

## See Also

- [Kubernetes Deployment](kubernetes.md)
- [Performance Tuning](../performance.md)
- [Security Guide](../security.md)
