# Kubernetes Deployment Guide

Production deployment of Portalis on Kubernetes with GPU support.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Helm Chart Installation](#helm-chart-installation)
- [GPU Node Configuration](#gpu-node-configuration)
- [Scaling Configuration](#scaling-configuration)
- [High Availability](#high-availability)
- [Monitoring Integration](#monitoring-integration)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- Kubernetes 1.24+ cluster
- kubectl 1.24+
- Helm 3.8+
- Persistent Volume provisioner
- Ingress controller (NGINX recommended)

### Optional (for GPU acceleration)

- NVIDIA GPU nodes
- NVIDIA Device Plugin
- CUDA 12.0+
- GPU-enabled container runtime

### Verify Prerequisites

```bash
# Check Kubernetes version
kubectl version --short

# Check Helm version
helm version --short

# Check GPU nodes (if applicable)
kubectl get nodes -l nvidia.com/gpu=true

# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia
```

---

## Architecture

### Deployment Components

```
┌─────────────────────────────────────────────────────┐
│                    Ingress (NGINX)                   │
│            portalis.example.com → Service            │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                  Portalis Service                    │
│              LoadBalancer / ClusterIP                │
└─────────────────────────────────────────────────────┘
                         ↓
┌────────────────────┬────────────────┬───────────────┐
│ Translation API    │  NeMo Service  │ CUDA Parser   │
│  (3 replicas)      │  (2 replicas)  │ (1 replica)   │
│  CPU pods          │  GPU pods      │ GPU pods      │
└────────────────────┴────────────────┴───────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              Persistent Storage (PVC)                │
│        Cache | Models | Artifacts                    │
└─────────────────────────────────────────────────────┘
```

### Resource Distribution

| Component | CPU | Memory | GPU | Replicas |
|-----------|-----|--------|-----|----------|
| Translation API | 500m-2 | 1-4Gi | - | 2-10 (HPA) |
| NeMo Service | 2-4 | 8-16Gi | 1 | 2-4 |
| CUDA Parser | 1-2 | 2-4Gi | 1 | 1-2 |
| Triton Server | 2-4 | 8-16Gi | 1 | 2-4 |

---

## Helm Chart Installation

### Add Helm Repository

```bash
# Add Portalis Helm repository
helm repo add portalis https://charts.portalis.dev
helm repo update

# Search for available versions
helm search repo portalis
```

### Install Chart

**Basic Installation** (CPU-only):

```bash
helm install portalis portalis/portalis \
  --namespace portalis \
  --create-namespace \
  --set gpu.enabled=false
```

**Production Installation** (with GPU):

```bash
helm install portalis portalis/portalis \
  --namespace portalis \
  --create-namespace \
  --set gpu.enabled=true \
  --set gpu.count=2 \
  --set replicaCount=3 \
  --set resources.requests.memory=4Gi \
  --set resources.requests.cpu=2 \
  --set persistence.enabled=true \
  --set persistence.size=50Gi
```

**Custom Values File**:

Create `values.yaml`:

```yaml
# values.yaml

# Number of replicas (auto-scaled)
replicaCount: 3

# GPU configuration
gpu:
  enabled: true
  count: 1  # GPUs per pod
  type: nvidia.com/gpu

# Resource limits
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: 1

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Persistent storage
persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 100Gi
  mountPath: /data

# Service configuration
service:
  type: LoadBalancer
  port: 80
  targetPort: 8080
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

# Ingress configuration
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: portalis.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: portalis-tls
      hosts:
        - portalis.example.com

# NVIDIA services
nemo:
  enabled: true
  replicaCount: 2
  resources:
    requests:
      memory: "8Gi"
      cpu: "2"
      nvidia.com/gpu: 1
    limits:
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: 1

triton:
  enabled: true
  replicaCount: 3
  modelRepository: /models
  resources:
    requests:
      memory: "8Gi"
      cpu: "2"
      nvidia.com/gpu: 1

# Monitoring
monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

Install with custom values:

```bash
helm install portalis portalis/portalis \
  --namespace portalis \
  --create-namespace \
  --values values.yaml
```

### Verify Installation

```bash
# Check deployment status
helm status portalis -n portalis

# Check pods
kubectl get pods -n portalis

# Check services
kubectl get svc -n portalis

# Check ingress
kubectl get ingress -n portalis

# Get service URL
kubectl get svc portalis -n portalis -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

---

## GPU Node Configuration

### Install NVIDIA Device Plugin

```bash
# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify installation
kubectl get pods -n kube-system | grep nvidia

# Expected output:
# nvidia-device-plugin-daemonset-xxxxx   1/1     Running   0          5m
```

### Label GPU Nodes

```bash
# List nodes
kubectl get nodes

# Label GPU nodes
kubectl label nodes <node-name> nvidia.com/gpu=true
kubectl label nodes <node-name> gpu-type=rtx4090

# Verify labels
kubectl get nodes -L nvidia.com/gpu,gpu-type
```

### Create GPU Node Pool (Cloud Providers)

**AWS (EKS)**:

```bash
eksctl create nodegroup \
  --cluster portalis-cluster \
  --name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5 \
  --node-ami-family Ubuntu2004 \
  --node-labels nvidia.com/gpu=true
```

**GCP (GKE)**:

```bash
gcloud container node-pools create gpu-pool \
  --cluster portalis-cluster \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5
```

**Azure (AKS)**:

```bash
az aks nodepool add \
  --cluster-name portalis-cluster \
  --resource-group portalis-rg \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5
```

### Verify GPU Availability

```bash
# Run GPU test pod
kubectl run gpu-test \
  --image=nvidia/cuda:12.0-base \
  --restart=Never \
  --rm -it \
  -- nvidia-smi

# Expected output shows GPU information
```

---

## Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

**Create HPA**:

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: portalis-hpa
  namespace: portalis
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: portalis
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
```

Apply HPA:

```bash
kubectl apply -f hpa.yaml

# Monitor HPA
kubectl get hpa -n portalis -w
```

### Cluster Autoscaler

Enable cluster autoscaler for automatic node scaling:

**AWS (EKS)**:

```bash
# Install cluster autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Add IAM policy for autoscaler
# Set auto-scaling group tags
```

**GCP (GKE)**:

```bash
# Enable autoscaling (already configured during node pool creation)
gcloud container clusters update portalis-cluster \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10
```

### Vertical Pod Autoscaler (VPA)

For automatic resource adjustment:

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: portalis-vpa
  namespace: portalis
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: portalis
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: portalis
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 4
        memory: 8Gi
```

---

## High Availability

### Multi-Zone Deployment

Spread pods across availability zones:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: portalis
spec:
  replicas: 3
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - portalis
              topologyKey: topology.kubernetes.io/zone
```

### Pod Disruption Budget

Ensure minimum availability during updates:

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: portalis-pdb
  namespace: portalis
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: portalis
```

Apply PDB:

```bash
kubectl apply -f pdb.yaml
```

### Health Checks

Configure liveness and readiness probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 2
```

---

## Monitoring Integration

### Prometheus

**Install Prometheus**:

```bash
# Using Prometheus Operator
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

**ServiceMonitor**:

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: portalis
  namespace: portalis
spec:
  selector:
    matchLabels:
      app: portalis
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Grafana Dashboards

Import Portalis dashboards:

```bash
# Get Grafana admin password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Open browser: http://localhost:3000
# Import dashboard ID: 12345 (Portalis Translation Metrics)
```

---

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n portalis

# Describe pod
kubectl describe pod <pod-name> -n portalis

# Check logs
kubectl logs <pod-name> -n portalis

# Common issues:
# - ImagePullBackOff: Check image registry credentials
# - CrashLoopBackOff: Check application logs
# - Pending: Check resource availability
```

### GPU Not Available

```bash
# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia

# Check GPU allocation
kubectl describe node <gpu-node>

# Test GPU
kubectl run gpu-test --image=nvidia/cuda:12.0-base --rm -it -- nvidia-smi
```

### Performance Issues

```bash
# Check resource usage
kubectl top pods -n portalis
kubectl top nodes

# Check HPA status
kubectl get hpa -n portalis

# Check metrics
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes
```

---

## See Also

- [Docker Compose Deployment](docker-compose.md)
- [Performance Tuning](../performance.md)
- [Security Guide](../security.md)
- [Monitoring Setup](../monitoring.md)
