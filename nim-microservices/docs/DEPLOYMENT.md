# Deployment Guide

Comprehensive guide for deploying Portalis NIM microservice in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Deployment Options](#deployment-options)
4. [Production Configuration](#production-configuration)
5. [Post-Deployment](#post-deployment)
6. [Operational Procedures](#operational-procedures)

## Prerequisites

### Hardware Requirements

**Minimum (Development):**
- 1x NVIDIA GPU (T4 or better)
- 8 CPU cores
- 16 GB RAM
- 100 GB storage

**Recommended (Production):**
- 3+ NVIDIA GPUs (A100 or better)
- 16+ CPU cores per node
- 32+ GB RAM per node
- 500+ GB SSD storage
- 10 Gbps network

### Software Requirements

- Kubernetes 1.24+
- NVIDIA GPU Operator 23.3.0+
- Helm 3.8+
- kubectl 1.24+
- Docker 20.10+ or containerd 1.6+

### Cloud Platforms

Supported platforms:
- **NVIDIA DGX Cloud** (recommended)
- AWS EKS with GPU nodes
- Google GKE with GPU nodes
- Azure AKS with GPU nodes
- On-premises Kubernetes with NVIDIA GPUs

## Infrastructure Setup

### 1. Kubernetes Cluster

#### DGX Cloud

```bash
# Create DGX Cloud cluster
ngc cluster create portalis-production \
  --type dgx-a100 \
  --nodes 3 \
  --gpu-type a100 \
  --gpus-per-node 8

# Get credentials
ngc cluster get-credentials portalis-production
```

#### AWS EKS

```bash
# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name portalis-production \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.8xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 20

# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
```

#### GKE

```bash
# Create GKE cluster
gcloud container clusters create portalis-production \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 20

# Install NVIDIA driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### 2. Install NVIDIA GPU Operator

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true

# Verify installation
kubectl get pods -n gpu-operator
```

### 3. Storage Setup

#### Create StorageClass for Models

```yaml
# fast-storage-class.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-storage
provisioner: kubernetes.io/aws-ebs  # Or appropriate provisioner
parameters:
  type: gp3
  iops: "16000"
  throughput: "1000"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

```bash
kubectl apply -f fast-storage-class.yaml
```

### 4. Networking Setup

#### Install Ingress Controller

```bash
# Install NGINX Ingress
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer

# Install cert-manager for TLS
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

#### Configure DNS

Point your domain to the LoadBalancer:
```bash
# Get LoadBalancer IP
kubectl get svc -n ingress-nginx ingress-nginx-controller

# Create A record
portalis-nim.yourdomain.com -> <LOADBALANCER-IP>
```

## Deployment Options

### Option 1: Helm Deployment (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/portalis/portalis.git
cd portalis/nim-microservices

# 2. Create namespace
kubectl create namespace portalis

# 3. Create secrets
kubectl create secret generic portalis-api-keys \
  --from-literal=keys="client1:$(openssl rand -hex 32)" \
  --namespace portalis

# 4. Upload models to PVC
kubectl create -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: portalis-models-pvc
  namespace: portalis
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: fast-storage
  resources:
    requests:
      storage: 50Gi
EOF

# Copy models
kubectl run -it --rm model-loader \
  --image=busybox \
  --namespace portalis \
  --overrides='{"spec":{"containers":[{"name":"model-loader","image":"busybox","command":["sh"],"volumeMounts":[{"mountPath":"/models","name":"models"}]}],"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"portalis-models-pvc"}}]}}' \
  -- sh

# In the pod:
# wget https://models.portalis.dev/nemo_translation.nemo -O /models/nemo_translation.nemo
# exit

# 5. Create values file
cat > production-values.yaml <<EOF
replicaCount: 3

image:
  repository: portalis-nim
  tag: "1.0.0"

ingress:
  enabled: true
  hosts:
    - host: portalis-nim.yourdomain.com
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

env:
  ENVIRONMENT: production
  ENABLE_AUTH: "true"
  TRITON_URL: triton-service.portalis.svc.cluster.local:8000

monitoring:
  enabled: true
EOF

# 6. Install with Helm
helm install portalis-nim ./helm \
  --namespace portalis \
  --values production-values.yaml \
  --wait

# 7. Verify deployment
kubectl get pods -n portalis -l app=portalis-nim
```

### Option 2: kubectl Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/base/ --namespace portalis

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s \
  deployment/portalis-nim --namespace portalis
```

### Option 3: GitOps with ArgoCD

```yaml
# application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: portalis-nim
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/portalis/portalis.git
    targetRevision: main
    path: nim-microservices/helm
    helm:
      valueFiles:
        - values.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: portalis
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Production Configuration

### Security Hardening

#### 1. Enable Authentication

```yaml
env:
  ENABLE_AUTH: "true"
  API_KEYS: "{{ .Values.secrets.apiKeys }}"
```

#### 2. Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: portalis-nim-netpol
spec:
  podSelector:
    matchLabels:
      app: portalis-nim
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 8000  # Triton
```

#### 3. Pod Security Standards

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: portalis
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### High Availability

#### 1. Multi-Zone Deployment

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app: portalis-nim
      topologyKey: topology.kubernetes.io/zone
```

#### 2. PodDisruptionBudget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: portalis-nim-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: portalis-nim
```

### Monitoring Setup

#### 1. Install Prometheus Stack

```bash
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

#### 2. Configure ServiceMonitor

Already included in Helm chart when `monitoring.enabled=true`.

#### 3. Import Grafana Dashboards

```bash
kubectl create configmap portalis-dashboards \
  --from-file=k8s/monitoring/grafana-dashboards/ \
  --namespace monitoring
```

## Post-Deployment

### Verification

```bash
# 1. Check pod health
kubectl get pods -n portalis -l app=portalis-nim

# 2. Check services
kubectl get svc -n portalis

# 3. Test health endpoint
kubectl run -it --rm test-curl \
  --image=curlimages/curl \
  --restart=Never \
  -- curl http://portalis-nim-api.portalis:8000/health

# 4. Check metrics
kubectl port-forward svc/portalis-nim-api 9090:9090 -n portalis
curl http://localhost:9090/metrics

# 5. Check GPU allocation
kubectl describe node <node-name> | grep nvidia.com/gpu
```

### Load Testing

```bash
# Install k6 or use Apache Bench
kubectl run -it --rm load-test \
  --image=loadimpact/k6 \
  --restart=Never \
  -- run - <<EOF
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 50 },
    { duration: '5m', target: 50 },
    { duration: '2m', target: 0 },
  ],
};

export default function () {
  let res = http.post('http://portalis-nim-api.portalis:8000/api/v1/translation/translate',
    JSON.stringify({
      python_code: 'def test(): pass',
      mode: 'fast'
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );
  check(res, { 'status is 200': (r) => r.status === 200 });
}
EOF
```

## Operational Procedures

### Updating the Service

```bash
# 1. Update image
helm upgrade portalis-nim ./helm \
  --namespace portalis \
  --set image.tag=1.1.0 \
  --wait

# 2. Rollback if needed
helm rollback portalis-nim --namespace portalis
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment portalis-nim --replicas=10 -n portalis

# Update HPA
kubectl patch hpa portalis-nim-hpa -n portalis \
  --patch '{"spec":{"maxReplicas":30}}'
```

### Backup and Recovery

```bash
# Backup Helm release
helm get values portalis-nim -n portalis > backup-values.yaml

# Backup models
kubectl create job backup-models --from=cronjob/model-backup -n portalis
```

### Monitoring Alerts

Configure alerts in Prometheus:

```yaml
groups:
- name: portalis-nim
  rules:
  - alert: HighErrorRate
    expr: rate(nim_failed_requests[5m]) > 0.05
    for: 5m
    annotations:
      summary: "High error rate detected"
  - alert: HighLatency
    expr: histogram_quantile(0.95, nim_request_duration_seconds) > 1.0
    for: 5m
    annotations:
      summary: "High latency detected"
```

### Disaster Recovery

See [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) for detailed procedures.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## Cost Optimization

- Use Spot/Preemptible instances for non-critical workloads
- Configure aggressive scale-down policies during low usage
- Use GPU sharing for development environments
- Monitor and optimize batch sizes
- Enable model caching to reduce cold starts

## Support

For deployment assistance:
- Email: ops@portalis.dev
- Slack: #portalis-ops
- Emergency: ops-oncall@portalis.dev
