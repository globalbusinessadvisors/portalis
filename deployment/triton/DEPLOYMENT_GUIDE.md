# Portalis Triton Deployment Guide

## Executive Summary

This guide provides a comprehensive walkthrough for deploying the Portalis translation service using NVIDIA Triton Inference Server. The deployment supports GPU-accelerated batch and interactive translation of Python code to Rust with auto-scaling, monitoring, and production-grade reliability.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Deployment Preparation](#pre-deployment-preparation)
3. [Deployment Steps](#deployment-steps)
4. [Configuration Options](#configuration-options)
5. [Post-Deployment Validation](#post-deployment-validation)
6. [Operational Procedures](#operational-procedures)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Requirements

### Hardware

**Minimum Configuration**:
- 3 Kubernetes nodes with NVIDIA GPUs
- 2x NVIDIA A100/V100 GPUs per node
- 64GB RAM per node
- 500GB SSD storage per node
- 10Gbps network connectivity

**Recommended Configuration**:
- 5-10 Kubernetes nodes with NVIDIA GPUs
- 4x NVIDIA A100 80GB GPUs per node
- 256GB RAM per node
- 1TB NVMe SSD per node
- 25Gbps network with RDMA support

### Software

**Required**:
- Kubernetes 1.24+
- NVIDIA GPU Operator 23.6+
- NVIDIA Device Plugin
- Container Runtime with NVIDIA support (containerd/cri-o)
- kubectl 1.24+
- Persistent Volume provisioner (with ReadWriteMany support)

**Optional**:
- Helm 3.x
- NVIDIA DGX Cloud access
- NGC CLI tools
- ArgoCD for GitOps deployment

### Network

**Required Ports**:
- 8000 (Triton HTTP API)
- 8001 (Triton gRPC API)
- 8002 (Prometheus metrics)
- 9090 (Prometheus UI)
- 3000 (Grafana UI)

**Bandwidth**:
- Minimum: 1Gbps per node
- Recommended: 10Gbps+ for batch operations

---

## Pre-Deployment Preparation

### 1. Verify GPU Availability

```bash
# Check GPU nodes
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

# Verify GPU operator
kubectl get pods -n gpu-operator-resources

# Test GPU scheduling
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.0-base
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

kubectl logs gpu-test
kubectl delete pod gpu-test
```

### 2. Configure Storage

```bash
# Create storage class (example for AWS EFS)
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-xxxxxxxx
  directoryPerms: "700"
volumeBindingMode: Immediate
allowVolumeExpansion: true
EOF

# Verify storage class
kubectl get storageclass fast-ssd
```

### 3. Create Namespaces

```bash
# Create namespaces
kubectl create namespace portalis-deployment
kubectl create namespace portalis-monitoring

# Label namespaces
kubectl label namespace portalis-deployment environment=production
kubectl label namespace portalis-monitoring monitoring=enabled
```

### 4. Set Up RBAC

```bash
# Create service account for Triton
kubectl create serviceaccount triton-sa -n portalis-deployment

# Create role for model management
cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: triton-model-manager
  namespace: portalis-deployment
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
EOF

# Bind role
kubectl create rolebinding triton-sa-binding \
  --role=triton-model-manager \
  --serviceaccount=portalis-deployment:triton-sa \
  -n portalis-deployment
```

---

## Deployment Steps

### Step 1: Clone Repository and Navigate

```bash
git clone https://github.com/portalis/portalis.git
cd portalis/deployment/triton
```

### Step 2: Configure Environment

```bash
# Copy and edit environment file
cat > .env <<EOF
NAMESPACE=portalis-deployment
MONITORING_NAMESPACE=portalis-monitoring
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3
REPLICAS=3
GPU_PER_POD=2
MODEL_REPOSITORY=/models
EOF

# Source environment
source .env
```

### Step 3: Build Custom Triton Image (Optional)

```bash
# Login to NGC (if using custom image)
docker login nvcr.io
# Username: $oauthtoken
# Password: <YOUR_NGC_API_KEY>

# Build custom image
docker build -t portalis-triton:latest .

# Tag and push to your registry
docker tag portalis-triton:latest your-registry.com/portalis-triton:latest
docker push your-registry.com/portalis-triton:latest

# Update TRITON_IMAGE in .env
export TRITON_IMAGE=your-registry.com/portalis-triton:latest
```

### Step 4: Deploy Triton Infrastructure

```bash
# Run automated deployment
cd scripts
chmod +x deploy-triton.sh
./deploy-triton.sh
```

The script will:
1. Verify prerequisites
2. Create namespaces and RBAC
3. Provision storage
4. Deploy Triton StatefulSet
5. Copy model files
6. Deploy monitoring stack
7. Configure auto-scaling
8. Verify deployment

**Expected Output**:
```
[INFO] Starting Portalis Triton Deployment...
[INFO] Checking prerequisites...
[INFO] Prerequisites check passed
[INFO] Creating namespaces...
[INFO] Namespaces created
[INFO] Deploying model repository storage...
[INFO] Storage provisioned
[INFO] Deploying Triton Inference Server...
[INFO] Triton server deployed
[INFO] Copying models to repository...
[INFO] Models copied
[INFO] Deploying monitoring stack...
[INFO] Monitoring deployed
[INFO] Deploying load balancer and autoscaling...
[INFO] Scaling configuration applied
[INFO] Verifying deployment...
[INFO] Deployment complete!
```

### Step 5: Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n portalis-deployment
# Expected: All triton-server-* pods in Running state

# Check services
kubectl get svc -n portalis-deployment
# Expected: triton-service and triton-headless

# Check HPA
kubectl get hpa -n portalis-deployment
# Expected: triton-server-hpa with current metrics

# Check PVCs
kubectl get pvc -n portalis-deployment
# Expected: triton-model-repository in Bound state
```

---

## Configuration Options

### Scaling Configuration

**Adjust Replicas**:
```bash
# Edit deploy-triton.sh before running
export REPLICAS=5  # Increase to 5 replicas

# Or scale after deployment
kubectl scale statefulset triton-server -n portalis-deployment --replicas=5
```

**Adjust GPU Allocation**:
```bash
# Edit deploy-triton.sh
export GPU_PER_POD=4  # Use 4 GPUs per pod
```

### Model Configuration

**Enable/Disable Models**:
```bash
# Edit model config.pbtxt
# Set instance_group count to 0 to disable
vim models/translation_model/config.pbtxt

# Reload configuration
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl -X POST http://localhost:8000/v2/repository/index
```

**Adjust Batch Sizes**:
```bash
# Edit config.pbtxt
vim models/translation_model/config.pbtxt

# Change max_batch_size and preferred_batch_size
max_batch_size: 64
dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]
}
```

### Resource Limits

**CPU and Memory**:
```yaml
# Edit StatefulSet
resources:
  requests:
    cpu: 8000m        # 8 cores
    memory: 32Gi      # 32 GB
  limits:
    cpu: 16000m       # 16 cores
    memory: 64Gi      # 64 GB
```

---

## Post-Deployment Validation

### 1. Health Checks

```bash
# Check liveness
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8000/v2/health/live

# Check readiness
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8000/v2/health/ready

# List loaded models
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8000/v2/models/translation_model
```

### 2. Functional Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run integration tests
cd tests
export TRITON_URL=<TRITON_SERVICE_IP>:8000
pytest test_triton_integration.py -v

# Expected: All tests pass
```

### 3. Performance Validation

```bash
# Run benchmark tests
pytest test_triton_integration.py::TestPerformance -v --benchmark

# Load test with locust (optional)
pip install locust
locust -f load_test.py --host=http://<TRITON_IP>:8000
```

### 4. Monitoring Validation

```bash
# Access Grafana
kubectl port-forward -n portalis-monitoring svc/grafana 3000:3000

# Open browser: http://localhost:3000
# Login: admin/admin
# Check dashboards are loading metrics
```

---

## Operational Procedures

### Starting and Stopping

**Stop Triton Servers**:
```bash
kubectl scale statefulset triton-server -n portalis-deployment --replicas=0
```

**Start Triton Servers**:
```bash
kubectl scale statefulset triton-server -n portalis-deployment --replicas=3
```

### Rolling Updates

**Update Triton Image**:
```bash
kubectl set image statefulset/triton-server \
  triton-server=nvcr.io/nvidia/tritonserver:24.02-py3 \
  -n portalis-deployment

# Monitor rollout
kubectl rollout status statefulset/triton-server -n portalis-deployment
```

### Backup and Restore

**Backup Models**:
```bash
# Create backup
kubectl exec -n portalis-deployment triton-server-0 -- \
  tar czf /tmp/models-backup-$(date +%Y%m%d).tar.gz /models

# Copy to local
kubectl cp portalis-deployment/triton-server-0:/tmp/models-backup-*.tar.gz \
  ./backups/
```

**Restore Models**:
```bash
# Upload backup
kubectl cp ./backups/models-backup-20250101.tar.gz \
  portalis-deployment/triton-server-0:/tmp/

# Restore
kubectl exec -n portalis-deployment triton-server-0 -- \
  tar xzf /tmp/models-backup-20250101.tar.gz -C /
```

---

## Performance Optimization

### GPU Utilization

**Monitor GPU Usage**:
```bash
# Check GPU metrics
kubectl exec -n portalis-deployment triton-server-0 -- nvidia-smi

# View Grafana GPU dashboard
# http://localhost:3000/d/triton-gpu
```

**Optimize for High Utilization**:
1. Increase batch sizes
2. Reduce max_queue_delay for faster batching
3. Add more model instances
4. Enable CUDA graphs

### Latency Optimization

**For Low Latency**:
```yaml
# Reduce queue delays
dynamic_batching {
  max_queue_delay_microseconds: 10000  # 10ms
  preserve_ordering: true
}

# Increase instance count
instance_group [
  { count: 8, kind: KIND_GPU }
]
```

### Throughput Optimization

**For High Throughput**:
```yaml
# Larger batch sizes
max_batch_size: 128

# Longer queue delays
dynamic_batching {
  max_queue_delay_microseconds: 500000  # 500ms
  preferred_batch_size: [ 64, 128 ]
}
```

---

## Troubleshooting Guide

### Issue: Pods Stuck in Pending

**Diagnosis**:
```bash
kubectl describe pod triton-server-0 -n portalis-deployment | grep Events -A 20
```

**Common Causes**:
1. Insufficient GPU resources
2. Storage PVC not bound
3. Node selector mismatch

**Solutions**:
```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check PVC
kubectl get pvc -n portalis-deployment

# Add GPU nodes or reduce GPU requests
```

### Issue: Models Not Loading

**Diagnosis**:
```bash
kubectl logs triton-server-0 -n portalis-deployment | grep ERROR
```

**Common Causes**:
1. Model files not copied
2. Invalid model configuration
3. Missing Python dependencies

**Solutions**:
```bash
# Re-copy models
kubectl cp models portalis-deployment/triton-server-0:/

# Validate config
kubectl exec -n portalis-deployment triton-server-0 -- \
  cat /models/translation_model/config.pbtxt

# Reload models
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl -X POST http://localhost:8000/v2/repository/index
```

### Issue: High Latency

**Diagnosis**:
```bash
# Check queue metrics
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8002/metrics | grep queue_duration
```

**Solutions**:
1. Scale up replicas
2. Increase GPU count per pod
3. Optimize batch sizes
4. Check GPU utilization

### Issue: Out of Memory

**Diagnosis**:
```bash
# Check memory usage
kubectl top pod triton-server-0 -n portalis-deployment

# Check GPU memory
kubectl exec -n portalis-deployment triton-server-0 -- nvidia-smi
```

**Solutions**:
```bash
# Increase shared memory
# Edit StatefulSet shm volume size

# Reduce batch sizes
# Edit model config max_batch_size

# Add memory limits
resources:
  limits:
    memory: 64Gi
```

---

## Support and Resources

### Documentation
- [Triton Inference Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [NeMo Framework Docs](https://docs.nvidia.com/deeplearning/nemo/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)

### Community
- NVIDIA Developer Forums
- Triton GitHub Issues
- Portalis Community Slack

### Professional Support
- NVIDIA Enterprise Support
- DGX Cloud Support
- Portalis Enterprise Support

---

## Appendix: Quick Reference

### Common Commands

```bash
# View logs
kubectl logs -f triton-server-0 -n portalis-deployment

# Execute in pod
kubectl exec -it triton-server-0 -n portalis-deployment -- bash

# Port forward for local access
kubectl port-forward -n portalis-deployment svc/triton-service 8000:8000

# Scale replicas
kubectl scale statefulset triton-server --replicas=5 -n portalis-deployment

# Check HPA status
kubectl get hpa -n portalis-deployment -w

# View metrics
kubectl exec -n portalis-deployment triton-server-0 -- \
  curl http://localhost:8002/metrics
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| NAMESPACE | portalis-deployment | Kubernetes namespace |
| TRITON_IMAGE | nvcr.io/nvidia/tritonserver:24.01-py3 | Triton image |
| REPLICAS | 3 | Number of Triton replicas |
| GPU_PER_POD | 2 | GPUs per pod |
| MODEL_REPOSITORY | /models | Model repository path |

---

**Version**: 1.0
**Last Updated**: 2025-10-03
**Author**: Portalis Deployment Team
