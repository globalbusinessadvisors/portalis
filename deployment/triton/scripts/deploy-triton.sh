#!/bin/bash
# Triton Inference Server Deployment Script for Portalis
# Deploys Triton with models, monitoring, and auto-scaling

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-portalis-deployment}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-portalis-monitoring}"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.01-py3}"
REPLICAS="${REPLICAS:-3}"
GPU_PER_POD="${GPU_PER_POD:-2}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-/models}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    # Check helm (optional)
    if ! command -v helm &> /dev/null; then
        log_warn "helm not found. Some features may not work."
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check NVIDIA device plugin
    if ! kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset &> /dev/null; then
        log_warn "NVIDIA device plugin not found. GPU scheduling may not work."
    fi

    log_info "Prerequisites check passed"
}

# Create namespaces
create_namespaces() {
    log_info "Creating namespaces..."

    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace "$MONITORING_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Label namespaces
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite
    kubectl label namespace "$MONITORING_NAMESPACE" name="$MONITORING_NAMESPACE" --overwrite

    log_info "Namespaces created"
}

# Deploy model repository storage
deploy_model_storage() {
    log_info "Deploying model repository storage..."

    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: triton-model-repository
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-storage
  namespace: $MONITORING_NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
EOF

    log_info "Storage provisioned"
}

# Deploy Triton server StatefulSet
deploy_triton_server() {
    log_info "Deploying Triton Inference Server..."

    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: triton-headless
  namespace: $NAMESPACE
spec:
  clusterIP: None
  selector:
    app: triton-server
  ports:
    - name: http
      port: 8000
    - name: grpc
      port: 8001
    - name: metrics
      port: 8002
---
apiVersion: v1
kind: Service
metadata:
  name: triton-service
  namespace: $NAMESPACE
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8002"
    prometheus.io/path: "/metrics"
spec:
  selector:
    app: triton-server
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: grpc
      port: 8001
      targetPort: 8001
    - name: metrics
      port: 8002
      targetPort: 8002
  type: ClusterIP
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: triton-server
  namespace: $NAMESPACE
spec:
  serviceName: triton-headless
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: triton-server
  template:
    metadata:
      labels:
        app: triton-server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8002"
    spec:
      priorityClassName: triton-high-priority
      containers:
      - name: triton-server
        image: $TRITON_IMAGE
        command: ["/opt/tritonserver/bin/tritonserver"]
        args:
          - --model-repository=$MODEL_REPOSITORY
          - --strict-model-config=false
          - --log-verbose=1
          - --allow-gpu-metrics=true
          - --allow-cpu-metrics=true
          - --metrics-port=8002
          - --backend-directory=/opt/tritonserver/backends
          - --backend-config=python,shm-default-byte-size=16777216
        ports:
          - containerPort: 8000
            name: http
          - containerPort: 8001
            name: grpc
          - containerPort: 8002
            name: metrics
        env:
          - name: CUDA_VISIBLE_DEVICES
            value: "0,1"
          - name: TRITON_SERVER_CPU_ONLY
            value: "0"
          - name: OMP_NUM_THREADS
            value: "8"
        resources:
          requests:
            cpu: 4000m
            memory: 16Gi
            nvidia.com/gpu: $GPU_PER_POD
          limits:
            cpu: 8000m
            memory: 32Gi
            nvidia.com/gpu: $GPU_PER_POD
        volumeMounts:
          - name: model-repository
            mountPath: $MODEL_REPOSITORY
          - name: shm
            mountPath: /dev/shm
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.3-3.1.4-ubuntu20.04
        ports:
          - containerPort: 9400
            name: dcgm-metrics
        env:
          - name: DCGM_EXPORTER_LISTEN
            value: ":9400"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
        securityContext:
          capabilities:
            add: ["SYS_ADMIN"]
      volumes:
        - name: model-repository
          persistentVolumeClaim:
            claimName: triton-model-repository
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
EOF

    log_info "Triton server deployed"
}

# Copy models to repository
copy_models() {
    log_info "Copying models to repository..."

    # Wait for at least one pod to be ready
    kubectl wait --for=condition=ready pod -l app=triton-server -n "$NAMESPACE" --timeout=300s || {
        log_warn "Timeout waiting for pod. Continuing anyway..."
    }

    POD=$(kubectl get pod -n "$NAMESPACE" -l app=triton-server -o jsonpath='{.items[0].metadata.name}')

    if [ -n "$POD" ]; then
        log_info "Copying models to pod: $POD"

        # Copy model files
        kubectl cp ../models "$NAMESPACE/$POD:$MODEL_REPOSITORY/" || log_warn "Failed to copy some models"

        # Reload models
        kubectl exec -n "$NAMESPACE" "$POD" -- curl -X POST http://localhost:8000/v2/repository/index || {
            log_warn "Failed to reload models"
        }
    else
        log_warn "No pods ready yet. Models need to be copied manually."
    fi

    log_info "Models copied"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."

    # Apply Prometheus config
    kubectl apply -f ../monitoring/prometheus-config.yaml

    # Deploy Grafana
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: $MONITORING_NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
          - containerPort: 3000
        env:
          - name: GF_SECURITY_ADMIN_PASSWORD
            value: "admin"
          - name: GF_INSTALL_PLUGINS
            value: "grafana-piechart-panel"
        volumeMounts:
          - name: grafana-storage
            mountPath: /var/lib/grafana
          - name: grafana-dashboards
            mountPath: /etc/grafana/provisioning/dashboards
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
      volumes:
        - name: grafana-storage
          emptyDir: {}
        - name: grafana-dashboards
          configMap:
            name: grafana-dashboards
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: $MONITORING_NAMESPACE
spec:
  selector:
    app: grafana
  ports:
    - port: 3000
      targetPort: 3000
  type: LoadBalancer
EOF

    # Create Grafana dashboard ConfigMap
    kubectl create configmap grafana-dashboards \
        --from-file=../monitoring/grafana-dashboards.json \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_info "Monitoring deployed"
}

# Deploy load balancer and autoscaling
deploy_scaling() {
    log_info "Deploying load balancer and autoscaling..."

    kubectl apply -f ../configs/load_balancer.yaml
    kubectl apply -f ../configs/autoscaling.yaml

    log_info "Scaling configuration applied"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."

    # Check Triton pods
    log_info "Checking Triton pods..."
    kubectl get pods -n "$NAMESPACE" -l app=triton-server

    # Check services
    log_info "Checking services..."
    kubectl get svc -n "$NAMESPACE"

    # Test health endpoint
    log_info "Testing health endpoint..."
    kubectl run -it --rm triton-test --image=curlimages/curl --restart=Never -n "$NAMESPACE" -- \
        curl -s http://triton-service:8000/v2/health/live || log_warn "Health check failed"

    # Check HPA
    log_info "Checking HPA..."
    kubectl get hpa -n "$NAMESPACE"

    # Get monitoring URLs
    GRAFANA_IP=$(kubectl get svc grafana -n "$MONITORING_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    if [ "$GRAFANA_IP" != "pending" ]; then
        log_info "Grafana dashboard: http://$GRAFANA_IP:3000 (admin/admin)"
    else
        log_info "Grafana LoadBalancer IP pending..."
    fi

    log_info "Deployment verification complete"
}

# Main deployment flow
main() {
    log_info "Starting Portalis Triton Deployment..."

    check_prerequisites
    create_namespaces
    deploy_model_storage
    deploy_triton_server
    copy_models
    deploy_monitoring
    deploy_scaling
    verify_deployment

    log_info "Deployment complete!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Access Grafana dashboard for monitoring"
    log_info "2. Test translation API with: kubectl port-forward -n $NAMESPACE svc/triton-service 8000:8000"
    log_info "3. Run integration tests: ./run-tests.sh"
}

# Run main function
main "$@"
