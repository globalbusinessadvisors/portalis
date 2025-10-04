# Load Testing for Portalis Rust Transpiler

This directory contains load testing scripts to validate the performance, scalability, and auto-scaling capabilities of the Portalis Rust Transpiler NIM microservices.

## Overview

We provide two load testing frameworks:

1. **Locust** (Python) - For flexible, programmable load tests with web UI
2. **K6** (JavaScript) - For performance testing with detailed metrics

## Prerequisites

### For Locust

```bash
pip install locust
```

### For K6

```bash
# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# macOS
brew install k6

# Windows
choco install k6
```

## Running Tests

### Locust Tests

#### 1. Normal Load Test (50 users)

```bash
locust -f load_test.py \
  --users 50 \
  --spawn-rate 10 \
  --run-time 10m \
  --host http://localhost:8000
```

#### 2. Stress Test (200 users)

```bash
locust -f load_test.py \
  --users 200 \
  --spawn-rate 50 \
  --run-time 5m \
  --host http://localhost:8000 \
  StressTestUser
```

#### 3. Spike Test (500 users)

```bash
locust -f load_test.py \
  --users 500 \
  --spawn-rate 100 \
  --run-time 2m \
  --host http://localhost:8000
```

#### 4. Web UI Mode

```bash
locust -f load_test.py --host http://localhost:8000
# Open http://localhost:8089 in browser
```

#### 5. Headless with CSV Output

```bash
locust -f load_test.py \
  --users 100 \
  --spawn-rate 20 \
  --run-time 5m \
  --host http://localhost:8000 \
  --headless \
  --csv=results/locust_results \
  --html=results/locust_report.html
```

### K6 Tests

#### 1. Standard Load Test

```bash
k6 run k6_load_test.js
```

#### 2. With Custom Target

```bash
BASE_URL=http://transpiler.portalis.dev k6 run k6_load_test.js
```

#### 3. Cloud Run (K6 Cloud)

```bash
k6 cloud k6_load_test.js
```

#### 4. With JSON Output

```bash
k6 run --out json=results/k6_results.json k6_load_test.js
```

#### 5. With InfluxDB Output

```bash
k6 run --out influxdb=http://localhost:8086/k6 k6_load_test.js
```

## Test Scenarios

### 1. Normal Load Test

- **Users:** 50 concurrent users
- **Duration:** 10 minutes
- **Ramp-up:** Linear over 1 minute
- **Purpose:** Validate normal operation and baseline metrics

**Expected Results:**
- P95 latency < 2s
- P99 latency < 5s
- Error rate < 1%
- Throughput > 100 req/s

### 2. Stress Test

- **Users:** 200 concurrent users
- **Duration:** 5 minutes
- **Ramp-up:** Aggressive (50 users/sec)
- **Purpose:** Test system limits and degradation patterns

**Expected Results:**
- P95 latency < 5s
- P99 latency < 10s
- Error rate < 5%
- Auto-scaling triggers at ~70% CPU

### 3. Spike Test

- **Users:** 500 concurrent users
- **Duration:** 2 minutes
- **Ramp-up:** Very aggressive (100 users/sec)
- **Purpose:** Test rapid scaling and recovery

**Expected Results:**
- System remains operational
- Auto-scaling responds within 60s
- No cascading failures
- Graceful degradation if limits reached

### 4. Soak Test

- **Users:** 30 concurrent users
- **Duration:** 1 hour
- **Ramp-up:** Slow (5 min)
- **Purpose:** Detect memory leaks and resource exhaustion

**Expected Results:**
- Stable memory usage
- No performance degradation over time
- Consistent response times
- No connection pool exhaustion

## Monitoring During Tests

### 1. Watch Kubernetes Pods

```bash
watch -n 2 kubectl get pods -n portalis-deployment
```

### 2. Monitor HPA Status

```bash
watch -n 5 kubectl get hpa -n portalis-deployment
```

### 3. Check GPU Utilization

```bash
kubectl exec -it -n portalis-deployment portalis-rust-transpiler-0 -- nvidia-smi
```

### 4. View Metrics

```bash
# Prometheus
kubectl port-forward -n portalis-monitoring svc/prometheus 9090:9090
# Open http://localhost:9090

# Grafana
kubectl port-forward -n portalis-monitoring svc/grafana 3000:3000
# Open http://localhost:3000
```

### 5. Stream Logs

```bash
kubectl logs -f -n portalis-deployment -l app=portalis-rust-transpiler --tail=100
```

## Analyzing Results

### Locust Results

Locust generates:
- **CSV files:** `results/locust_results_stats.csv`, `results/locust_results_failures.csv`
- **HTML report:** `results/locust_report.html`

Key metrics to check:
- Request count
- Failure rate
- Average/median/P95/P99 response times
- Requests per second
- Failure distribution

### K6 Results

K6 outputs metrics to stdout by default. Key metrics:

```
http_req_duration..............: avg=XXXms p(95)=XXXms
http_req_failed................: X.XX%
http_reqs......................: XXXX/s
```

For JSON output analysis:

```bash
jq '.metrics.http_req_duration' results/k6_results.json
```

## Auto-scaling Validation

### Expected Behavior

1. **Scale Up Trigger:**
   - CPU > 70% for 30s
   - GPU > 75% for 30s
   - Queue depth > 50 for 30s

2. **Scale Up Response:**
   - New pods start within 60s
   - Traffic distributes to new pods
   - Latency decreases

3. **Scale Down Trigger:**
   - CPU < 50% for 5 minutes
   - GPU < 50% for 5 minutes
   - Queue depth < 20 for 5 minutes

4. **Scale Down Response:**
   - Pods terminate gracefully
   - No request failures during scale down
   - Minimum replicas maintained (3)

### Validation Commands

```bash
# Check current replica count
kubectl get deployment portalis-rust-transpiler -n portalis-deployment

# Check HPA status
kubectl describe hpa portalis-rust-transpiler-hpa -n portalis-deployment

# Check events
kubectl get events -n portalis-deployment --sort-by='.lastTimestamp' | grep portalis-rust-transpiler
```

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| P50 Latency | < 500ms | < 1s |
| P95 Latency | < 2s | < 5s |
| P99 Latency | < 5s | < 10s |
| Error Rate | < 0.1% | < 1% |
| Throughput | > 100 req/s | > 50 req/s |
| GPU Utilization | 70-85% | 60-90% |
| Scale-up Time | < 60s | < 120s |
| Scale-down Time | < 300s | < 600s |

## Troubleshooting

### High Error Rate

1. Check pod logs for errors
2. Verify GPU availability
3. Check resource limits
4. Verify Triton connection

### High Latency

1. Check GPU utilization (may be saturated)
2. Verify auto-scaling is working
3. Check network latency
4. Review batch size configuration

### Auto-scaling Not Working

1. Verify metrics server is running
2. Check HPA configuration
3. Verify custom metrics are available
4. Check resource requests/limits

### Connection Timeouts

1. Increase timeout values
2. Check ingress configuration
3. Verify service endpoints
4. Check network policies

## Best Practices

1. **Warm-up:** Run a small load test first to warm caches
2. **Incremental:** Start with small loads and increase gradually
3. **Monitor:** Watch metrics during tests
4. **Document:** Record test parameters and results
5. **Baseline:** Establish baseline metrics before changes
6. **Repeat:** Run tests multiple times for consistency

## Continuous Load Testing

For CI/CD integration:

```bash
# Add to CI pipeline
k6 run --quiet --no-color k6_load_test.js > k6_results.txt
if [ $? -ne 0 ]; then
  echo "Load test failed"
  exit 1
fi
```

## Additional Resources

- [Locust Documentation](https://docs.locust.io/)
- [K6 Documentation](https://k6.io/docs/)
- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Prometheus Metrics](https://prometheus.io/docs/practices/naming/)
