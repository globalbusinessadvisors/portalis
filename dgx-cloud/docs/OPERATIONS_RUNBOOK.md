# Portalis DGX Cloud - Operations Runbook

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Operations](#operations)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Cost Management](#cost-management)
8. [Disaster Recovery](#disaster-recovery)
9. [Scaling](#scaling)
10. [Security](#security)

---

## Overview

### System Components

The Portalis DGX Cloud infrastructure consists of:

- **Ray Cluster**: Distributed compute cluster with head and worker nodes
- **GPU Workers**: NVIDIA DGX nodes (A100/H100) for translation workloads
- **Redis Cluster**: Distributed caching for translation results and embeddings
- **S3 Storage**: Model repository, cache storage, and result archival
- **Monitoring Stack**: Prometheus + Grafana for metrics and alerting
- **Cost Optimizer**: Budget tracking and optimization recommendations

### Target Metrics

- **Cluster Utilization**: >70%
- **Cost per Translation**: <$0.10
- **Scale-up Time**: <5 minutes
- **Job Queue Latency**: <30 seconds
- **Fault Recovery Time**: <2 minutes

---

## Architecture

### Network Topology

```
┌─────────────────────────────────────────────────────┐
│                    Internet                          │
└───────────────────┬─────────────────────────────────┘
                    │
            ┌───────▼────────┐
            │  Load Balancer  │
            └───────┬────────┘
                    │
    ┌───────────────┴───────────────┐
    │         VPC (10.0.0.0/16)     │
    │                                │
    │  ┌──────────────┬──────────┐  │
    │  │ Public Subnet│  NAT GW  │  │
    │  └──────┬───────┴──────────┘  │
    │         │                      │
    │  ┌──────▼──────────────────┐  │
    │  │   Private Subnets        │  │
    │  │                          │  │
    │  │  ┌────────────────────┐ │  │
    │  │  │  Ray Head Node     │ │  │
    │  │  │  (8x A100 80GB)    │ │  │
    │  │  └────────────────────┘ │  │
    │  │                          │  │
    │  │  ┌────────────────────┐ │  │
    │  │  │  Ray Workers (0-8) │ │  │
    │  │  │  (8x A100 80GB ea) │ │  │
    │  │  └────────────────────┘ │  │
    │  │                          │  │
    │  │  ┌────────────────────┐ │  │
    │  │  │  Redis Cluster     │ │  │
    │  │  │  (3 nodes)         │ │  │
    │  │  └────────────────────┘ │  │
    │  └──────────────────────────┘  │
    └─────────────────────────────────┘
                    │
            ┌───────▼────────┐
            │   S3 Buckets   │
            │ Models │ Cache │
            │ Results        │
            └────────────────┘
```

### Data Flow

1. **Job Submission**: Client → Ray Head Node → Task Scheduler
2. **Task Distribution**: Scheduler → Worker Pool → GPU Workers
3. **Translation**: Worker → NeMo Model (cached in S3) → Rust Code
4. **Caching**: Worker → Redis (check/store) → S3 (persistence)
5. **Result Storage**: Worker → S3 Results Bucket → Client notification

---

## Deployment

### Prerequisites

- AWS account with DGX access
- Terraform >= 1.5
- AWS CLI configured
- SSH key pair for cluster access

### Infrastructure Deployment

#### 1. Set up Terraform Backend

```bash
# Create S3 bucket for Terraform state
aws s3 mb s3://portalis-terraform-state --region us-east-1

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name portalis-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

#### 2. Deploy Infrastructure

```bash
cd /workspace/portalis/dgx-cloud/terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan -out=tfplan

# Apply infrastructure
terraform apply tfplan
```

#### 3. Configure Ray Cluster

```bash
# SSH to head node
ssh -i ~/.ssh/portalis_dgx_key ubuntu@<head-node-ip>

# Clone repository
git clone https://github.com/your-org/portalis.git
cd portalis/dgx-cloud

# Install dependencies
pip install -r requirements.txt

# Start Ray head node
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=8
```

#### 4. Deploy Workers

```bash
# Use Ray autoscaler
ray up config/ray_cluster.yaml

# Verify cluster
ray status
```

### Application Deployment

#### 1. Upload NeMo Models to S3

```bash
# Upload translation model
aws s3 cp translation_model.nemo \
  s3://portalis-models-prod/models/translation/latest/model.nemo

# Verify upload
aws s3 ls s3://portalis-models-prod/models/translation/latest/
```

#### 2. Start Services

```bash
# Start metrics collector
python -m dgx_cloud.src.monitoring.metrics_collector &

# Start cost optimizer
python -m dgx_cloud.src.cost.optimizer &

# Start workload manager
python -m dgx_cloud.src.workload.distributed_manager
```

#### 3. Deploy Monitoring

```bash
# Start Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/config/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Start Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana

# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @config/grafana_dashboard.json
```

---

## Operations

### Daily Operations

#### Morning Checks (9 AM)

```bash
# Check cluster health
ray status

# Check GPU utilization
nvidia-smi

# Review overnight jobs
python scripts/job_summary.py --since "24 hours ago"

# Check budget status
python -m dgx_cloud.src.cost.optimizer report --daily
```

#### Capacity Planning (Weekly)

```bash
# Review utilization trends
python scripts/utilization_report.py --period week

# Review cost trends
python -m dgx_cloud.src.cost.optimizer report --weekly

# Plan worker scaling
# If avg utilization > 80% for 3+ days, increase max_workers
# If avg utilization < 40% for 3+ days, decrease min_workers
```

### Job Management

#### Submit Translation Job

```bash
# Interactive job (high priority)
python -m dgx_cloud.examples.submit_job \
  --files /path/to/code/*.py \
  --priority interactive \
  --tenant customer-123

# Batch job (cost-optimized)
python -m dgx_cloud.examples.submit_job \
  --files /path/to/codebase/**/*.py \
  --priority batch \
  --tenant customer-456 \
  --use-spot
```

#### Monitor Job

```bash
# Get job status
python -m dgx_cloud.src.workload.distributed_manager \
  status --job-id job-abc123

# Watch job progress
watch -n 5 'ray list tasks | grep job-abc123'

# View job logs
ray logs --job-id job-abc123
```

#### Cancel Job

```bash
# Cancel specific job
python -m dgx_cloud.src.workload.distributed_manager \
  cancel --job-id job-abc123

# Cancel all jobs for tenant
python -m dgx_cloud.src.workload.distributed_manager \
  cancel --tenant customer-123
```

### Worker Management

#### Add Workers Manually

```bash
# Add 2 workers
ray up config/ray_cluster.yaml --max-workers 10

# Verify workers joined
ray status
```

#### Remove Workers

```bash
# Remove idle workers
ray down config/ray_cluster.yaml --workers-only --keep-min-workers 2

# Force remove specific worker
ray stop --address <worker-node-ip>:6379
```

#### Update Worker Configuration

```bash
# Update Ray config
vim config/ray_cluster.yaml

# Apply changes (rolling update)
ray up config/ray_cluster.yaml --restart-only
```

---

## Monitoring

### Key Dashboards

#### 1. Cluster Overview Dashboard

**URL**: `http://<grafana-host>:3000/d/portalis-overview`

**Key Metrics**:
- Active workers
- Total GPU utilization
- Active/queued jobs
- Hourly cost

#### 2. GPU Performance Dashboard

**Metrics**:
- Per-GPU utilization (%)
- GPU memory usage (GB)
- GPU temperature (°C)
- Power draw (W)

#### 3. Job Performance Dashboard

**Metrics**:
- Job latency (p50, p95, p99)
- Throughput (jobs/hour)
- Queue depth by priority
- Success/failure rates

#### 4. Cost Dashboard

**Metrics**:
- Hourly cost trend
- Cost by instance type
- Cost by tenant
- Budget vs. actual spend

### Alerts

#### Critical Alerts (PagerDuty)

1. **High Error Rate**: >5% errors for 5 minutes
2. **GPU Overheating**: Any GPU >85°C
3. **Budget Exceeded**: Daily budget >100%
4. **Cluster Unavailable**: Head node unreachable

#### Warning Alerts (Slack)

1. **High Utilization**: Avg GPU util >90% for 10 minutes
2. **Long Queue**: Queue depth >50 for 5 minutes
3. **Budget Warning**: Daily budget >80%
4. **Spot Interruption**: Spot instance terminated

### Log Locations

```bash
# Ray logs
/tmp/ray/session_latest/logs/

# Application logs
/var/log/portalis/

# Worker logs
/tmp/ray/session_latest/logs/worker-*.log

# Cost optimizer logs
/var/log/portalis/cost-optimizer.log
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Workers Not Joining Cluster

**Symptoms**:
- `ray status` shows fewer workers than expected
- Workers stuck in "pending" state

**Diagnosis**:
```bash
# Check worker logs
ssh worker-node-1
tail -f /tmp/ray/session_latest/logs/raylet.log

# Check network connectivity
ping <head-node-ip>
telnet <head-node-ip> 6379
```

**Solutions**:
1. Check security group rules (port 6379 open)
2. Verify IAM role has S3 access
3. Check worker instance status in AWS console
4. Restart Ray on worker: `ray stop && ray start --address=<head-ip>:6379`

#### Issue 2: High GPU Memory Usage

**Symptoms**:
- Out-of-memory errors in logs
- Jobs failing with CUDA errors
- GPU memory at 95%+

**Diagnosis**:
```bash
# Check GPU memory
nvidia-smi

# Check process memory
nvidia-smi pmon

# Review job sizes
python scripts/job_memory_report.py
```

**Solutions**:
1. Reduce batch size in translation config
2. Enable GPU memory growth in NeMo config
3. Clear GPU cache between jobs
4. Add more workers to distribute load

#### Issue 3: Slow Translation Performance

**Symptoms**:
- Jobs taking 2x+ longer than expected
- High latency (p95 >5s for small jobs)

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi dmon

# Check CPU bottleneck
htop

# Check network I/O
iftop

# Profile translation
python -m cProfile -o profile.stats worker.py
```

**Solutions**:
1. Increase worker GPU allocation
2. Enable result caching in Redis
3. Optimize model quantization
4. Check for S3 throttling (increase provisioned throughput)

#### Issue 4: Cost Overruns

**Symptoms**:
- Daily cost >2x budget
- Spot instances running as on-demand
- Workers not scaling down

**Diagnosis**:
```bash
# Check current cost
python -m dgx_cloud.src.cost.optimizer report --today

# Check active instances
aws ec2 describe-instances --filters "Name=tag:portalis-cluster,Values=*"

# Check autoscaling config
cat config/resource_allocation.yaml | grep -A 10 auto_scaling
```

**Solutions**:
1. Enable aggressive scale-down (reduce idle timeout to 2 min)
2. Force spot instance usage: `--use-spot-only`
3. Set hard budget limits: `hard_stop_enabled: true`
4. Review and terminate idle instances manually

---

## Cost Management

### Budget Configuration

#### Set Tenant Budget

```python
from dgx_cloud.src.cost.optimizer import CostOptimizer, BudgetConfig

optimizer = CostOptimizer()

budget = BudgetConfig(
    tenant_id="customer-123",
    daily_limit=500.0,      # $500/day
    weekly_limit=2000.0,    # $2000/week
    monthly_limit=5000.0,   # $5000/month
    alert_threshold_pct=0.8,
    hard_stop_enabled=False,
    notification_emails=["ops@portalis.ai"]
)

optimizer.register_budget(budget)
```

### Cost Optimization Strategies

#### 1. Use Spot Instances (70% savings)

```yaml
# config/resource_allocation.yaml
spot_instances:
  enabled: true
  strategy:
    max_price_multiplier: 0.7
    fallback_enabled: true
```

#### 2. Enable Aggressive Auto-Scaling

```yaml
# config/resource_allocation.yaml
auto_scaling:
  scale_down:
    - metric: idle_time_minutes
      threshold: 2  # Scale down after 2 min idle
      min_workers: 0  # Allow scaling to zero
```

#### 3. Right-Size Instances

```bash
# Review utilization
python scripts/instance_sizing_report.py

# Recommendations will show:
# - Oversized instances (util <50%)
# - Undersized instances (util >90%)
# - Suggested instance type changes
```

#### 4. Use Reserved Instances (40% savings)

For sustained workloads, purchase 1-year reserved instances:

```bash
# Review sustained usage
python scripts/reserved_instance_recommendation.py

# Purchase recommendation:
# "Based on 30-day usage, reserve 2x dgxa100.80g instances"
# "Annual savings: $18,000"
```

### Cost Reporting

#### Daily Cost Report

```bash
python -m dgx_cloud.src.cost.optimizer report --daily
```

Output:
```
=== Daily Cost Report (2025-10-03) ===
Total Cost: $127.45

By Instance Type:
  dgxa100.80g (on-demand): $98.50
  dgxa100.40g (spot):      $28.95

By Tenant:
  customer-123: $85.20
  customer-456: $42.25

Recommendations:
  1. Use spot instances: Save $69/day
  2. Reduce idle time: Save $15/day
```

---

## Disaster Recovery

### Backup Strategy

#### 1. Redis Backup (Daily)

```bash
# Automated via ElastiCache
# Retention: 5 days
# Window: 3:00-5:00 AM UTC
```

#### 2. S3 Versioning (Enabled)

All S3 buckets have versioning enabled:
- Models: Indefinite retention
- Results: 90 days
- Cache: 7 days

#### 3. Terraform State Backup

```bash
# Automated via S3 versioning
# Manual backup
aws s3 cp s3://portalis-terraform-state/dgx-cloud/terraform.tfstate \
  s3://portalis-backups/terraform-state-$(date +%Y%m%d).tfstate
```

### Recovery Procedures

#### Scenario 1: Head Node Failure

**Detection**: Head node health check fails

**Recovery**:
```bash
# 1. Launch new head node from AMI
terraform apply -replace=aws_instance.ray_head

# 2. Restore Ray state from S3
aws s3 sync s3://portalis-cluster-state/ray/ /tmp/ray/

# 3. Restart Ray
ray start --head --port=6379

# 4. Workers will auto-reconnect
# 5. Resume jobs from Redis queue

# Expected RTO: 5 minutes
```

#### Scenario 2: Redis Cluster Failure

**Detection**: Redis unavailable for 2+ minutes

**Recovery**:
```bash
# 1. ElastiCache auto-failover (30 seconds)
# 2. Verify failover
aws elasticache describe-replication-groups \
  --replication-group-id portalis-redis

# 3. Update application config if needed
# 4. Warm cache from S3 backup

# Expected RTO: 2 minutes
```

#### Scenario 3: Complete Region Outage

**Detection**: All AWS services unavailable in us-east-1

**Recovery**:
```bash
# 1. Deploy to us-west-2 (disaster recovery region)
cd terraform/
terraform workspace select dr
terraform apply

# 2. Restore models from S3 cross-region replication
# 3. Resume jobs from saved state
# 4. Update DNS to point to DR region

# Expected RTO: 30 minutes
```

---

## Scaling

### Horizontal Scaling

#### Add Capacity (Scale Up)

```bash
# Increase max workers
ray up config/ray_cluster.yaml --max-workers 15

# Will add workers based on autoscaling policy
# Typical scale-up time: 3-5 minutes
```

#### Remove Capacity (Scale Down)

```bash
# Decrease max workers
ray up config/ray_cluster.yaml --max-workers 5

# Will gracefully terminate idle workers
# Running jobs will complete before termination
```

### Vertical Scaling

#### Upgrade Instance Type

```yaml
# config/ray_cluster.yaml
worker_nodes:
  - name: gpu_workers
    node_config:
      InstanceType: dgxh100.80g  # Upgrade from A100 to H100
```

```bash
# Apply changes (rolling update)
ray up config/ray_cluster.yaml
```

### Auto-Scaling Configuration

#### Scale on Queue Depth

```yaml
# config/resource_allocation.yaml
auto_scaling:
  scale_up:
    - metric: queue_length
      threshold: 20
      duration_seconds: 60
      action:
        add_workers: 3
        max_workers: 15
```

#### Scale on GPU Utilization

```yaml
auto_scaling:
  scale_up:
    - metric: avg_gpu_utilization
      threshold: 0.8
      duration_seconds: 300
      action:
        add_workers: 2
```

---

## Security

### Access Control

#### SSH Access

```bash
# Production: Use bastion host
ssh -i ~/.ssh/portalis_dgx_key -J bastion@bastion-host \
  ubuntu@worker-node

# Development: Direct access (restricted IPs)
```

#### API Access

```bash
# Authenticate with API token
export PORTALIS_API_TOKEN="secret-token"

# Submit job
curl -X POST https://api.portalis.ai/v1/jobs \
  -H "Authorization: Bearer $PORTALIS_API_TOKEN" \
  -d @job.json
```

### Encryption

- **At Rest**: All S3 buckets use AES-256 encryption
- **In Transit**: TLS 1.3 for all API/dashboard access
- **Redis**: TLS enabled for cluster communication

### Secrets Management

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
  --name portalis/api-keys/customer-123 \
  --secret-string '{"api_key": "secret-key"}'

# Retrieve in application
python -m dgx_cloud.src.utils.secrets get api-keys/customer-123
```

### Security Auditing

```bash
# Review security group rules
aws ec2 describe-security-groups \
  --filters "Name=tag:portalis-cluster,Values=*"

# Review IAM permissions
aws iam get-role-policy \
  --role-name portalis-ray-node-role \
  --policy-name portalis-s3-access

# Review access logs
aws s3 sync s3://portalis-access-logs/$(date +%Y/%m/%d)/ /tmp/logs/
```

---

## Appendix

### Useful Commands

```bash
# Ray cluster status
ray status

# GPU monitoring
nvidia-smi dmon -s pucvmet -c 100

# Cost summary
python -m dgx_cloud.src.cost.optimizer report

# Storage stats
python -m dgx_cloud.src.storage.distributed_storage stats

# Worker health
ray list nodes --json | jq '.[] | {node_id, state, resources}'
```

### Configuration Files

- Ray Cluster: `/workspace/portalis/dgx-cloud/config/ray_cluster.yaml`
- Resource Allocation: `/workspace/portalis/dgx-cloud/config/resource_allocation.yaml`
- Grafana Dashboard: `/workspace/portalis/dgx-cloud/config/grafana_dashboard.json`
- Terraform: `/workspace/portalis/dgx-cloud/terraform/main.tf`

### Support Contacts

- **On-Call Engineer**: oncall@portalis.ai
- **ML Ops Team**: ml-ops@portalis.ai
- **AWS Support**: AWS Enterprise Support Portal
- **NVIDIA Support**: NVIDIA DGX Support Portal

---

**Document Version**: 1.0
**Last Updated**: 2025-10-03
**Maintained By**: ML Ops Team
