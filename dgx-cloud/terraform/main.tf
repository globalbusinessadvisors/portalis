# Portalis DGX Cloud Infrastructure
# Terraform configuration for AWS deployment

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "portalis-terraform-state"
    key            = "dgx-cloud/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "portalis-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Portalis"
      Component   = "DGX-Cloud"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "cluster_name" {
  description = "Ray cluster name"
  type        = string
  default     = "portalis-translation-cluster"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "max_workers" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = true
}

# VPC Configuration
resource "aws_vpc" "portalis" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

# Public Subnets (for head node and bastion)
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.portalis.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.cluster_name}-public-${count.index + 1}"
  }
}

# Private Subnets (for worker nodes)
resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.portalis.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.cluster_name}-private-${count.index + 1}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "portalis" {
  vpc_id = aws_vpc.portalis.id

  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

# NAT Gateway
resource "aws_eip" "nat" {
  count  = 2
  domain = "vpc"

  tags = {
    Name = "${var.cluster_name}-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "portalis" {
  count = 2

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "${var.cluster_name}-nat-${count.index + 1}"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.portalis.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.portalis.id
  }

  tags = {
    Name = "${var.cluster_name}-public-rt"
  }
}

resource "aws_route_table" "private" {
  count  = 2
  vpc_id = aws_vpc.portalis.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.portalis[count.index].id
  }

  tags = {
    Name = "${var.cluster_name}-private-rt-${count.index + 1}"
  }
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "ray_head" {
  name_description = "Security group for Ray head node"
  vpc_id           = aws_vpc.portalis.id

  # Ray client port
  ingress {
    from_port   = 10001
    to_port     = 10001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Ray client port"
  }

  # Ray dashboard
  ingress {
    from_port   = 8265
    to_port     = 8265
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Ray dashboard"
  }

  # Ray GCS server
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Ray GCS server"
  }

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }

  # Prometheus
  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Prometheus"
  }

  # Grafana
  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Grafana"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "${var.cluster_name}-ray-head-sg"
  }
}

resource "aws_security_group" "ray_worker" {
  name_description = "Security group for Ray worker nodes"
  vpc_id           = aws_vpc.portalis.id

  # Allow all traffic from head node
  ingress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = [aws_security_group.ray_head.id]
    description     = "All traffic from head node"
  }

  # Allow worker-to-worker communication
  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
    description = "Worker-to-worker communication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "${var.cluster_name}-ray-worker-sg"
  }
}

# S3 Buckets
resource "aws_s3_bucket" "models" {
  bucket = "${var.cluster_name}-models-${var.environment}"

  tags = {
    Name = "Portalis Models"
    Type = "models"
  }
}

resource "aws_s3_bucket" "cache" {
  bucket = "${var.cluster_name}-cache-${var.environment}"

  tags = {
    Name = "Portalis Cache"
    Type = "cache"
  }
}

resource "aws_s3_bucket" "results" {
  bucket = "${var.cluster_name}-results-${var.environment}"

  tags = {
    Name = "Portalis Results"
    Type = "results"
  }
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Lifecycle Rules
resource "aws_s3_bucket_lifecycle_configuration" "cache" {
  bucket = aws_s3_bucket.cache.id

  rule {
    id     = "expire-old-cache"
    status = "Enabled"

    expiration {
      days = 7
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "results" {
  bucket = aws_s3_bucket.results.id

  rule {
    id     = "archive-old-results"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 90
    }
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.cluster_name}-redis-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "redis" {
  name_description = "Security group for Redis cluster"
  vpc_id           = aws_vpc.portalis.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ray_head.id, aws_security_group.ray_worker.id]
    description     = "Redis from Ray cluster"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-redis-sg"
  }
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.cluster_name}-redis"
  replication_group_description = "Redis cluster for Portalis caching"

  engine               = "redis"
  engine_version       = "7.0"
  node_type            = "cache.r6g.large"
  number_cache_clusters = 3
  port                 = 6379

  subnet_group_name   = aws_elasticache_subnet_group.redis.name
  security_group_ids  = [aws_security_group.redis.id]

  automatic_failover_enabled = true
  multi_az_enabled           = true

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = 5
  snapshot_window          = "03:00-05:00"

  tags = {
    Name = "${var.cluster_name}-redis"
  }
}

# IAM Role for EC2 Instances
resource "aws_iam_role" "ray_node" {
  name = "${var.cluster_name}-ray-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ray_node_s3" {
  name = "${var.cluster_name}-s3-access"
  role = aws_iam_role.ray_node.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Effect = "Allow"
        Resource = [
          "${aws_s3_bucket.models.arn}/*",
          "${aws_s3_bucket.cache.arn}/*",
          "${aws_s3_bucket.results.arn}/*",
          aws_s3_bucket.models.arn,
          aws_s3_bucket.cache.arn,
          aws_s3_bucket.results.arn
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ray_node" {
  name = "${var.cluster_name}-ray-node-profile"
  role = aws_iam_role.ray_node.name
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.portalis.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "ray_head_security_group_id" {
  description = "Ray head node security group ID"
  value       = aws_security_group.ray_head.id
}

output "ray_worker_security_group_id" {
  description = "Ray worker security group ID"
  value       = aws_security_group.ray_worker.id
}

output "s3_models_bucket" {
  description = "S3 bucket for models"
  value       = aws_s3_bucket.models.bucket
}

output "s3_cache_bucket" {
  description = "S3 bucket for cache"
  value       = aws_s3_bucket.cache.bucket
}

output "s3_results_bucket" {
  description = "S3 bucket for results"
  value       = aws_s3_bucket.results.bucket
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "iam_instance_profile" {
  description = "IAM instance profile for Ray nodes"
  value       = aws_iam_instance_profile.ray_node.name
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}
