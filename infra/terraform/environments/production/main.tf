# Step 107: Terraform — AWS infrastructure for AuthentiGuard.
#
# Module layout:
#   modules/vpc    — VPC, subnets, NAT gateways, route tables
#   modules/eks    — EKS cluster + node groups (CPU + GPU)
#   modules/rds    — PostgreSQL 16 Multi-AZ RDS instance
#   modules/redis  — ElastiCache Redis 7 cluster
#   modules/s3     — S3 buckets for uploads, reports, models, ALB logs
#
# Usage:
#   cd infra/terraform/environments/production
#   terraform init
#   terraform plan -var-file=terraform.tfvars
#   terraform apply -var-file=terraform.tfvars

# ── Root variables ────────────────────────────────────────────
variable "aws_region"    { default = "us-east-1" }
variable "environment"   {}                       # "staging" | "production"
variable "project"       { default = "authentiguard" }
variable "vpc_cidr"      { default = "10.0.0.0/16" }
variable "db_password"   { sensitive = true }
variable "redis_auth"    { sensitive = true }
variable "eks_k8s_version" { default = "1.30" }

locals {
  name   = "${var.project}-${var.environment}"
  tags   = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }

  # Remote state — S3 backend with DynamoDB locking
  backend "s3" {
    bucket         = "authentiguard-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "authentiguard-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags { tags = local.tags }
}

# ── VPC Module ────────────────────────────────────────────────
module "vpc" {
  source = "../../modules/vpc"

  name         = local.name
  cidr         = var.vpc_cidr
  environment  = var.environment

  azs              = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets   = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = false       # one NAT per AZ for HA
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Required tags for EKS subnet auto-discovery
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"                     = "1"
    "kubernetes.io/cluster/${local.name}"                 = "shared"
  }
  public_subnet_tags = {
    "kubernetes.io/role/elb"                              = "1"
    "kubernetes.io/cluster/${local.name}"                 = "shared"
  }
}

# ── EKS Module ────────────────────────────────────────────────
module "eks" {
  source = "../../modules/eks"

  cluster_name    = local.name
  cluster_version = var.eks_k8s_version
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  # Endpoint: private only (no public API server)
  cluster_endpoint_public_access  = false
  cluster_endpoint_private_access = true

  # CPU node group
  node_groups = {
    cpu = {
      instance_types = ["m6i.xlarge", "m6i.2xlarge"]
      min_size       = 3
      desired_size   = 4
      max_size       = 20
      capacity_type  = "ON_DEMAND"
      disk_size      = 100
      labels         = { "role" = "cpu-worker" }
    }
    gpu = {
      instance_types = ["g4dn.xlarge"]
      min_size       = 1
      desired_size   = 2
      max_size       = 8
      capacity_type  = "ON_DEMAND"
      disk_size      = 200
      labels         = { "role" = "gpu-worker" }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      # Custom GPU AMI with NVIDIA drivers pre-installed
      ami_type = "AL2_x86_64_GPU"
    }
  }

  # AWS Load Balancer Controller (required for ALB ingress)
  enable_aws_load_balancer_controller = true

  # EFS CSI driver (required for ReadWriteMany model cache)
  enable_aws_efs_csi_driver = true

  # KEDA (queue-based autoscaling)
  enable_keda = true

  # External Secrets Operator
  enable_external_secrets = true

  # Cluster Autoscaler
  enable_cluster_autoscaler = true

  tags = local.tags
}

# ── RDS PostgreSQL Module ─────────────────────────────────────
module "rds" {
  source = "../../modules/rds"

  identifier       = local.name
  engine_version   = "16.2"
  instance_class   = "db.r7g.large"     # 2 vCPU, 16 GB RAM — Graviton3
  allocated_storage = 100
  storage_type     = "gp3"
  multi_az         = true

  database_name    = "authentiguard"
  username         = "authentiguard"
  password         = var.db_password

  # Place RDS in database subnets (not accessible from internet)
  db_subnet_group_name = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [module.vpc.rds_security_group_id]

  # Encryption
  storage_encrypted   = true
  kms_key_id          = module.kms.rds_key_arn

  # Backups
  backup_retention_period = 30
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"
  deletion_protection     = true

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7

  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = module.iam.rds_monitoring_role_arn

  # PostgreSQL parameters
  parameters = [
    {name = "max_connections",   value = "500"},
    {name = "shared_buffers",    value = "4GB"},
    {name = "log_connections",   value = "1"},
    {name = "log_disconnections",value = "1"},
  ]

  tags = local.tags
}

# ── ElastiCache Redis Module ──────────────────────────────────
module "redis" {
  source = "../../modules/redis"

  cluster_id     = local.name
  engine_version = "7.2"
  node_type      = "cache.r7g.large"    # 2 vCPU, 13 GB RAM — Graviton3

  # Cluster mode disabled (single shard, multi-AZ with failover)
  num_cache_nodes            = 1
  automatic_failover_enabled = true
  multi_az_enabled           = true

  subnet_group_name = module.vpc.elasticache_subnet_group_name
  security_group_ids = [module.vpc.redis_security_group_id]

  # TLS in transit
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth

  # Encryption at rest
  at_rest_encryption_enabled = true
  kms_key_id                 = module.kms.redis_key_arn

  # Backups
  snapshot_retention_limit = 7
  snapshot_window          = "02:00-03:00"

  tags = local.tags
}

# ── S3 Buckets Module ─────────────────────────────────────────
module "s3" {
  source = "../../modules/s3"
  name   = local.name

  buckets = {
    uploads = {
      versioning = false
      lifecycle_rules = [{
        id     = "expire-uploads"
        status = "Enabled"
        expiration = {days = 31}
      }]
    }
    reports = {
      versioning = true
      lifecycle_rules = [{
        id     = "archive-old-reports"
        status = "Enabled"
        transition = [{days = 90, storage_class = "STANDARD_IA"}]
        expiration = {days = 365}
      }]
    }
    models = {
      versioning = true
      # Model artifacts retained indefinitely (no lifecycle rule)
    }
    alb-logs = {
      versioning = false
      lifecycle_rules = [{
        id     = "expire-logs"
        status = "Enabled"
        expiration = {days = 90}
      }]
    }
  }

  kms_key_arn = module.kms.s3_key_arn
  tags        = local.tags
}

# ── Outputs (referenced by CI/CD and Helm values) ─────────────
output "cluster_name"        { value = module.eks.cluster_name }
output "cluster_endpoint"    { value = module.eks.cluster_endpoint; sensitive = true }
output "rds_endpoint"        { value = module.rds.endpoint; sensitive = true }
output "redis_endpoint"      { value = module.redis.primary_endpoint; sensitive = true }
output "s3_bucket_uploads"   { value = module.s3.bucket_names.uploads }
output "s3_bucket_reports"   { value = module.s3.bucket_names.reports }
output "ecr_registry"        { value = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com" }

data "aws_caller_identity" "current" {}
