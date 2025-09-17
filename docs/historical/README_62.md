# Terraform Infrastructure as Code

## Overview
Production-ready Terraform configurations for deploying cloud computing infrastructure across multiple platforms.

## Supported Platforms

### 1. AWS Infrastructure
- **ECS/Fargate**: Container orchestration
- **EKS**: Kubernetes clusters
- **SageMaker**: ML model training and serving
- **Lambda**: Serverless functions
- **EC2**: Virtual machines
- **RDS**: Managed databases
- **S3**: Object storage

### 2. Google Cloud Infrastructure
- **GKE**: Kubernetes clusters
- **Cloud Run**: Serverless containers
- **Vertex AI**: ML platform
- **Compute Engine**: Virtual machines
- **Cloud SQL**: Managed databases
- **Cloud Storage**: Object storage

### 3. Azure Infrastructure
- **AKS**: Kubernetes clusters
- **Container Instances**: Serverless containers
- **ML Studio**: ML platform
- **Virtual Machines**: Compute instances
- **SQL Database**: Managed databases
- **Blob Storage**: Object storage

## Infrastructure Components

### 1. Base Infrastructure
```hcl
# infrastructure/terraform/base/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  
  enable_nat_gateway = true
  single_nat_gateway = false
  
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = var.common_tags
}

# Security Groups
resource "aws_security_group" "app_sg" {
  name_prefix = "${var.project_name}-app-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = var.common_tags
}
```

### 2. Kubernetes Cluster (EKS)
```hcl
# infrastructure/terraform/kubernetes/main.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = "${var.project_name}-eks"
  cluster_version = "1.28"
  
  vpc_id     = var.vpc_id
  subnet_ids = var.private_subnet_ids
  
  cluster_endpoint_public_access = true
  
  eks_managed_node_groups = {
    general = {
      desired_capacity = 2
      min_capacity     = 1
      max_capacity     = 10
      
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
    }
    
    ml = {
      desired_capacity = 2
      min_capacity     = 1
      max_capacity     = 10
      
      instance_types = ["g4dn.xlarge"]
      capacity_type  = "ON_DEMAND"
      
      labels = {
        node-type = "ml"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
    }
  }
  
  tags = var.common_tags
}

# GPU Operator for NVIDIA GPUs
resource "helm_release" "gpu_operator" {
  name       = "gpu-operator"
  repository = "https://helm.ngc.nvidia.com/nvidia"
  chart      = "gpu-operator"
  namespace  = "gpu-operator"
  create_namespace = true
  
  depends_on = [module.eks]
}
```

### 3. ML Infrastructure (SageMaker)
```hcl
# infrastructure/terraform/ml/main.tf
# SageMaker Domain
resource "aws_sagemaker_domain" "ml_domain" {
  domain_name = "${var.project_name}-ml-domain"
  auth_mode   = "IAM"
  vpc_id      = var.vpc_id
  subnet_ids  = var.private_subnet_ids
  
  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution_role.arn
    
    jupyter_server_app_settings {
      default_resource_spec {
        instance_type       = "ml.t3.medium"
        sagemaker_image_arn = data.aws_sagemaker_prebuilt_ecr_image.jupyter.registry_path
      }
    }
    
    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type       = "ml.t3.medium"
        sagemaker_image_arn = data.aws_sagemaker_prebuilt_ecr_image.kernel.registry_path
      }
    }
  }
  
  tags = var.common_tags
}

# SageMaker Model Endpoint
resource "aws_sagemaker_model" "ml_model" {
  name               = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn
  
  primary_container {
    image = var.model_image_uri
    model_data_url = var.model_artifact_url
  }
  
  tags = var.common_tags
}

resource "aws_sagemaker_endpoint_configuration" "ml_endpoint_config" {
  name = "${var.project_name}-endpoint-config"
  
  production_variants {
    variant_name           = "primary"
    model_name            = aws_sagemaker_model.ml_model.name
    instance_type         = "ml.m5.large"
    initial_instance_count = 1
  }
  
  tags = var.common_tags
}

resource "aws_sagemaker_endpoint" "ml_endpoint" {
  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ml_endpoint_config.name
  
  tags = var.common_tags
}
```

### 4. Monitoring Infrastructure
```hcl
# infrastructure/terraform/monitoring/main.tf
# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/aws/application/${var.project_name}"
  retention_in_days = 30
  
  tags = var.common_tags
}

# Prometheus Workspace
resource "aws_prometheus_workspace" "monitoring" {
  alias = "${var.project_name}-prometheus"
  
  tags = var.common_tags
}

# Grafana Workspace
resource "aws_grafana_workspace" "monitoring" {
  account_access_type      = "CURRENT_ACCOUNT"
  authentication_providers = ["AWS_SSO"]
  permission_type          = "SERVICE_MANAGED"
  role_arn                 = aws_iam_role.grafana_role.arn
  
  tags = var.common_tags
}

# S3 Bucket for ML Artifacts
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${var.project_name}-ml-artifacts-${random_string.bucket_suffix.result}"
  
  tags = var.common_tags
}

resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  
  rule {
    id     = "cleanup_old_versions"
    status = "Enabled"
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}
```

### 5. CI/CD Pipeline
```hcl
# infrastructure/terraform/cicd/main.tf
# CodeBuild Project
resource "aws_codebuild_project" "ml_pipeline" {
  name          = "${var.project_name}-ml-pipeline"
  description   = "ML model training and deployment pipeline"
  build_timeout = "60"
  service_role  = aws_iam_role.codebuild_role.arn
  
  artifacts {
    type = "CODEPIPELINE"
  }
  
  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/amazonlinux2-x86_64-standard:4.0"
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"
    
    environment_variable {
      name  = "ENVIRONMENT"
      value = var.environment
    }
  }
  
  source {
    type = "CODEPIPELINE"
    buildspec = yamlencode({
      version = "0.2"
      phases = {
        install = {
          "runtime-versions" = {
            python = "3.9"
          }
          commands = [
            "pip install -r requirements.txt"
          ]
        }
        pre_build = {
          commands = [
            "echo Logging in to Amazon ECR...",
            "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"
          ]
        }
        build = {
          commands = [
            "echo Training model...",
            "python train_model.py",
            "echo Building Docker image...",
            "docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .",
            "docker tag $IMAGE_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG"
          ]
        }
        post_build = {
          commands = [
            "echo Pushing the image...",
            "docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG",
            "echo Writing image definitions file...",
            "printf '[{\"name\":\"container_name\",\"imageUri\":\"%s\"}]' $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG > imagedefinitions.json"
          ]
        }
      }
      artifacts = {
        files = ["imagedefinitions.json"]
      }
    })
  }
  
  tags = var.common_tags
}

# CodePipeline
resource "aws_codepipeline" "ml_pipeline" {
  name     = "${var.project_name}-ml-pipeline"
  role_arn = aws_iam_role.codepipeline_role.arn
  
  artifact_store {
    location = aws_s3_bucket.artifacts.bucket
    type     = "S3"
  }
  
  stage {
    name = "Source"
    
    action {
      name             = "Source"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeCommit"
      version          = "1"
      output_artifacts = ["source_output"]
      
      configuration = {
        RepositoryName = aws_codecommit_repository.ml_repo.repository_name
        BranchName     = "main"
      }
    }
  }
  
  stage {
    name = "Build"
    
    action {
      name            = "Build"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      input_artifacts = ["source_output"]
      version         = "1"
      
      configuration = {
        ProjectName = aws_codebuild_project.ml_pipeline.name
      }
    }
  }
  
  stage {
    name = "Deploy"
    
    action {
      name            = "Deploy"
      category        = "Deploy"
      owner           = "AWS"
      provider        = "ECS"
      input_artifacts = ["source_output"]
      version         = "1"
      
      configuration = {
        ClusterName = var.ecs_cluster_name
        ServiceName = var.ecs_service_name
        FileName    = "imagedefinitions.json"
      }
    }
  }
  
  tags = var.common_tags
}
```

## Variables and Outputs

### Variables
```hcl
# infrastructure/terraform/variables.tf
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Environment = "dev"
    Project     = "ml-platform"
    ManagedBy   = "terraform"
  }
}
```

### Outputs
```hcl
# infrastructure/terraform/outputs.tf
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "sagemaker_domain_id" {
  description = "SageMaker domain ID"
  value       = aws_sagemaker_domain.ml_domain.id
}

output "sagemaker_endpoint_url" {
  description = "SageMaker endpoint URL"
  value       = aws_sagemaker_endpoint.ml_endpoint.endpoint_name
}

output "prometheus_workspace_id" {
  description = "Prometheus workspace ID"
  value       = aws_prometheus_workspace.monitoring.id
}

output "grafana_workspace_id" {
  description = "Grafana workspace ID"
  value       = aws_grafana_workspace.monitoring.id
}
```

## Usage

### 1. Initialize Terraform
```bash
cd infrastructure/terraform
terraform init
```

### 2. Plan Deployment
```bash
terraform plan -var-file="environments/dev.tfvars"
```

### 3. Apply Infrastructure
```bash
terraform apply -var-file="environments/dev.tfvars"
```

### 4. Destroy Infrastructure
```bash
terraform destroy -var-file="environments/dev.tfvars"
```

## Best Practices

### 1. State Management
- **Remote State**: Use S3 backend for state storage
- **State Locking**: Use DynamoDB for state locking
- **Workspaces**: Use workspaces for environment separation

### 2. Security
- **IAM Roles**: Use least privilege principle
- **Encryption**: Enable encryption at rest and in transit
- **Network Security**: Use security groups and NACLs
- **Secrets Management**: Use AWS Secrets Manager

### 3. Cost Optimization
- **Resource Tagging**: Tag resources for cost allocation
- **Auto-scaling**: Use auto-scaling groups
- **Spot Instances**: Use spot instances for batch workloads
- **Reserved Instances**: Use reserved instances for predictable workloads

### 4. Monitoring
- **CloudWatch**: Set up comprehensive monitoring
- **Logging**: Centralize logs in CloudWatch Logs
- **Alerts**: Set up CloudWatch alarms
- **Dashboards**: Create monitoring dashboards
