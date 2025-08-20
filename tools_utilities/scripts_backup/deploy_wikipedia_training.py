#!/usr/bin/env python3
"""
Wikipedia Training Deployment Automation
========================================

Automated deployment and management of Wikipedia training infrastructure
across AWS, GCP, and Azure cloud platforms.

Purpose: Automate cloud deployment for large-scale Wikipedia training
Inputs: Cloud configuration, training parameters, credentials
Outputs: Deployed infrastructure, training endpoints, monitoring dashboards
Seeds: N/A (infrastructure deployment)
Dependencies: boto3, google-cloud, azure-mgmt, kubernetes, docker
"""

import os, sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import yaml
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    import boto3
    from kubernetes import client, config
    import docker
    HAS_CLOUD_DEPS = True
except ImportError:
    HAS_CLOUD_DEPS = False
    logging.warning("Cloud dependencies not installed. Install with: pip install boto3 kubernetes docker")


@dataclass
class DeploymentConfig:
    """Configuration for Wikipedia training deployment."""
    
    # Cloud Provider Settings
    cloud_provider: str = "aws"  # aws, gcp, azure
    region: str = "us-west-2"
    
    # Cluster Configuration
    cluster_name: str = "quark-wikipedia-training"
    node_count: int = 4
    instance_type: str = "p3.8xlarge"  # AWS instance type
    gpu_count_per_node: int = 4
    
    # Container Configuration
    container_image: str = "quark/wikipedia-training:latest"
    namespace: str = "quark-training"
    
    # Storage Configuration
    storage_bucket: str = "quark-wikipedia-training"
    cache_size_gb: int = 500
    output_size_gb: int = 200
    
    # Training Configuration
    wikipedia_dump_date: str = "20240101"
    model_name: str = "microsoft/DialoGPT-medium"
    max_training_hours: int = 48
    
    # Monitoring Configuration
    enable_monitoring: bool = True
    wandb_api_key: Optional[str] = None
    
    # Security Configuration
    create_new_keypair: bool = True
    allowed_ssh_cidrs: List[str] = None
    
    def __post_init__(self):
        if self.allowed_ssh_cidrs is None:
            self.allowed_ssh_cidrs = ["0.0.0.0/0"]  # WARNING: Open to all, change for production


class AWSDeploymentManager:
    """Manages AWS deployment for Wikipedia training."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not HAS_CLOUD_DEPS:
            raise ImportError("Cloud dependencies not installed")
            
        # Initialize AWS clients
        self.session = boto3.Session()
        self.ec2 = self.session.client('ec2', region_name=config.region)
        self.s3 = self.session.client('s3', region_name=config.region)
        self.eks = self.session.client('eks', region_name=config.region)
        self.iam = self.session.client('iam')
        
    def create_s3_bucket(self) -> str:
        """Create S3 bucket for training data and models."""
        bucket_name = self.config.storage_bucket
        
        try:
            if self.config.region == 'us-east-1':
                self.s3.create_bucket(Bucket=bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.config.region}
                )
            
            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Set lifecycle policy
            lifecycle_config = {
                'Rules': [
                    {
                        'ID': 'QuarkTrainingDataLifecycle',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': 'cache/'},
                        'Transitions': [
                            {
                                'Days': 30,
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': 90,
                                'StorageClass': 'GLACIER'
                            }
                        ]
                    }
                ]
            }
            
            self.s3.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            
            self.logger.info(f"Created S3 bucket: {bucket_name}")
            return bucket_name
            
        except Exception as e:
            if "BucketAlreadyExists" in str(e) or "BucketAlreadyOwnedByYou" in str(e):
                self.logger.info(f"S3 bucket already exists: {bucket_name}")
                return bucket_name
            else:
                raise
    
    def create_iam_roles(self) -> Dict[str, str]:
        """Create IAM roles for EKS cluster and nodes."""
        roles = {}
        
        # EKS Cluster Service Role
        cluster_role_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "eks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            cluster_role_response = self.iam.create_role(
                RoleName=f"{self.config.cluster_name}-cluster-role",
                AssumeRolePolicyDocument=json.dumps(cluster_role_doc),
                Description="EKS Cluster Service Role for Quark Wikipedia Training"
            )
            roles['cluster_role_arn'] = cluster_role_response['Role']['Arn']
            
            # Attach required policies
            cluster_policies = [
                'arn:aws:iam::aws:policy/AmazonEKSClusterPolicy'
            ]
            
            for policy_arn in cluster_policies:
                self.iam.attach_role_policy(
                    RoleName=f"{self.config.cluster_name}-cluster-role",
                    PolicyArn=policy_arn
                )
                
        except Exception as e:
            if "EntityAlreadyExists" in str(e):
                role_response = self.iam.get_role(RoleName=f"{self.config.cluster_name}-cluster-role")
                roles['cluster_role_arn'] = role_response['Role']['Arn']
            else:
                raise
        
        # EKS Node Group Role
        node_role_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            node_role_response = self.iam.create_role(
                RoleName=f"{self.config.cluster_name}-node-role",
                AssumeRolePolicyDocument=json.dumps(node_role_doc),
                Description="EKS Node Group Role for Quark Wikipedia Training"
            )
            roles['node_role_arn'] = node_role_response['Role']['Arn']
            
            # Attach required policies
            node_policies = [
                'arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy',
                'arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy',
                'arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess'  # For accessing training data
            ]
            
            for policy_arn in node_policies:
                self.iam.attach_role_policy(
                    RoleName=f"{self.config.cluster_name}-node-role",
                    PolicyArn=policy_arn
                )
                
        except Exception as e:
            if "EntityAlreadyExists" in str(e):
                role_response = self.iam.get_role(RoleName=f"{self.config.cluster_name}-node-role")
                roles['node_role_arn'] = role_response['Role']['Arn']
            else:
                raise
        
        self.logger.info(f"Created/verified IAM roles: {roles}")
        return roles
    
    def create_eks_cluster(self, roles: Dict[str, str]) -> str:
        """Create EKS cluster for training."""
        cluster_name = self.config.cluster_name
        
        # Get default VPC and subnets
        vpcs = self.ec2.describe_vpcs(Filters=[{'Name': 'is-default', 'Values': ['true']}])
        if not vpcs['Vpcs']:
            raise ValueError("No default VPC found")
        
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        subnets = self.ec2.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
        )
        subnet_ids = [subnet['SubnetId'] for subnet in subnets['Subnets']]
        
        try:
            cluster_response = self.eks.create_cluster(
                name=cluster_name,
                version='1.28',
                roleArn=roles['cluster_role_arn'],
                resourcesVpcConfig={
                    'subnetIds': subnet_ids,
                    'endpointConfigPrivate': False,
                    'endpointConfigPublic': True,
                    'publicAccessCidrs': ['0.0.0.0/0']
                },
                tags={
                    'Project': 'quark-brain-simulation',
                    'Component': 'wikipedia-training',
                    'Environment': 'production'
                }
            )
            
            cluster_arn = cluster_response['cluster']['arn']
            
            # Wait for cluster to be active
            self.logger.info(f"Creating EKS cluster: {cluster_name}")
            waiter = self.eks.get_waiter('cluster_active')
            waiter.wait(name=cluster_name, WaiterConfig={'Delay': 30, 'MaxAttempts': 40})
            
            self.logger.info(f"EKS cluster created successfully: {cluster_name}")
            return cluster_arn
            
        except Exception as e:
            if "ResourceInUseException" in str(e):
                self.logger.info(f"EKS cluster already exists: {cluster_name}")
                cluster_response = self.eks.describe_cluster(name=cluster_name)
                return cluster_response['cluster']['arn']
            else:
                raise
    
    def create_node_group(self, cluster_name: str, roles: Dict[str, str]) -> str:
        """Create GPU-enabled node group for training."""
        node_group_name = f"{cluster_name}-gpu-nodes"
        
        # Get subnets
        cluster_info = self.eks.describe_cluster(name=cluster_name)
        subnet_ids = cluster_info['cluster']['resourcesVpcConfig']['subnetIds']
        
        try:
            node_group_response = self.eks.create_nodegroup(
                clusterName=cluster_name,
                nodegroupName=node_group_name,
                scalingConfig={
                    'minSize': 1,
                    'maxSize': self.config.node_count * 2,
                    'desiredSize': self.config.node_count
                },
                instanceTypes=[self.config.instance_type],
                subnets=subnet_ids,
                nodeRole=roles['node_role_arn'],
                amiType='AL2_x86_64_GPU',
                capacityType='ON_DEMAND',
                tags={
                    'Project': 'quark-brain-simulation',
                    'Component': 'wikipedia-training-nodes',
                    'Environment': 'production'
                }
            )
            
            # Wait for node group to be active
            self.logger.info(f"Creating node group: {node_group_name}")
            waiter = self.eks.get_waiter('nodegroup_active')
            waiter.wait(
                clusterName=cluster_name,
                nodegroupName=node_group_name,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 40}
            )
            
            self.logger.info(f"Node group created successfully: {node_group_name}")
            return node_group_response['nodegroup']['nodegroupArn']
            
        except Exception as e:
            if "ResourceInUseException" in str(e):
                self.logger.info(f"Node group already exists: {node_group_name}")
                node_group_response = self.eks.describe_nodegroup(
                    clusterName=cluster_name,
                    nodegroupName=node_group_name
                )
                return node_group_response['nodegroup']['nodegroupArn']
            else:
                raise
    
    def deploy_training_infrastructure(self) -> Dict[str, str]:
        """Deploy complete training infrastructure on AWS."""
        self.logger.info("Deploying Wikipedia training infrastructure on AWS...")
        
        # Create S3 bucket
        bucket_name = self.create_s3_bucket()
        
        # Create IAM roles
        roles = self.create_iam_roles()
        
        # Create EKS cluster
        cluster_arn = self.create_eks_cluster(roles)
        
        # Create node group
        node_group_arn = self.create_node_group(self.config.cluster_name, roles)
        
        # Update kubeconfig
        subprocess.run([
            'aws', 'eks', 'update-kubeconfig',
            '--region', self.config.region,
            '--name', self.config.cluster_name
        ], check=True)
        
        return {
            'provider': 'aws',
            'region': self.config.region,
            'cluster_name': self.config.cluster_name,
            'cluster_arn': cluster_arn,
            'node_group_arn': node_group_arn,
            'bucket_name': bucket_name,
            'roles': roles
        }


class KubernetesDeploymentManager:
    """Manages Kubernetes deployment for Wikipedia training."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load Kubernetes config
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()
    
    def create_namespace(self) -> None:
        """Create namespace for training workloads."""
        namespace = client.V1Namespace(
            metadata=client.V1ObjectMeta(
                name=self.config.namespace,
                labels={
                    'project': 'quark-brain-simulation',
                    'component': 'wikipedia-training'
                }
            )
        )
        
        try:
            self.v1.create_namespace(namespace)
            self.logger.info(f"Created namespace: {self.config.namespace}")
        except Exception as e:
            if "AlreadyExists" in str(e):
                self.logger.info(f"Namespace already exists: {self.config.namespace}")
            else:
                raise
    
    def deploy_training_workload(self, infrastructure_info: Dict[str, str]) -> Dict[str, str]:
        """Deploy Wikipedia training workload to Kubernetes."""
        self.logger.info("Deploying training workload to Kubernetes...")
        
        # Create namespace
        self.create_namespace()
        
        # Load deployment YAML
        deployment_yaml_path = Path(__file__).parent.parent / "wikipedia_training_deployment.yaml"
        
        with open(deployment_yaml_path, 'r') as f:
            documents = list(yaml.safe_load_all(f))
        
        deployed_resources = []
        
        for doc in documents:
            if doc is None:
                continue
                
            kind = doc.get('kind')
            name = doc.get('metadata', {}).get('name')
            
            # Update environment variables with infrastructure info
            if kind == 'Deployment':
                containers = doc.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                for container in containers:
                    if 'env' in container:
                        for env_var in container['env']:
                            if env_var['name'] == 'S3_BUCKET':
                                env_var['value'] = infrastructure_info.get('bucket_name', self.config.storage_bucket)
            
            try:
                if kind == 'ConfigMap':
                    self.v1.create_namespaced_config_map(self.config.namespace, doc)
                elif kind == 'Deployment':
                    self.apps_v1.create_namespaced_deployment(self.config.namespace, doc)
                elif kind == 'Service':
                    self.v1.create_namespaced_service(self.config.namespace, doc)
                elif kind == 'Job':
                    self.batch_v1.create_namespaced_job(self.config.namespace, doc)
                elif kind == 'PersistentVolumeClaim':
                    self.v1.create_namespaced_persistent_volume_claim(self.config.namespace, doc)
                
                deployed_resources.append(f"{kind}/{name}")
                self.logger.info(f"Deployed {kind}: {name}")
                
            except Exception as e:
                if "AlreadyExists" in str(e):
                    self.logger.info(f"{kind} already exists: {name}")
                else:
                    self.logger.error(f"Failed to deploy {kind} {name}: {e}")
                    raise
        
        return {
            'namespace': self.config.namespace,
            'deployed_resources': deployed_resources
        }


class WikipediaTrainingDeployer:
    """Main deployment orchestrator for Wikipedia training."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cloud-specific managers
        if config.cloud_provider == "aws":
            self.cloud_manager = AWSDeploymentManager(config)
        else:
            raise NotImplementedError(f"Cloud provider {config.cloud_provider} not yet implemented")
        
        self.k8s_manager = KubernetesDeploymentManager(config)
    
    def build_and_push_image(self) -> str:
        """Build and push training container image."""
        self.logger.info("Building and pushing container image...")
        
        # Get Docker client
        docker_client = docker.from_env()
        
        # Build image
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile.wikipedia-training"
        context_path = Path(__file__).parent.parent.parent.parent
        
        image_tag = f"{self.config.container_image}:{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.logger.info(f"Building image: {image_tag}")
        image, logs = docker_client.images.build(
            path=str(context_path),
            dockerfile=str(dockerfile_path),
            tag=image_tag,
            rm=True,
            pull=True
        )
        
        # Log build output
        for log in logs:
            if 'stream' in log:
                self.logger.debug(log['stream'].strip())
        
        # Tag as latest
        image.tag(self.config.container_image, 'latest')
        
        # Push to registry (assuming ECR for AWS)
        if self.config.cloud_provider == "aws":
            # Get ECR login token
            import base64
            
            ecr = boto3.client('ecr', region_name=self.config.region)
            token_response = ecr.get_authorization_token()
            token = token_response['authorizationData'][0]['authorizationToken']
            username, password = base64.b64decode(token).decode().split(':')
            
            # Login to ECR
            docker_client.login(
                username=username,
                password=password,
                registry=token_response['authorizationData'][0]['proxyEndpoint']
            )
        
        # Push image
        self.logger.info(f"Pushing image: {image_tag}")
        docker_client.images.push(image_tag)
        docker_client.images.push(self.config.container_image, 'latest')
        
        return image_tag
    
    def deploy_complete_infrastructure(self) -> Dict[str, str]:
        """Deploy complete Wikipedia training infrastructure."""
        self.logger.info("Starting complete infrastructure deployment...")
        
        try:
            # Build and push container image
            image_tag = self.build_and_push_image()
            
            # Deploy cloud infrastructure
            infrastructure_info = self.cloud_manager.deploy_training_infrastructure()
            
            # Wait for cluster to be ready
            time.sleep(60)
            
            # Deploy Kubernetes workloads
            k8s_info = self.k8s_manager.deploy_training_workload(infrastructure_info)
            
            # Combine results
            deployment_info = {
                **infrastructure_info,
                **k8s_info,
                'container_image': image_tag,
                'deployment_time': datetime.now().isoformat(),
                'config': self.config.__dict__
            }
            
            self.logger.info("Infrastructure deployment completed successfully!")
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
    
    def get_deployment_status(self) -> Dict[str, str]:
        """Get current deployment status."""
        status = {}
        
        try:
            # Check cluster status
            if self.config.cloud_provider == "aws":
                cluster_response = self.cloud_manager.eks.describe_cluster(
                    name=self.config.cluster_name
                )
                status['cluster_status'] = cluster_response['cluster']['status']
            
            # Check pods status
            pods = self.k8s_manager.v1.list_namespaced_pod(self.config.namespace)
            status['pod_count'] = len(pods.items)
            status['pods'] = [
                {
                    'name': pod.metadata.name,
                    'status': pod.status.phase,
                    'ready': pod.status.container_statuses[0].ready if pod.status.container_statuses else False
                }
                for pod in pods.items
            ]
            
        except Exception as e:
            status['error'] = str(e)
        
        return status
    
    def cleanup_deployment(self) -> None:
        """Clean up deployed resources."""
        self.logger.info("Cleaning up deployment...")
        
        try:
            # Delete Kubernetes resources
            self.k8s_manager.v1.delete_namespace(name=self.config.namespace)
            self.logger.info(f"Deleted namespace: {self.config.namespace}")
            
            # Clean up cloud resources
            if self.config.cloud_provider == "aws":
                # Delete node group
                self.cloud_manager.eks.delete_nodegroup(
                    clusterName=self.config.cluster_name,
                    nodegroupName=f"{self.config.cluster_name}-gpu-nodes"
                )
                
                # Wait for node group deletion
                time.sleep(300)  # 5 minutes
                
                # Delete cluster
                self.cloud_manager.eks.delete_cluster(name=self.config.cluster_name)
                
                self.logger.info("Cloud resources cleanup initiated")
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(description="Deploy Wikipedia training infrastructure")
    
    parser.add_argument("--cloud-provider", type=str, default="aws", choices=["aws", "gcp", "azure"])
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--cluster-name", type=str, default="quark-wikipedia-training")
    parser.add_argument("--node-count", type=int, default=4)
    parser.add_argument("--instance-type", type=str, default="p3.8xlarge")
    parser.add_argument("--container-image", type=str, default="quark/wikipedia-training:latest")
    parser.add_argument("--storage-bucket", type=str, default="quark-wikipedia-training")
    parser.add_argument("--wandb-api-key", type=str, help="Weights & Biases API key")
    
    parser.add_argument("--action", type=str, default="deploy", 
                       choices=["deploy", "status", "cleanup"],
                       help="Action to perform")
    parser.add_argument("--config", type=str, help="Path to deployment configuration JSON")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deployment.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = DeploymentConfig(**config_dict)
    else:
        config = DeploymentConfig(
            cloud_provider=args.cloud_provider,
            region=args.region,
            cluster_name=args.cluster_name,
            node_count=args.node_count,
            instance_type=args.instance_type,
            container_image=args.container_image,
            storage_bucket=args.storage_bucket,
            wandb_api_key=args.wandb_api_key
        )
    
    # Initialize deployer
    deployer = WikipediaTrainingDeployer(config)
    
    # Execute action
    if args.action == "deploy":
        deployment_info = deployer.deploy_complete_infrastructure()
        print(f"Deployment completed: {json.dumps(deployment_info, indent=2)}")
        
        # Save deployment info
        with open('deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
            
    elif args.action == "status":
        status = deployer.get_deployment_status()
        print(f"Deployment status: {json.dumps(status, indent=2)}")
        
    elif args.action == "cleanup":
        deployer.cleanup_deployment()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
