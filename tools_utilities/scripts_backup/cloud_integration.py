"""
Cloud Integration for SmallMind Multi-Model Training

Enables training all models simultaneously on cloud platforms
for better performance, scalability, and cost efficiency.
"""

import os, sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import subprocess

try:
    import boto3
    import botocore
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import compute_v1
    from google.cloud import storage
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

try:
    from azure.mgmt.compute import ComputeManagementClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CloudConfig:
    """Configuration for cloud platform"""
    platform: str  # aws, gcp, azure
    region: str
    instance_type: str
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    storage_gb: int = 100
    max_instances: int = 3
    project_id: Optional[str] = None  # For GCP
    ssh_key_name: Optional[str] = None
    security_group: Optional[str] = None
    subnet_id: Optional[str] = None

@dataclass
class CloudInstance:
    """Cloud instance information"""
    instance_id: str
    public_ip: str
    private_ip: str
    status: str
    platform: str
    instance_type: str
    launch_time: datetime
    cost_per_hour: float
    gpu_info: Optional[str] = None

class CloudTrainer:
    """Cloud-based multi-model training system"""
    
    def __init__(self, cloud_config: CloudConfig, output_dir: str = "./cloud_training"):
        self.cloud_config = cloud_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cloud clients
        self.aws_client = None
        self.gcp_client = None
        self.azure_client = None
        
        # Instance tracking
        self.instances: Dict[str, CloudInstance] = {}
        self.training_status = {}
        
        # Initialize cloud clients
        self._init_cloud_clients()
        
        logger.info(f"CloudTrainer initialized for {cloud_config.platform}")
    
    def _init_cloud_clients(self):
        """Initialize cloud platform clients"""
        if self.cloud_config.platform == "aws" and AWS_AVAILABLE:
            try:
                self.aws_client = boto3.client('ec2', region_name=self.cloud_config.region)
                logger.info("✅ AWS client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AWS client: {e}")
        
        elif self.cloud_config.platform == "gcp" and GOOGLE_CLOUD_AVAILABLE:
            try:
                self.gcp_client = compute_v1.InstancesClient()
                logger.info("✅ Google Cloud client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud client: {e}")
        
        elif self.cloud_config.platform == "azure" and AZURE_AVAILABLE:
            try:
                credential = DefaultAzureCredential()
                self.azure_client = ComputeManagementClient(credential, self.cloud_config.region)
                logger.info("✅ Azure client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Azure client: {e}")
    
    def _create_aws_instance(self, model_name: str, gpu_type: str = "g4dn.xlarge") -> Optional[CloudInstance]:
        """Create AWS EC2 instance for training"""
        if not self.aws_client:
            logger.error("AWS client not available")
            return None
        
        try:
            # AMI selection based on GPU type
            ami_map = {
                "g4dn.xlarge": "ami-0c7217cdde317cfec",  # Deep Learning AMI GPU PyTorch
                "p3.2xlarge": "ami-0c7217cdde317cfec",
                "g5.xlarge": "ami-0c7217cdde317cfec"
            }
            
            ami_id = ami_map.get(gpu_type, "ami-0c7217cdde317cfec")
            
            # User data script for automatic setup
            user_data = f"""#!/bin/bash
# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone SmallMind repository
cd HOME/ec2-user
git clone https://github.com/smallmind/smallmind.git
cd smallmind

# Create training script for {model_name}
cat > train_{model_name.lower().replace('-', '_').replace('.', '_')}.py << 'EOF'
#!/usr/bin/env python3
import os, sys
sys.path.insert(0, 'src')

from smallmind.models.continuous_trainer import train_forever

# Start training
train_forever(
    model_path="models/models/checkpoints/{model_name.lower()}",
    output_dir="./cloud_output"
)
EOF

# Install Python dependencies
pip3 install datasets transformers torch accelerate

# Start training
python3 train_{model_name.lower().replace('-', '_').replace('.', '_')}.py
"""
            
            # Launch instance
            instance_params = {
                'ImageId': ami_id,
                'MinacceleratorCount': 1,
                'MaxacceleratorCount': 1,
                'InstanceacceleratorType': gpu_type,
                'KeyName': self.cloud_config.ssh_key_name,
                'UserData': user_data,
                'TagSpecifications': [
                    {
                        'ResourceacceleratorType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'SmallMind-{model_name}'},
                            {'Key': 'Project', 'Value': 'SmallMind-Training'},
                            {'Key': 'Model', 'Value': model_name}
                        ]
                    }
                ]
            }
            
            # Add optional parameters only if provided
            if self.cloud_config.security_group and self.cloud_config.security_group.strip():
                instance_params['SecurityGroupIds'] = [self.cloud_config.security_group]
            if self.cloud_config.subnet_id and self.cloud_config.subnet_id.strip():
                instance_params['SubnetId'] = self.cloud_config.subnet_id
            
            response = self.aws_client.run_instances(**instance_params)
            
            instance_id = response['Instances'][0]['InstanceId']
            
            # Wait for instance to be running
            waiter = self.aws_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Get instance details
            instance_info = self.aws_client.describe_instances(InstanceIds=[instance_id])
            instance = instance_info['Reservations'][0]['Instances'][0]
            
            # Calculate cost (approximate)
            cost_per_hour = self._get_aws_instance_cost(gpu_type)
            
            cloud_instance = CloudInstance(
                instance_id=instance_id,
                public_ip=instance.get('PublicIpAddress', ''),
                private_ip=instance.get('PrivateIpAddress', ''),
                status=instance['State']['Name'],
                platform='aws',
                instance_acceleratorType=gpu_acceleratorType,
                launch_time=instance['LaunchTime'],
                cost_per_hour=cost_per_hour,
                gpu_info=gpu_acceleratorType
            )
            
            logger.info(f"✅ AWS instance created: {instance_id} ({gpu_type})")
            return cloud_instance
            
        except Exception as e:
            logger.error(f"Failed to create AWS instance: {e}")
            return None
    
    def _create_gcp_instance(self, model_name: str, gpu_type: str = "n1-standard-4") -> Optional[CloudInstance]:
        """Create Google Cloud instance for training"""
        if not self.gcp_client:
            logger.error("Google Cloud client not available")
            return None
        
        try:
            # GPU configuration
            gpu_config = {
                "n1-standard-4": {"gpu_type": "nvidia-tesla-t4", "gpu_count": 1},
                "n1-standard-8": {"gpu_type": "nvidia-tesla-v100", "gpu_count": 1},
                "n1-standard-16": {"gpu_type": "nvidia-tesla-a100", "gpu_count": 1}
            }
            
            gpu_spec = gpu_config.get(gpu_type, {"gpu_type": "nvidia-tesla-t4", "gpu_count": 1})
            
            # Instance configuration
            instance_config = {
                "name": f"smallmind-{model_name.lower().replace('.', '').replace('-', '')}",
                "machine_type": f"zones/{self.cloud_config.region}-a/machineTypes/{gpu_type}",
                "disks": [{
                    "boot": True,
                    "auto_delete": True,
                    "initialize_params": {
                        "source_image": "projects/debian-cloud/global/images/family/debian-11"
                    }
                }],
                "network_interfaces": [{
                    "network": "global/networks/default",
                    "access_configs": [{"name": "External NAT"}]
                }],
                "metadata": {
                    "items": [{
                        "key": "startup-script",
                        "value": f"""
# Install dependencies
pip3 install datasets transformers torch accelerate

# Clone SmallMind
cd /home
git clone https://github.com/smallmind/smallmind.git
cd smallmind

# Start training
python3 -c "
import sys
sys.path.insert(0, 'src')
from smallmind.models.continuous_trainer import train_forever
train_forever('models/models/checkpoints/{model_name.lower()}', './cloud_output')
"
"""
                    }]
                }
            }
            
            # Add GPU configuration if specified
            if gpu_spec["gpu_count"] > 0:
                instance_config["guest_accelerators"] = [{
                    "accelerator_count": gpu_spec["gpu_count"],
                    "accelerator_type": f"zones/{self.cloud_config.region}-a/acceleratorTypes/{gpu_spec['gpu_type']}"
                }]
                # Disable live migration for GPU instances
                instance_config["scheduling"] = {
                    "on_host_maintenance": "TERMINATE"
                }
            
            # Create instance
            operation = self.gcp_client.insert(
                project=self.cloud_config.project_id,  # Use config project ID
                zone=f"{self.cloud_config.region}-a",
                instance_resource=instance_config
            )
            
            # Wait for operation to complete
            operation.result()
            
            # Get instance details
            instance = self.gcp_client.get(
                project=self.cloud_config.project_id,
                zone=f"{self.cloud_config.region}-a",
                instance=instance_config["name"]
            )
            
            # Calculate cost (approximate)
            cost_per_hour = self._get_gcp_instance_cost(gpu_type, gpu_spec["gpu_type"])
            
            cloud_instance = CloudInstance(
                instance_id=instance.id,
                public_ip=instance.network_interfaces[0].access_configs[0].nat_i_p if instance.network_interfaces[0].access_configs else "",
                private_ip=instance.network_interfaces[0].network_i_p,
                status=instance.status,
                platform='gcp',
                instance_type=gpu_type,
                launch_time=datetime.fromisoformat(instance.creation_timestamp.replace('Z', '+00:00')),
                cost_per_hour=cost_per_hour,
                gpu_info=f"{gpu_spec['gpu_type']} x{gpu_spec['gpu_count']}"
            )
            
            logger.info(f"✅ GCP instance created: {instance.id} ({gpu_type})")
            return cloud_instance
            
        except Exception as e:
            logger.error(f"Failed to create GCP instance: {e}")
            return None
    
    def _get_aws_instance_cost(self, instance_type: str) -> float:
        """Get approximate hourly cost for AWS instance"""
        cost_map = {
            "g4dn.xlarge": 0.526,      # $0.526/hour
            "p3.2xlarge": 3.06,        # $3.06/hour
            "g5.xlarge": 1.006,        # $1.006/hour
            "p4d.24xlarge": 32.77      # $32.77/hour
        }
        return cost_map.get(instance_type, 1.0)
    
    def _get_gcp_instance_cost(self, instance_type: str, gpu_type: str) -> float:
        """Get approximate hourly cost for GCP instance"""
        instance_cost = {
            "n1-standard-4": 0.19,
            "n1-standard-8": 0.38,
            "n1-standard-16": 0.76
        }
        
        gpu_cost = {
            "nvidia-tesla-t4": 0.35,
            "nvidia-tesla-v100": 2.48,
            "nvidia-tesla-a100": 2.93
        }
        
        return instance_cost.get(instance_type, 0.5) + gpu_cost.get(gpu_type, 0.5)
    
    def launch_training_instances(self, models: List[str]) -> Dict[str, CloudInstance]:
        """Launch cloud instances for all models"""
        logger.info(f"🚀 Launching cloud instances for {len(models)} models")
        
        launched_instances = {}
        
        for i, model_name in enumerate(models):
            if i >= self.cloud_config.max_instances:
                logger.warning(f"Maximum instances ({self.cloud_config.max_instances}) reached")
                break
            
            logger.info(f"Launching instance for {model_name}...")
            
            if self.cloud_config.platform == "aws":
                instance = self._create_aws_instance(model_name)
            elif self.cloud_config.platform == "gcp":
                instance = self._create_gcp_instance(model_name)
            elif self.cloud_config.platform == "azure":
                # Azure implementation would go here
                logger.warning("Azure not yet implemented")
                continue
            else:
                logger.error(f"Unsupported platform: {self.cloud_config.platform}")
                continue
            
            if instance:
                launched_instances[model_name] = instance
                self.instances[model_name] = instance
                
                # Wait between launches to avoid rate limits
                time.sleep(10)
        
        logger.info(f"✅ Launched {len(launched_instances)} instances")
        return launched_instances
    
    def monitor_training(self) -> Dict[str, Any]:
        """Monitor training progress across all instances"""
        status = {
            "total_instances": len(self.instances),
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cost_per_hour": 0.0,
            "instances": {}
        }
        
        for model_name, instance in self.instances.items():
            # Update instance status
            if self.cloud_config.platform == "aws":
                try:
                    response = self.aws_client.describe_instances(InstanceIds=[instance.instance_id])
                    current_status = response['Reservations'][0]['Instances'][0]['State']['Name']
                    instance.status = current_status
                except Exception as e:
                    logger.warning(f"Failed to get status for {model_name}: {e}")
            
            # acceleratorCount statuses
            if instance.status == "running":
                status["running"] += 1
            elif instance.status == "terminated":
                status["completed"] += 1
            elif instance.status in ["stopped", "stopping"]:
                status["failed"] += 1
            
            # Calculate costs
            status["cost_per_hour"] += instance.cost_per_hour
            
            # Instance details
            status["instances"][model_name] = {
                "instance_id": instance.instance_id,
                "status": instance.status,
                "public_ip": instance.public_ip,
                "gpu_info": instance.gpu_info,
                "cost_per_hour": instance.cost_per_hour,
                "launch_time": instance.launch_time.isoformat()
            }
        
        return status
    
    def stop_all_instances(self):
        """Stop all running instances"""
        logger.info("🛑 Stopping all cloud instances...")
        
        for model_name, instance in self.instances.items():
            if instance.status == "running":
                try:
                    if self.cloud_config.platform == "aws":
                        self.aws_client.terminate_instances(InstanceIds=[instance.instance_id])
                    elif self.cloud_config.platform == "gcp":
                        self.gcp_client.delete(
                            project="your-project-id",
                            zone=f"{self.cloud_config.region}-a",
                            instance=instance.instance_id
                        )
                    
                    logger.info(f"Stopped instance for {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to stop instance for {model_name}: {e}")
        
        logger.info("All instances stopped")
    
    def get_cost_estimate(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost estimate for training"""
        total_cost_per_hour = sum(instance.cost_per_hour for instance in self.instances.values())
        
        return {
            "cost_per_hour": total_cost_per_hour,
            "cost_per_day": total_cost_per_hour * 24,
            "cost_per_week": total_cost_per_hour * 24 * 7,
            "estimated_hours": hours,
            "estimated_total": total_cost_per_hour * hours,
            "instances": [
                {
                    "model": model_name,
                    "instance_acceleratorType": instance.instance_acceleratorType,
                    "gpu_info": instance.gpu_info,
                    "cost_per_hour": instance.cost_per_hour
                }
                for model_name, instance in self.instances.items()
            ]
        }

# Convenience functions
def create_aws_trainer(region: str = "us-east-1", ssh_key: str = "your-key", 
                      security_group: str = "sg-12345678") -> CloudTrainer:
    """Create AWS-based cloud trainer"""
    config = CloudConfig(
        platform="aws",
        region=region,
        instance_acceleratorType="g4dn.xlarge",
        gpu_acceleratorType="g4dn.xlarge",
        gpu_acceleratorCount=1,
        storage_gb=100,
        max_instances=3,
        ssh_key_name=ssh_key,
        security_group=security_group
    )
    return CloudTrainer(config)

def create_gcp_trainer(region: str = "us-central1", project_id: str = "your-project") -> CloudTrainer:
    """Create GCP-based cloud trainer"""
    config = CloudConfig(
        platform="gcp",
        region=region,
        instance_type="n1-standard-4",
        gpu_type="nvidia-tesla-t4",
        gpu_count=1,
        storage_gb=100,
        max_instances=3,
        project_id=project_id
    )
    return CloudTrainer(config)
