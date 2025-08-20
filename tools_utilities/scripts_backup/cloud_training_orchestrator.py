#!/usr/bin/env python3
"""
Cloud Training Orchestrator for Exponential Learning System
Manages parallel model training across multiple cloud platforms
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import subprocess
import os
import yaml
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TrainingJob:
    """Represents a training job configuration"""
    job_id: str
    model_name: str
    platform: str
    region: str
    instance_type: str
    training_script: str
    hyperparameters: Dict[str, Any]
    data_path: str
    output_path: str
    status: str = "pending"
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.logs is None:
            self.logs = []
        if self.metrics is None:
            self.metrics = {}

@dataclass
class CloudPlatform:
    """Represents a cloud platform configuration"""
    name: str
    provider: str
    regions: List[str]
    instance_types: List[str]
    credentials_path: str
    max_instances: int
    cost_per_hour: float

class CloudTrainingOrchestrator:
    """
    Orchestrates training across multiple cloud platforms
    Manages resource allocation, cost optimization, and parallel execution
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "cloud_training_config.yaml"
        self.platforms = {}
        self.active_jobs = {}
        self.job_queue = []
        self.max_concurrent_jobs = 10
        self.cost_budget = 100.0  # Daily budget in USD
        self.training_scripts_dir = Path("training_scripts")
        
        # Load configuration
        self.load_configuration()
        
        # Initialize platforms
        self.initialize_platforms()
        
        logger.info("☁️ Cloud Training Orchestrator initialized")
    
    def load_configuration(self):
        """Load cloud platform configurations"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                for platform_config in config.get('platforms', []):
                    platform = CloudPlatform(**platform_config)
                    self.platforms[platform.name] = platform
                
                self.max_concurrent_jobs = config.get('max_concurrent_jobs', 10)
                self.cost_budget = config.get('daily_budget', 100.0)
                
                logger.info(f"✅ Loaded {len(self.platforms)} cloud platforms")
            else:
                # Create default configuration
                self.create_default_config()
                
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create default cloud platform configuration"""
        default_config = {
            'platforms': [
                {
                    'name': 'aws_ec2',
                    'provider': 'aws',
                    'regions': ['us-east-1', 'us-west-2', 'eu-west-1'],
                    'instance_types': ['t3.medium', 't3.large', 'c5.large'],
                    'credentials_path': '~/.aws/credentials',
                    'max_instances': 5,
                    'cost_per_hour': 0.05
                },
                {
                    'name': 'gcp_compute',
                    'provider': 'gcp',
                    'regions': ['us-central1', 'europe-west1', 'asia-east1'],
                    'instance_types': ['n1-standard-2', 'n1-standard-4', 'n1-highmem-2'],
                    'credentials_path': '~/.config/gcloud/application_default_credentials.json',
                    'max_instances': 5,
                    'cost_per_hour': 0.06
                },
                {
                    'name': 'azure_vm',
                    'provider': 'azure',
                    'regions': ['eastus', 'westus2', 'westeurope'],
                    'instance_types': ['Standard_B2s', 'Standard_B4ms', 'Standard_D2s_v3'],
                    'credentials_path': '~/.azure/credentials',
                    'max_instances': 5,
                    'cost_per_hour': 0.07
                }
            ],
            'max_concurrent_jobs': 10,
            'daily_budget': 100.0
        }
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        # Load the default configuration
        for platform_config in default_config['platforms']:
            platform = CloudPlatform(**platform_config)
            self.platforms[platform.name] = platform
        
        logger.info("✅ Created and loaded default configuration")
    
    def initialize_platforms(self):
        """Initialize cloud platform connections"""
        for name, platform in self.platforms.items():
            try:
                if platform.provider == 'aws':
                    self.initialize_aws_platform(platform)
                elif platform.provider == 'gcp':
                    self.initialize_gcp_platform(platform)
                elif platform.provider == 'azure':
                    self.initialize_azure_platform(platform)
                
                logger.info(f"✅ Initialized {name} platform")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize {name}: {e}")
    
    def initialize_aws_platform(self, platform: CloudPlatform):
        """Initialize AWS platform"""
        # Check AWS CLI installation
        try:
            subprocess.run(['aws', '--version'], check=True, capture_output=True)
            logger.info(f"✅ AWS CLI found for {platform.name}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"⚠️ AWS CLI not found for {platform.name}")
    
    def initialize_gcp_platform(self, platform: CloudPlatform):
        """Initialize GCP platform"""
        # Check gcloud CLI installation
        try:
            subprocess.run(['gcloud', '--version'], check=True, capture_output=True)
            logger.info(f"✅ Google Cloud CLI found for {platform.name}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"⚠️ Google Cloud CLI not found for {platform.name}")
    
    def initialize_azure_platform(self, platform: CloudPlatform):
        """Initialize Azure platform"""
        # Check Azure CLI installation
        try:
            subprocess.run(['az', '--version'], check=True, capture_output=True)
            logger.info(f"✅ Azure CLI found for {platform.name}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"⚠️ Azure CLI not found for {platform.name}")
    
    async def submit_training_job(self, model_name: str, training_config: Dict[str, Any]) -> str:
        """Submit a new training job"""
        # Generate job ID
        job_id = f"train_{model_name}_{int(time.time())}"
        
        # Select optimal platform and configuration
        platform, region, instance_type = self.select_optimal_configuration(training_config)
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            platform=platform.name,
            region=region,
            instance_type=instance_type,
            training_script=training_config.get('script', f'train_{model_name}.py'),
            hyperparameters=training_config.get('hyperparameters', {}),
            data_path=training_config.get('data_path', f'data/{model_name}'),
            output_path=training_config.get('output_path', f'outputs/{model_name}')
        )
        
        # Add to queue
        self.job_queue.append(job)
        self.active_jobs[job_id] = job
        
        logger.info(f"📝 Submitted training job {job_id} for {model_name} on {platform.name}")
        
        # Start processing if capacity available
        await self.process_job_queue()
        
        return job_id
    
    def select_optimal_configuration(self, training_config: Dict[str, Any]) -> tuple:
        """Select optimal platform, region, and instance type"""
        best_platform = None
        best_region = None
        best_instance = None
        best_score = float('-inf')
        
        for platform in self.platforms.values():
            # Check if platform has capacity
            active_jobs_on_platform = sum(
                1 for job in self.active_jobs.values() 
                if job.platform == platform.name and job.status in ['running', 'starting']
            )
            
            if active_jobs_on_platform >= platform.max_instances:
                continue
            
            for region in platform.regions:
                for instance_type in platform.instance_types:
                    # Calculate score based on cost, performance, and availability
                    score = self.calculate_configuration_score(
                        platform, region, instance_type, training_config
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_platform = platform
                        best_region = region
                        best_instance = instance_type
        
        if best_platform is None:
            # Fallback to first available platform
            best_platform = list(self.platforms.values())[0]
            best_region = best_platform.regions[0]
            best_instance = best_platform.instance_types[0]
        
        return best_platform, best_region, best_instance
    
    def calculate_configuration_score(self, platform: CloudPlatform, region: str, 
                                   instance_type: str, training_config: Dict[str, Any]) -> float:
        """Calculate score for a platform configuration"""
        score = 0.0
        
        # Cost factor (lower is better)
        cost_factor = 1.0 / (platform.cost_per_hour + 0.01)
        score += cost_factor * 0.4
        
        # Performance factor (higher is better)
        if 'large' in instance_type or 'highmem' in instance_type:
            score += 0.3
        elif 'medium' in instance_type:
            score += 0.2
        else:
            score += 0.1
        
        # Availability factor
        active_jobs = sum(
            1 for job in self.active_jobs.values() 
            if job.platform == platform.name and job.status in ['running', 'starting']
        )
        availability = 1.0 - (active_jobs / platform.max_instances)
        score += availability * 0.3
        
        return score
    
    async def process_job_queue(self):
        """Process pending jobs in the queue"""
        while self.job_queue and len([j for j in self.active_jobs.values() if j.status in ['running', 'starting']]) < self.max_concurrent_jobs:
            job = self.job_queue.pop(0)
            await self.start_training_job(job)
    
    async def start_training_job(self, job: TrainingJob):
        """Start a training job on the selected platform"""
        try:
            job.status = "starting"
            job.started_at = datetime.now()
            
            logger.info(f"🚀 Starting training job {job.job_id} on {job.platform}")
            
            # Create training script
            script_path = self.create_training_script(job)
            
            # Launch on cloud platform
            if job.platform.startswith('aws'):
                await self.launch_aws_job(job, script_path)
            elif job.platform.startswith('gcp'):
                await self.launch_gcp_job(job, script_path)
            elif job.platform.startswith('azure'):
                await self.launch_azure_job(job, script_path)
            else:
                raise ValueError(f"Unknown platform: {job.platform}")
            
            job.status = "running"
            logger.info(f"✅ Training job {job.job_id} started successfully")
            
        except Exception as e:
            job.status = "failed"
            job.logs.append(f"Failed to start job: {str(e)}")
            logger.error(f"❌ Failed to start training job {job.job_id}: {e}")
    
    def create_training_script(self, job: TrainingJob) -> str:
        """Create training script for the job"""
        # Ensure training scripts directory exists
        self.training_scripts_dir.mkdir(exist_ok=True)
        
        script_path = self.training_scripts_dir / f"{job.job_id}.py"
        
        # Generate training script content
        script_content = self.generate_training_script_content(job)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def generate_training_script_content(self, job: TrainingJob) -> str:
        """Generate Python training script content"""
        script = f'''#!/usr/bin/env python3
"""
Training script for {job.model_name}
Job ID: {job.job_id}
Platform: {job.platform}
"""

import os, sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.smallmind.models.exponential_learning.exponential_learning_system import ExponentialLearningSystem
from src.smallmind.models.exponential_learning.research_agents import ResearchAgentHub
from src.smallmind.models.exponential_learning.knowledge_synthesizer import KnowledgeSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    logger.info(f"🚀 Starting training for {job.model_name}")
    logger.info(f"Job ID: {job.job_id}")
    logger.info(f"Platform: {job.platform}")
    
    try:
        # Initialize components
        learning_system = ExponentialLearningSystem()
        research_hub = ResearchAgentHub()
        synthesizer = KnowledgeSynthesizer()
        
        # Start research and learning
        logger.info("🔍 Starting research and learning process...")
        
        # Run learning cycles
        for cycle in range(10):  # Run 10 learning cycles
            logger.info(f"🔄 Learning cycle {cycle + 1}")
            
            # Research new topics
            research_topics = [
                "artificial intelligence",
                "machine learning",
                "neural networks",
                "deep learning",
                "reinforcement learning"
            ]
            
            for topic in research_topics:
                logger.info(f"🔍 Researching: {topic}")
                # Simulate research (replace with actual research calls)
                await research_hub.search_all_sources(topic)
            
            # Synthesize knowledge
            logger.info("🔬 Synthesizing knowledge...")
            # Simulate synthesis (replace with actual synthesis calls)
            
            # Update learning system
            learning_system.learning_cycles += 1
            learning_system.grow_learning_capacity()
            
            logger.info(f"✅ Cycle {cycle + 1} completed")
        
        logger.info("🎉 Training completed successfully!")
        
        # Save results
        results = {{
            "job_id": "{job.job_id}",
            "model_name": "{job.model_name}",
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "learning_cycles": learning_system.learning_cycles,
            "knowledge_base_size": len(learning_system.knowledge_base)
        }}
        
        output_dir = Path("{job.output_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to {{output_dir}}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return script
    
    async def launch_aws_job(self, job: TrainingJob, script_path: str):
        """Launch training job on AWS EC2"""
        try:
            # Create EC2 instance using AWS CLI
            cmd = [
                'aws', 'ec2', 'run-instances',
                '--region', job.region,
                '--instance-type', job.instance_type,
                '--image-id', 'ami-0c02fb55956c7d316',  # Ubuntu 20.04 LTS
                '--key-name', 'smallmind-key',  # You'll need to create this
                '--security-group-ids', 'sg-0123456789abcdef0',  # You'll need to create this
                '--subnet-id', 'subnet-0123456789abcdef0',  # You'll need to create this
                '--user-data', f'#!/bin/bash\ncd HOME/ubuntu\npython3 {script_path}',
                '--tag-specifications', f'ResourceType=instance,Tags=[{{Key=Name,Value={job.job_id}}}]'
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse instance ID from output
                instance_id = self.parse_aws_instance_id(result.stdout)
                job.metrics['instance_id'] = instance_id
                logger.info(f"✅ AWS instance {instance_id} launched for job {job.job_id}")
            else:
                raise Exception(f"AWS CLI error: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to launch AWS job: {e}")
    
    async def launch_gcp_job(self, job: TrainingJob, script_path: str):
        """Launch training job on Google Cloud Compute"""
        try:
            # Create VM instance using gcloud CLI
            cmd = [
                'gcloud', 'compute', 'instances', 'create', job.job_id,
                '--zone', f'{job.region}-a',
                '--machine-type', job.instance_type,
                '--image-family', 'ubuntu-2004-lts',
                '--image-project', 'ubuntu-os-cloud',
                '--metadata', f'startup-script=cd HOME/ubuntu && python3 {script_path}',
                '--tags', 'smallmind-training'
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ GCP instance {job.job_id} launched successfully")
            else:
                raise Exception(f"gcloud CLI error: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to launch GCP job: {e}")
    
    async def launch_azure_job(self, job: TrainingJob, script_path: str):
        """Launch training job on Azure VM"""
        try:
            # Create VM using Azure CLI
            cmd = [
                'az', 'vm', 'create',
                '--resource-group', 'smallmind-rg',
                '--name', job.job_id,
                '--image', 'Canonical:0001-com-ubuntu-server-focal:20_04-lts:latest',
                '--size', job.instance_type,
                '--admin-username', 'ubuntu',
                '--generate-ssh-keys',
                '--custom-data', script_path
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Azure VM {job.job_id} created successfully")
            else:
                raise Exception(f"Azure CLI error: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to launch Azure job: {e}")
    
    def parse_aws_instance_id(self, output: str) -> str:
        """Parse instance ID from AWS CLI output"""
        # Simple parsing - in production, use proper JSON parsing
        lines = output.split('\n')
        for line in lines:
            if 'InstanceId' in line:
                return line.split()[-1].strip('"')
        return "unknown"
    
    async def monitor_jobs(self):
        """Monitor active training jobs"""
        while True:
            try:
                for job_id, job in list(self.active_jobs.items()):
                    if job.status == "running":
                        # Check job status on cloud platform
                        await self.check_job_status(job)
                    
                    elif job.status == "completed":
                        # Clean up completed job
                        await self.cleanup_completed_job(job)
                        del self.active_jobs[job_id]
                
                # Process queue
                await self.process_job_queue()
                
                # Log status
                running_jobs = len([j for j in self.active_jobs.values() if j.status == "running"])
                pending_jobs = len(self.job_queue)
                logger.info(f"📊 Status: {running_jobs} running, {pending_jobs} pending")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in job monitoring: {e}")
                await asyncio.sleep(60)
    
    async def check_job_status(self, job: TrainingJob):
        """Check status of a running job"""
        try:
            if job.platform.startswith('aws'):
                await self.check_aws_job_status(job)
            elif job.platform.startswith('gcp'):
                await self.check_gcp_job_status(job)
            elif job.platform.startswith('azure'):
                await self.check_azure_job_status(job)
                
        except Exception as e:
            logger.error(f"❌ Error checking job {job.job_id} status: {e}")
    
    async def check_aws_job_status(self, job: TrainingJob):
        """Check AWS job status"""
        try:
            instance_id = job.metrics.get('instance_id')
            if not instance_id:
                return
            
            cmd = ['aws', 'ec2', 'describe-instances', '--instance-ids', instance_id, '--region', job.region]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse status from output
                if 'stopped' in result.stdout.lower():
                    job.status = "completed"
                    job.completed_at = datetime.now()
                    logger.info(f"✅ Job {job.job_id} completed on AWS")
                    
        except Exception as e:
            logger.error(f"❌ Error checking AWS job status: {e}")
    
    async def check_gcp_job_status(self, job: TrainingJob):
        """Check GCP job status"""
        try:
            cmd = ['gcloud', 'compute', 'instances', 'describe', job.job_id, '--zone', f'{job.region}-a']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse status from output
                if 'TERMINATED' in result.stdout:
                    job.status = "completed"
                    job.completed_at = datetime.now()
                    logger.info(f"✅ Job {job.job_id} completed on GCP")
                    
        except Exception as e:
            logger.error(f"❌ Error checking GCP job status: {e}")
    
    async def check_azure_job_status(self, job: TrainingJob):
        """Check Azure job status"""
        try:
            cmd = ['az', 'vm', 'show', '--resource-group', 'smallmind-rg', '--name', job.job_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse status from output
                if 'deallocated' in result.stdout.lower():
                    job.status = "completed"
                    job.completed_at = datetime.now()
                    logger.info(f"✅ Job {job.job_id} completed on Azure")
                    
        except Exception as e:
            logger.error(f"❌ Error checking Azure job status: {e}")
    
    async def cleanup_completed_job(self, job: TrainingJob):
        """Clean up completed job resources"""
        try:
            if job.platform.startswith('aws'):
                await self.cleanup_aws_job(job)
            elif job.platform.startswith('gcp'):
                await self.cleanup_gcp_job(job)
            elif job.platform.startswith('azure'):
                await self.cleanup_azure_job(job)
                
            logger.info(f"🧹 Cleaned up completed job {job.job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up job {job.job_id}: {e}")
    
    async def cleanup_aws_job(self, job: TrainingJob):
        """Clean up AWS job resources"""
        try:
            instance_id = job.metrics.get('instance_id')
            if instance_id:
                cmd = ['aws', 'ec2', 'terminate-instances', '--instance-ids', instance_id, '--region', job.region]
                subprocess.run(cmd, capture_output=True)
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up AWS job: {e}")
    
    async def cleanup_gcp_job(self, job: TrainingJob):
        """Clean up GCP job resources"""
        try:
            cmd = ['gcloud', 'compute', 'instances', 'delete', job.job_id, '--zone', f'{job.region}-a', '--quiet']
            subprocess.run(cmd, capture_output=True)
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up GCP job: {e}")
    
    async def cleanup_azure_job(self, job: TrainingJob):
        """Clean up Azure job resources"""
        try:
            cmd = ['az', 'vm', 'delete', '--resource-group', 'smallmind-rg', '--name', job.job_id, '--yes']
            subprocess.run(cmd, capture_output=True)
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up Azure job: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return asdict(job)
        return None
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get status of all jobs"""
        return [asdict(job) for job in self.active_jobs.values()]
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary of active jobs"""
        total_cost = 0.0
        platform_costs = defaultdict(float)
        
        for job in self.active_jobs.values():
            if job.status in ['running', 'starting']:
                platform = self.platforms.get(job.platform)
                if platform:
                    # Calculate cost based on runtime
                    runtime_hours = 0
                    if job.started_at:
                        runtime = datetime.now() - job.started_at
                        runtime_hours = runtime.total_seconds() / 3600
                    
                    cost = runtime_hours * platform.cost_per_hour
                    total_cost += cost
                    platform_costs[job.platform] += cost
        
        return {
            "total_cost": total_cost,
            "platform_costs": dict(platform_costs),
            "daily_budget": self.cost_budget,
            "budget_remaining": self.cost_budget - total_cost
        }

async def main():
    """Test the cloud training orchestrator"""
    orchestrator = CloudTrainingOrchestrator()
    
    # Submit a test training job
    training_config = {
        "script": "train_test_model.py",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        "data_path": "data/test_model",
        "output_path": "outputs/test_model"
    }
    
    job_id = await orchestrator.submit_training_job("test_model", training_config)
    print(f"📝 Submitted training job: {job_id}")
    
    # Start monitoring
    print("🔍 Starting job monitoring...")
    await orchestrator.monitor_jobs()

if __name__ == "__main__":
    asyncio.run(main())
