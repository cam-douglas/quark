"""
Multi-Cloud Training System for SmallMind

Launches training instances across multiple cloud platforms simultaneously
for better resource availability and redundancy.
"""

import os, sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from datetime import datetime

from .....................................................cloud_integration import CloudTrainer, CloudConfig, create_aws_trainer, create_gcp_trainer

logger = logging.getLogger(__name__)

@dataclass
class MultiCloudConfig:
    """Configuration for multi-cloud training"""
    aws_enabled: bool = True
    gcp_enabled: bool = True
    aws_region: str = "us-east-1"
    gcp_region: str = "us-central1"
    max_instances_per_platform: int = 2
    models_per_platform: int = 1  # Distribute models across platforms

class MultiCloudTrainer:
    """Multi-cloud training system that launches instances across platforms"""
    
    def __init__(self, config: MultiCloudConfig, output_dir: str = "./multi_cloud_training"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cloud trainers
        self.aws_trainer = None
        self.gcp_trainer = None
        self.instances = {}
        self.training_status = {}
        
        self._init_cloud_trainers()
        
        logger.info("MultiCloudTrainer initialized")
    
    def _init_cloud_trainers(self):
        """Initialize cloud trainers for each platform"""
        if self.config.aws_enabled:
            try:
                # Check if AWS credentials are available
                if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
                    self.aws_trainer = create_aws_trainer(
                        region=self.config.aws_region,
                        ssh_key=os.getenv('AWS_SSH_KEY_NAME', 'default'),
                        security_group=os.getenv('AWS_SECURITY_GROUP_ID', 'default')
                    )
                    logger.info("✅ AWS trainer initialized")
                else:
                    logger.warning("AWS credentials not found, AWS disabled")
                    self.config.aws_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize AWS trainer: {e}")
                self.config.aws_enabled = False
        
        if self.config.gcp_enabled:
            try:
                # Check if GCP credentials are available
                if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                    self.gcp_trainer = create_gcp_trainer(
                        region=self.config.gcp_region,
                        project_id=os.getenv('GOOGLE_CLOUD_PROJECT', 'default')
                    )
                    logger.info("✅ GCP trainer initialized")
                else:
                    logger.warning("GCP credentials not found, GCP disabled")
                    self.config.gcp_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize GCP trainer: {e}")
                self.config.gcp_enabled = False
        
        if not self.config.aws_enabled and not self.config.gcp_enabled:
            raise RuntimeError("No cloud platforms available")
    
    def distribute_models(self, models: List[str]) -> Dict[str, List[str]]:
        """Distribute models across available cloud platforms"""
        distribution = {}
        
        if self.config.aws_enabled:
            aws_models = models[:self.config.models_per_platform]
            distribution['aws'] = aws_models
            models = models[self.config.models_per_platform:]
        
        if self.config.gcp_enabled:
            gcp_models = models[:self.config.models_per_platform]
            distribution['gcp'] = gcp_models
            models = models[self.config.models_per_platform:]
        
        # If we have more models than platforms, distribute remaining
        remaining_models = models
        while remaining_models:
            if self.config.aws_enabled and len(distribution.get('aws', [])) < self.config.max_instances_per_platform:
                distribution.setdefault('aws', []).append(remaining_models.pop(0))
            elif self.config.gcp_enabled and len(distribution.get('gcp', [])) < self.config.max_instances_per_platform:
                distribution.setdefault('gcp', []).append(remaining_models.pop(0))
            else:
                break
        
        return distribution
    
    def launch_multi_cloud_training(self, models: List[str]) -> Dict[str, Any]:
        """Launch training instances across multiple cloud platforms"""
        logger.info(f"🚀 Launching multi-cloud training for {len(models)} models")
        
        # Distribute models across platforms
        distribution = self.distribute_models(models)
        logger.info(f"Model distribution: {distribution}")
        
        launched_instances = {}
        
        # Launch AWS instances
        if 'aws' in distribution and self.aws_trainer:
            logger.info(f"Launching {len(distribution['aws'])} models on AWS")
            try:
                aws_instances = self.aws_trainer.launch_training_instances(distribution['aws'])
                launched_instances['aws'] = aws_instances
                logger.info(f"✅ Launched {len(aws_instances)} AWS instances")
            except Exception as e:
                logger.error(f"Failed to launch AWS instances: {e}")
        
        # Launch GCP instances
        if 'gcp' in distribution and self.gcp_trainer:
            logger.info(f"Launching {len(distribution['gcp'])} models on GCP")
            try:
                gcp_instances = self.gcp_trainer.launch_training_instances(distribution['gcp'])
                launched_instances['gcp'] = gcp_instances
                logger.info(f"✅ Launched {len(gcp_instances)} GCP instances")
            except Exception as e:
                logger.error(f"Failed to launch GCP instances: {e}")
        
        # Combine all instances
        all_instances = {}
        for platform, instances in launched_instances.items():
            for model_name, instance in instances.items():
                all_instances[f"{platform}_{model_name}"] = instance
        
        self.instances = all_instances
        
        logger.info(f"🎉 Multi-cloud training launched: {len(all_instances)} total instances")
        return launched_instances
    
    def monitor_multi_cloud_training(self) -> Dict[str, Any]:
        """Monitor training progress across all cloud platforms"""
        status = {
            "total_instances": len(self.instances),
            "platforms": {},
            "overall_status": {
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cost_per_hour": 0.0
            }
        }
        
        # Monitor AWS instances
        if self.aws_trainer:
            aws_status = self.aws_trainer.monitor_training()
            status["platforms"]["aws"] = aws_status
            status["overall_status"]["running"] += aws_status.get("running", 0)
            status["overall_status"]["completed"] += aws_status.get("completed", 0)
            status["overall_status"]["failed"] += aws_status.get("failed", 0)
            status["overall_status"]["cost_per_hour"] += aws_status.get("cost_per_hour", 0.0)
        
        # Monitor GCP instances
        if self.gcp_trainer:
            gcp_status = self.gcp_trainer.monitor_training()
            status["platforms"]["gcp"] = gcp_status
            status["overall_status"]["running"] += gcp_status.get("running", 0)
            status["overall_status"]["completed"] += gcp_status.get("completed", 0)
            status["overall_status"]["failed"] += gcp_status.get("failed", 0)
            status["overall_status"]["cost_per_hour"] += gcp_status.get("cost_per_hour", 0.0)
        
        return status
    
    def stop_all_instances(self):
        """Stop all instances across all platforms"""
        logger.info("🛑 Stopping all multi-cloud instances...")
        
        if self.aws_trainer:
            self.aws_trainer.stop_all_instances()
        
        if self.gcp_trainer:
            self.gcp_trainer.stop_all_instances()
        
        logger.info("All multi-cloud instances stopped")
    
    def get_multi_cloud_cost_estimate(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost estimate across all cloud platforms"""
        total_cost_per_hour = 0.0
        platform_costs = {}
        
        if self.aws_trainer:
            aws_cost = self.aws_trainer.get_cost_estimate(hours)
            platform_costs["aws"] = aws_cost
            total_cost_per_hour += aws_cost["cost_per_hour"]
        
        if self.gcp_trainer:
            gcp_cost = self.gcp_trainer.get_cost_estimate(hours)
            platform_costs["gcp"] = gcp_cost
            total_cost_per_hour += gcp_cost["cost_per_hour"]
        
        return {
            "total_cost_per_hour": total_cost_per_hour,
            "total_cost_per_day": total_cost_per_hour * 24,
            "total_cost_per_week": total_cost_per_hour * 24 * 7,
            "total_cost_per_month": total_cost_per_hour * 24 * 30,
            "estimated_hours": hours,
            "estimated_total": total_cost_per_hour * hours,
            "platform_costs": platform_costs
        }

# Convenience functions
def create_multi_cloud_trainer(
    aws_enabled: bool = True,
    gcp_enabled: bool = True,
    aws_region: str = "us-east-1",
    gcp_region: str = "us-central1",
    max_instances_per_platform: int = 2
) -> MultiCloudTrainer:
    """Create a multi-cloud trainer"""
    config = MultiCloudConfig(
        aws_enabled=aws_enabled,
        gcp_enabled=gcp_enabled,
        aws_region=aws_region,
        gcp_region=gcp_region,
        max_instances_per_platform=max_instances_per_platform
    )
    return MultiCloudTrainer(config)

def launch_hybrid_training(models: List[str], **kwargs) -> MultiCloudTrainer:
    """Quick function to launch hybrid AWS + GCP training"""
    trainer = create_multi_cloud_trainer(**kwargs)
    trainer.launch_multi_cloud_training(models)
    return trainer
