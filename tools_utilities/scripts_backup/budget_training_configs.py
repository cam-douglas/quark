#!/usr/bin/env python3
"""
Budget-Friendly Wikipedia Training Configurations
=================================================

Cost-effective alternatives for Wikipedia training with significant savings.
Multiple budget tiers from free to low-cost cloud options.

Purpose: Provide affordable Wikipedia training options
Inputs: Budget constraints, performance requirements
Outputs: Optimized configurations for cost savings
Seeds: N/A (configuration file)
Dependencies: None (configuration only)
"""

from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from typing import Dict, List, Optional
import json


@dataclass
class BudgetTrainingConfig:
    """Budget-optimized training configuration."""
    
    name: str
    description: str
    cost_per_hour: str
    estimated_total_cost: str
    training_time: str
    
    # Cloud configuration
    cloud_provider: str
    instance_type: str
    num_nodes: int
    gpu_type: str
    memory_gb: int
    storage_gb: int
    
    # Training parameters
    model_size: str
    batch_size: int
    gradient_accumulation_steps: int
    max_articles: Optional[int]
    use_fp16: bool
    use_gradient_checkpointing: bool
    
    # Cost optimization features
    use_spot_instances: bool
    auto_shutdown: bool
    use_preemptible: bool
    storage_tier: str


# Budget Configuration Tiers
BUDGET_CONFIGS = {
    "free_tier": BudgetTrainingConfig(
        name="Free Tier - Google Colab",
        description="Completely free using Google Colab Pro with limitations",
        cost_per_hour="$0",
        estimated_total_cost="$0 (with time limits)",
        training_time="~20 hours (with interruptions)",
        
        cloud_provider="colab",
        instance_type="colab-gpu",
        num_nodes=1,
        gpu_type="T4 or V100 (variable)",
        memory_gb=13,
        storage_gb=50,
        
        model_size="small (117M params)",
        batch_size=2,
        gradient_accumulation_steps=32,
        max_articles=100000,  # Limited subset
        use_fp16=True,
        use_gradient_checkpointing=True,
        
        use_spot_instances=False,
        auto_shutdown=True,
        use_preemptible=False,
        storage_tier="free"
    ),
    
    "ultra_budget": BudgetTrainingConfig(
        name="Ultra Budget - Spot Instances",
        description="AWS/GCP spot instances with aggressive cost optimization",
        cost_per_hour="$0.15-0.30",
        estimated_total_cost="$8-15",
        training_time="~24-36 hours",
        
        cloud_provider="aws",
        instance_type="g4dn.xlarge",  # Much cheaper than p3.8xlarge
        num_nodes=1,
        gpu_type="T4 (16GB)",
        memory_gb=16,
        storage_gb=125,
        
        model_size="small (117M params)",
        batch_size=4,
        gradient_accumulation_steps=16,
        max_articles=500000,  # Half Wikipedia
        use_fp16=True,
        use_gradient_checkpointing=True,
        
        use_spot_instances=True,
        auto_shutdown=True,
        use_preemptible=True,
        storage_tier="cold"
    ),
    
    "low_budget": BudgetTrainingConfig(
        name="Low Budget - Multi-Node Spot",
        description="2 spot instances for faster training at low cost",
        cost_per_hour="$0.30-0.60",
        estimated_total_cost="$15-30",
        training_time="~12-20 hours",
        
        cloud_provider="aws",
        instance_type="g4dn.2xlarge",
        num_nodes=2,
        gpu_type="T4 (16GB each)",
        memory_gb=32,
        storage_gb=225,
        
        model_size="medium (345M params)",
        batch_size=4,
        gradient_accumulation_steps=8,
        max_articles=1000000,  # Most of Wikipedia
        use_fp16=True,
        use_gradient_checkpointing=True,
        
        use_spot_instances=True,
        auto_shutdown=True,
        use_preemptible=True,
        storage_tier="standard"
    ),
    
    "balanced_budget": BudgetTrainingConfig(
        name="Balanced Budget - Mixed Instances",
        description="Mix of spot and on-demand for reliability",
        cost_per_hour="$0.50-1.00",
        estimated_total_cost="$25-50",
        training_time="~8-15 hours",
        
        cloud_provider="gcp",
        instance_type="n1-standard-8",
        num_nodes=2,
        gpu_type="V100 (16GB each)",
        memory_gb=60,
        storage_gb=375,
        
        model_size="medium (345M params)",
        batch_size=6,
        gradient_accumulation_steps=6,
        max_articles=None,  # Full Wikipedia
        use_fp16=True,
        use_gradient_checkpointing=True,
        
        use_spot_instances=True,
        auto_shutdown=True,
        use_preemptible=True,
        storage_tier="standard"
    ),
    
    "performance_budget": BudgetTrainingConfig(
        name="Performance Budget - Optimized Spot",
        description="Best performance per dollar with A100 spots",
        cost_per_hour="$1.00-2.00",
        estimated_total_cost="$50-100",
        training_time="~4-8 hours",
        
        cloud_provider="aws",
        instance_type="p4d.xlarge",
        num_nodes=2,
        gpu_type="A100 (40GB each)",
        memory_gb=96,
        storage_gb=400,
        
        model_size="large (774M params)",
        batch_size=8,
        gradient_accumulation_steps=4,
        max_articles=None,  # Full Wikipedia
        use_fp16=True,
        use_gradient_checkpointing=False,  # A100 has enough memory
        
        use_spot_instances=True,
        auto_shutdown=True,
        use_preemptible=True,
        storage_tier="ssd"
    )
}


def get_cost_comparison():
    """Compare costs with original expensive configuration."""
    original_cost = {
        "name": "Original Configuration",
        "instance_type": "4 x p3.8xlarge",
        "cost_per_hour": "$12.24",
        "estimated_total_cost": "$400-500",
        "training_time": "~24 hours"
    }
    
    print("💰 COST COMPARISON: Budget vs Original")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Cost/Hour':<12} {'Total Cost':<15} {'Savings'}")
    print("-" * 60)
    
    # Original
    print(f"{'Original (4xp3.8xl)':<25} {'$12.24':<12} {'$400-500':<15} {'Baseline'}")
    
    # Budget options
    for config in BUDGET_CONFIGS.values():
        savings_pct = "99%" if config.cost_per_hour == "$0" else "95-98%"
        print(f"{config.name[:24]:<25} {config.cost_per_hour:<12} {config.estimated_total_cost:<15} {savings_pct}")
    
    print("\n🎯 RECOMMENDATION: Start with 'Ultra Budget' for testing,")
    print("   then move to 'Balanced Budget' for full Wikipedia training.")


def generate_deployment_config(config_name: str) -> Dict:
    """Generate deployment configuration for budget option."""
    if config_name not in BUDGET_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(BUDGET_CONFIGS.keys())}")
    
    config = BUDGET_CONFIGS[config_name]
    
    # Base deployment config
    deployment_config = {
        "cloud_provider": config.cloud_provider,
        "region": "us-west-2" if config.cloud_provider == "aws" else "us-central1",
        "cluster_name": f"quark-budget-{config_name}",
        "node_count": config.num_nodes,
        "instance_type": config.instance_type,
        "container_image": "quark/wikipedia-training:budget",
        "storage_bucket": f"quark-budget-{config_name}",
        "cache_size_gb": config.storage_gb,
        "output_size_gb": 50,
        "enable_monitoring": True,
        
        # Budget optimizations
        "use_spot_instances": config.use_spot_instances,
        "auto_shutdown_minutes": 60 if config.auto_shutdown else None,
        "storage_class": config.storage_tier,
        "enable_autoscaling": True,
        "min_nodes": 0,
        "max_nodes": config.num_nodes * 2
    }
    
    # Training config
    training_config = {
        "model_name": {
            "small": "microsoft/DialoGPT-small",
            "medium": "microsoft/DialoGPT-medium", 
            "large": "microsoft/DialoGPT-large"
        }.get(config.model_size.split()[0], "microsoft/DialoGPT-small"),
        
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_articles": config.max_articles,
        "fp16": config.use_fp16,
        "gradient_checkpointing": config.use_gradient_checkpointing,
        
        # Aggressive optimization for budget
        "num_epochs": 2 if config.max_articles else 3,
        "learning_rate": 3e-5,  # Slightly higher for faster convergence
        "warmup_steps": 2000,
        "save_steps": 10000,  # Less frequent saves
        "eval_steps": 20000,   # Less frequent evaluation
        "logging_steps": 500,
        
        # Memory optimization
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False if config.memory_gb < 32 else True,
        "max_grad_norm": 1.0,
        
        # Cache optimization
        "preprocessing_workers": 4,
        "cache_dir": "/tmp/cache" if config.storage_gb < 100 else "/mnt/cache",
        "output_dir": "/tmp/models" if config.storage_gb < 100 else "/mnt/models"
    }
    
    return {
        "deployment": deployment_config,
        "training": training_config,
        "config_info": {
            "name": config.name,
            "description": config.description,
            "estimated_cost": config.estimated_total_cost,
            "training_time": config.training_time
        }
    }


def create_colab_notebook_config():
    """Create configuration for free Google Colab training."""
    return {
        "notebook_name": "quark_wikipedia_training_free.ipynb",
        "description": "Free Wikipedia training on Google Colab",
        "cells": [
            {
                "type": "markdown",
                "content": """
# 🧠 Quark Wikipedia Training - FREE Edition

Train on Wikipedia completely free using Google Colab GPU!

## Features:
- ✅ Free Google Colab GPU (T4/V100)
- ✅ Up to 100K Wikipedia articles
- ✅ Consciousness integration ready
- ✅ No cloud costs
- ⚠️ 12-hour session limits (need to restart)
"""
            },
            {
                "type": "code",
                "content": """
# Install dependencies
!pip install torch transformers datasets accelerate wandb
!pip install wikipedia-api beautifulsoup4 lxml

# Clone Quark repository
!git clone https://github.com/your-username/quark.git
%cd quark

# Import training modules
import sys
sys.path.append('/content/quark')
from data_knowledge.datasets_knowledge.datasets_knowledge.knowledge_systems.training_pipelines.wikipedia_cloud_training import *
"""
            },
            {
                "type": "code", 
                "content": """
# Configure for Colab (free tier)
config = WikipediaTrainingConfig(
    model_name="microsoft/DialoGPT-small",
    max_articles=100000,  # Limited for free tier
    batch_size=2,
    gradient_accumulation_steps=32,
    num_epochs=2,
    fp16=True,
    gradient_checkpointing=True,
    cache_dir="/content/cache",
    output_dir="/content/models",
    preprocessing_workers=2,
    wandb_project="quark-wikipedia-free"
)

print(f"Configuration: {config.model_name}")
print(f"Max articles: {config.max_articles:,}")
print(f"Estimated training time: 8-12 hours")
"""
            },
            {
                "type": "code",
                "content": """
# Start training
import asyncio
trainer = WikipediaTrainer(config)
results = await trainer.run_complete_pipeline()
print(f"Training completed: {results}")

# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/models /content/drive/MyDrive/quark_wikipedia_model
"""
            }
        ]
    }


def save_budget_configs():
    """Save all budget configurations to files."""
    import os
    
    budget_dir = "configs/budget_training"
    os.makedirs(budget_dir, exist_ok=True)
    
    # Save each configuration
    for name, config in BUDGET_CONFIGS.items():
        deployment_config = generate_deployment_config(name)
        
        config_file = f"{budget_dir}/{name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        print(f"✅ Saved: {config_file}")
    
    # Save Colab notebook config
    colab_config = create_colab_notebook_config()
    colab_file = f"{budget_dir}/colab_free_training.json"
    with open(colab_file, 'w') as f:
        json.dump(colab_config, f, indent=2)
    
    print(f"✅ Saved: {colab_file}")
    
    # Create summary file
    summary = {
        "budget_tiers": {name: {
            "cost": config.estimated_total_cost,
            "time": config.training_time,
            "description": config.description
        } for name, config in BUDGET_CONFIGS.items()},
        "recommendations": {
            "testing": "free_tier or ultra_budget",
            "development": "low_budget or balanced_budget", 
            "production": "performance_budget",
            "cost_savings": "95-99% vs original configuration"
        }
    }
    
    summary_file = f"{budget_dir}/budget_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved: {summary_file}")
    print(f"\n📁 All budget configs saved to: {budget_dir}/")


if __name__ == "__main__":
    print("🧠 QUARK BUDGET WIKIPEDIA TRAINING CONFIGURATIONS")
    print("=" * 60)
    
    # Show cost comparison
    get_cost_comparison()
    
    print("\n📋 Available Budget Configurations:")
    for name, config in BUDGET_CONFIGS.items():
        print(f"\n🔹 {config.name}")
        print(f"   Cost: {config.estimated_total_cost}")
        print(f"   Time: {config.training_time}")
        print(f"   Instance: {config.num_nodes}x {config.instance_type}")
        print(f"   Description: {config.description}")
    
    # Save configurations
    print("\n💾 Saving budget configurations...")
    save_budget_configs()
    
    print("\n🚀 Quick Start Commands:")
    print("# Free training (Google Colab):")
    print("# → Upload colab_free_training.json to Google Colab")
    print()
    print("# Ultra budget ($8-15 total):")
    print("python scripts/quick_start_wikipedia_training.py --config configs/budget_training/ultra_budget_config.json")
    print()
    print("# Balanced budget ($25-50 total):")
    print("python scripts/quick_start_wikipedia_training.py --config configs/budget_training/balanced_budget_config.json")
