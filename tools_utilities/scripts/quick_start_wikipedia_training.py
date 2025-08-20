#!/usr/bin/env python3
"""
Quick Start Wikipedia Training
==============================

One-click setup and launch for Wikipedia training with cloud deployment.
Handles all setup, configuration, and monitoring automatically.

Purpose: Simplified interface for starting Wikipedia training
Inputs: Basic configuration parameters
Outputs: Running training infrastructure and monitoring links
Seeds: Random seeds for reproducible training setup
Dependencies: All training pipeline dependencies
"""

import os, sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_systems.training_pipelines.wikipedia_cloud_training import (
    WikipediaTrainer, WikipediaTrainingConfig
)
from deployment.cloud_computing.scripts.deploy_wikipedia_training import (
    WikipediaTrainingDeployer, DeploymentConfig
)
from brain_modules.conscious_agent.integrations.wikipedia_consciousness_integration import (
    WikipediaConsciousnessAgent, WikipediaConsciousnessConfig
)
from expert_domains.data_engineering.knowledge_base import KnowledgeBase


async def quick_setup_wikipedia_training(
    cloud_provider: str = "aws",
    region: str = "us-west-2",
    num_nodes: int = 4,
    max_articles: Optional[int] = None,
    enable_consciousness_integration: bool = True,
    dry_run: bool = False
) -> Dict[str, str]:
    """Quick setup for Wikipedia training with all components."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting quick Wikipedia training setup...")
    
    # Step 1: Create training configuration
    training_config = WikipediaTrainingConfig(
        model_name="microsoft/DialoGPT-medium",
        cloud_provider=cloud_provider,
        cloud_region=region,
        num_nodes=num_nodes,
        max_articles=max_articles,
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-5,
        wikipedia_dump_date="20240101",
        output_dir="./models/wikipedia_trained",
        wandb_project="quark-wikipedia-training"
    )
    
    logger.info(f"Training configuration: {training_config.model_name} on {cloud_provider}")
    
    # Step 2: Create deployment configuration
    deployment_config = DeploymentConfig(
        cloud_provider=cloud_provider,
        region=region,
        cluster_name="quark-wikipedia-training",
        node_count=num_nodes,
        instance_type="p3.8xlarge",
        container_image="quark/wikipedia-training:latest",
        storage_bucket="quark-wikipedia-training",
        enable_monitoring=True
    )
    
    # Step 3: Save configurations
    config_dir = Path("./configs/wikipedia_training")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    training_config_path = config_dir / "training_config.json"
    with open(training_config_path, 'w') as f:
        json.dump(training_config.__dict__, f, indent=2)
    
    deployment_config_path = config_dir / "deployment_config.json"
    with open(deployment_config_path, 'w') as f:
        json.dump(deployment_config.__dict__, f, indent=2)
    
    logger.info(f"Configurations saved to {config_dir}")
    
    if dry_run:
        logger.info("Dry run mode - configurations created but no deployment started")
        return {
            'status': 'dry_run_complete',
            'training_config': str(training_config_path),
            'deployment_config': str(deployment_config_path)
        }
    
    # Step 4: Deploy infrastructure
    logger.info("Deploying cloud infrastructure...")
    deployer = WikipediaTrainingDeployer(deployment_config)
    
    try:
        deployment_info = deployer.deploy_complete_infrastructure()
        logger.info("Infrastructure deployment completed successfully")
        
        # Save deployment info
        deployment_info_path = config_dir / "deployment_info.json"
        with open(deployment_info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Step 5: Initialize trainer
        logger.info("Initializing Wikipedia trainer...")
        trainer = WikipediaTrainer(training_config)
        
        # Step 6: Start training (async)
        logger.info("Starting Wikipedia training...")
        training_task = asyncio.create_task(trainer.run_complete_pipeline())
        
        # Step 7: Setup consciousness integration if enabled
        consciousness_info = {}
        if enable_consciousness_integration:
            logger.info("Setting up consciousness integration...")
            
            consciousness_config = WikipediaConsciousnessConfig(
                wikipedia_model_path=training_config.output_dir,
                max_context_length=2048,
                fine_tune_on_consciousness=True
            )
            
            consciousness_agent = WikipediaConsciousnessAgent(consciousness_config)
            
            consciousness_info = {
                'integration_enabled': True,
                'model_path': consciousness_config.wikipedia_model_path,
                'config': consciousness_config.__dict__
            }
        
        # Step 8: Setup monitoring and return info
        setup_info = {
            'status': 'training_started',
            'deployment_info': deployment_info,
            'training_config': training_config.__dict__,
            'consciousness_integration': consciousness_info,
            'monitoring_urls': {
                'wandb': f"https://wandb.ai/{training_config.wandb_project}",
                'kubernetes_dashboard': deployment_info.get('cluster_name', 'N/A')
            },
            'setup_time': datetime.now().isoformat(),
            'config_directory': str(config_dir)
        }
        
        # Wait for initial training setup to complete
        await asyncio.sleep(60)  # Give time for pods to start
        
        logger.info("Wikipedia training setup completed successfully!")
        return setup_info
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'configs_saved': str(config_dir)
        }


def check_prerequisites() -> Dict[str, bool]:
    """Check if all prerequisites are installed and configured."""
    
    prerequisites = {
        'python': False,
        'torch': False,
        'transformers': False,
        'docker': False,
        'kubectl': False,
        'cloud_cli': False
    }
    
    # Check Python
    try:
        import sys
        prerequisites['python'] = sys.version_info >= (3, 8)
    except:
        pass
    
    # Check PyTorch
    try:
        import torch
        prerequisites['torch'] = True
    except ImportError:
        pass
    
    # Check Transformers
    try:
        import transformers
        prerequisites['transformers'] = True
    except ImportError:
        pass
    
    # Check Docker
    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], capture_output=True)
        prerequisites['docker'] = result.returncode == 0
    except:
        pass
    
    # Check kubectl
    try:
        import subprocess
        result = subprocess.run(['kubectl', 'version', '--client'], capture_output=True)
        prerequisites['kubectl'] = result.returncode == 0
    except:
        pass
    
    # Check cloud CLI (AWS for now)
    try:
        import subprocess
        result = subprocess.run(['aws', '--version'], capture_output=True)
        prerequisites['cloud_cli'] = result.returncode == 0
    except:
        pass
    
    return prerequisites


def install_missing_dependencies():
    """Install missing dependencies automatically."""
    
    logger = logging.getLogger(__name__)
    
    # Install Python packages
    python_packages = [
        'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
        'transformers[torch]',
        'datasets',
        'accelerate',
        'wandb',
        'boto3',
        'kubernetes',
        'docker'
    ]
    
    for package in python_packages:
        try:
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + package.split(), check=True)
            logger.info(f"Installed: {package}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {package}: {e}")


def print_setup_summary(setup_info: Dict[str, str]):
    """Print a beautiful summary of the setup."""
    
    print("\n" + "="*80)
    print("🧠 QUARK WIKIPEDIA TRAINING SETUP COMPLETE 🧠")
    print("="*80)
    
    print(f"\n📊 Status: {setup_info['status'].upper()}")
    print(f"🕒 Setup Time: {setup_info.get('setup_time', 'N/A')}")
    print(f"📁 Config Directory: {setup_info.get('config_directory', 'N/A')}")
    
    if setup_info.get('deployment_info'):
        deployment = setup_info['deployment_info']
        print(f"\n☁️  Cloud Deployment:")
        print(f"   Provider: {deployment.get('provider', 'N/A')}")
        print(f"   Region: {deployment.get('region', 'N/A')}")
        print(f"   Cluster: {deployment.get('cluster_name', 'N/A')}")
        print(f"   Storage: {deployment.get('bucket_name', 'N/A')}")
    
    if setup_info.get('monitoring_urls'):
        monitoring = setup_info['monitoring_urls']
        print(f"\n📈 Monitoring:")
        print(f"   W&B Project: {monitoring.get('wandb', 'N/A')}")
        print(f"   K8s Dashboard: {monitoring.get('kubernetes_dashboard', 'N/A')}")
    
    if setup_info.get('consciousness_integration'):
        consciousness = setup_info['consciousness_integration']
        print(f"\n🧠 Consciousness Integration:")
        print(f"   Enabled: {consciousness.get('integration_enabled', False)}")
        print(f"   Model Path: {consciousness.get('model_path', 'N/A')}")
    
    print("\n🚀 Next Steps:")
    print("   1. Monitor training progress at the W&B URL above")
    print("   2. Check Kubernetes pods: kubectl get pods -n quark-training")
    print("   3. View logs: kubectl logs -n quark-training -l app=wikipedia-training")
    print("   4. Test consciousness integration once training completes")
    
    print("\n" + "="*80)


async def main():
    """Main entry point for quick start script."""
    
    parser = argparse.ArgumentParser(description="Quick start Wikipedia training with cloud deployment")
    
    parser.add_argument("--cloud-provider", type=str, default="aws", 
                       choices=["aws", "gcp", "azure"], help="Cloud provider")
    parser.add_argument("--region", type=str, default="us-west-2", help="Cloud region")
    parser.add_argument("--num-nodes", type=int, default=4, help="Number of training nodes")
    parser.add_argument("--max-articles", type=int, help="Maximum articles to train on (for testing)")
    parser.add_argument("--consciousness-integration", action="store_true", default=True,
                       help="Enable consciousness integration")
    parser.add_argument("--dry-run", action="store_true", help="Setup configs but don't deploy")
    parser.add_argument("--check-prereqs", action="store_true", help="Check prerequisites only")
    parser.add_argument("--install-deps", action="store_true", help="Install missing dependencies")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('wikipedia_training_setup.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Check prerequisites if requested
    if args.check_prereqs:
        prereqs = check_prerequisites()
        print("\n📋 Prerequisites Check:")
        print("-" * 30)
        for name, status in prereqs.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {name}: {'OK' if status else 'Missing'}")
        
        if not all(prereqs.values()):
            print(f"\n💡 Install missing dependencies with: python {sys.argv[0]} --install-deps")
        
        return
    
    # Install dependencies if requested
    if args.install_deps:
        print("📦 Installing missing dependencies...")
        install_missing_dependencies()
        print("✅ Dependency installation completed")
        return
    
    # Run quick setup
    try:
        setup_info = await quick_setup_wikipedia_training(
            cloud_provider=args.cloud_provider,
            region=args.region,
            num_nodes=args.num_nodes,
            max_articles=args.max_articles,
            enable_consciousness_integration=args.consciousness_integration,
            dry_run=args.dry_run
        )
        
        # Print beautiful summary
        print_setup_summary(setup_info)
        
        # Save setup info
        setup_info_path = Path("./configs/wikipedia_training/setup_info.json")
        setup_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(setup_info_path, 'w') as f:
            json.dump(setup_info, f, indent=2)
        
        logger.info(f"Setup information saved to: {setup_info_path}")
        
        if setup_info['status'] == 'training_started':
            print(f"\n🎉 Wikipedia training is now running!")
            print(f"📝 Full setup info saved to: {setup_info_path}")
        
    except Exception as e:
        logger.error(f"Quick start failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("📝 Check wikipedia_training_setup.log for details")


if __name__ == "__main__":
    asyncio.run(main())
