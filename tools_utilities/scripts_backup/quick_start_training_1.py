#!/usr/bin/env python3
"""
🚀 Quick Start Training Script

This script demonstrates how to use all the training scripts in the unified training system.
It provides examples for training different types of models and shows how to configure
and run training sessions for various domains.
"""

import os, sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickStartTrainer:
    """Quick start trainer for demonstrating all training capabilities"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.training_scripts = {
            "brain_development": "brain_development/fetal_brain_trainer.py",
            "neural_architectures": "neural_architectures/childlike_learner_trainer.py",
            "cognitive_engines": "cognitive_engines/curiosity_engine_trainer.py",
            "multi_agent": "multi_agent/agent_coordinator_trainer.py"
        }
        
        # Default configurations for each domain
        self.default_configs = {
            "brain_development": {
                "epochs": 50,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "description": "Fetal brain development simulation training"
            },
            "neural_architectures": {
                "epochs": 30,
                "batch_size": 16,
                "learning_rate": 3e-5,
                "curiosity_weight": 0.1,
                "description": "Childlike learning neural architecture training"
            },
            "cognitive_engines": {
                "epochs": 40,
                "batch_size": 12,
                "learning_rate": 5e-5,
                "novelty_weight": 0.3,
                "uncertainty_weight": 0.2,
                "description": "Curiosity-driven cognitive engine training"
            },
            "multi_agent": {
                "epochs": 60,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "agent_count": 3,
                "coordination_weight": 0.2,
                "description": "Multi-agent coordination training"
            }
        }
    
    def list_available_domains(self) -> List[str]:
        """List all available training domains"""
        return list(self.training_scripts.keys())
    
    def get_domain_info(self, domain: str) -> Optional[Dict]:
        """Get information about a specific domain"""
        if domain not in self.training_scripts:
            return None
        
        script_path = self.base_path / "training_systems" / self.training_scripts[domain]
        
        return {
            "domain": domain,
            "script_path": str(script_path),
            "config": self.default_configs.get(domain, {}),
            "available": script_path.exists()
        }
    
    def run_training(self, domain: str, custom_args: Optional[Dict] = None) -> bool:
        """Run training for a specific domain"""
        if domain not in self.training_scripts:
            logger.error(f"Unknown domain: {domain}")
            return False
        
        script_path = self.base_path / "training_systems" / self.training_scripts[domain]
        
        if not script_path.exists():
            logger.error(f"Training script not found: {script_path}")
            return False
        
        # Build command arguments
        cmd = [sys.executable, str(script_path)]
        
        # Add custom arguments
        if custom_args:
            for key, value in custom_args.items():
                if key.startswith("--"):
                    cmd.extend([key, str(value)])
                else:
                    cmd.extend([f"--{key}", str(value)])
        
        # Add default arguments if not provided
        default_config = self.default_configs.get(domain, {})
        if custom_args is None:
            custom_args = {}
        
        for key, value in default_config.items():
            if key not in custom_args and key != "description":
                cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running training for domain: {domain}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the training script
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Training completed successfully!")
            logger.info(f"Output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def run_demo_training(self, domain: str = None) -> bool:
        """Run a quick demo training session"""
        if domain is None:
            # Run a quick demo for neural architectures (fastest)
            domain = "neural_architectures"
        
        logger.info(f"Running demo training for domain: {domain}")
        
        # Use minimal training parameters for demo
        demo_args = {
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": 1e-3
        }
        
        return self.run_training(domain, demo_args)
    
    def run_full_training_suite(self) -> Dict[str, bool]:
        """Run training for all available domains"""
        results = {}
        
        logger.info("Running full training suite for all domains...")
        
        for domain in self.training_scripts.keys():
            logger.info(f"Starting training for domain: {domain}")
            success = self.run_training(domain)
            results[domain] = success
            
            if success:
                logger.info(f"✅ Training completed successfully for {domain}")
            else:
                logger.error(f"❌ Training failed for {domain}")
        
        return results
    
    def create_training_config(self, domain: str, output_path: str = None) -> bool:
        """Create a configuration file for a specific domain"""
        if domain not in self.default_configs:
            logger.error(f"Unknown domain: {domain}")
            return False
        
        config = self.default_configs[domain]
        
        if output_path is None:
            output_path = f"{domain}_config.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration file created: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create configuration file: {e}")
            return False
    
    def show_training_summary(self) -> None:
        """Show a summary of all available training options"""
        print("\n" + "="*60)
        print("🧠 UNIFIED TRAINING SYSTEM - QUICK START GUIDE")
        print("="*60)
        
        print("\n📁 Available Training Domains:")
        for domain in self.training_scripts.keys():
            info = self.get_domain_info(domain)
            if info:
                print(f"\n  🎯 {domain.upper()}")
                print(f"     Description: {info['config'].get('description', 'No description available')}")
                print(f"     Script: {info['script_path']}")
                print(f"     Available: {'✅' if info['available'] else '❌'}")
                
                config = info['config']
                print(f"     Default Config:")
                for key, value in config.items():
                    if key != "description":
                        print(f"       {key}: {value}")
        
        print("\n🚀 Quick Start Commands:")
        print("  python quick_start_training.py --demo")
        print("  python quick_start_training.py --domain brain_development --epochs 10")
        print("  python quick_start_training.py --all")
        
        print("\n📋 Example Usage:")
        print("  # Run a quick demo")
        print("  python quick_start_training.py --demo")
        print("")
        print("  # Train brain development models")
        print("  python quick_start_training.py --domain brain_development --epochs 50")
        print("")
        print("  # Train childlike learning with custom curiosity")
        print("  python quick_start_training.py --domain neural_architectures --curiosity-weight 0.2")
        print("")
        print("  # Train multi-agent system with 5 agents")
        print("  python quick_start_training.py --domain multi_agent --agent-count 5")
        print("")
        print("  # Run all training domains")
        print("  python quick_start_training.py --all")
        
        print("\n" + "="*60)

def main():
    """Main entry point for quick start training"""
    parser = argparse.ArgumentParser(description="Quick Start Training for Unified Training System")
    parser.add_argument("--domain", type=str, choices=["brain_development", "neural_architectures", "cognitive_engines", "multi_agent"],
                       help="Training domain to execute")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo training session")
    parser.add_argument("--all", action="store_true", help="Run training for all available domains")
    parser.add_argument("--list", action="store_true", help="List all available training domains")
    parser.add_argument("--summary", action="store_true", help="Show training summary and usage guide")
    parser.add_argument("--config", type=str, help="Create configuration file for specified domain")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--curiosity-weight", type=float, help="Curiosity weight (for neural architectures)")
    parser.add_argument("--novelty-weight", type=float, help="Novelty weight (for cognitive engines)")
    parser.add_argument("--agent-count", type=int, help="Number of agents (for multi-agent)")
    parser.add_argument("--coordination-weight", type=float, help="Coordination weight (for multi-agent)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = QuickStartTrainer()
    
    # Show summary if requested
    if args.summary:
        trainer.show_training_summary()
        return
    
    # List domains if requested
    if args.list:
        domains = trainer.list_available_domains()
        print("Available training domains:")
        for domain in domains:
            info = trainer.get_domain_info(domain)
            if info:
                print(f"  - {domain}: {info['config'].get('description', 'No description')}")
        return
    
    # Create configuration if requested
    if args.config:
        success = trainer.create_training_config(args.config)
        if success:
            print(f"Configuration file created for {args.config}")
        else:
            print(f"Failed to create configuration for {args.config}")
        return
    
    # Build custom arguments
    custom_args = {}
    if args.epochs:
        custom_args["epochs"] = args.epochs
    if args.batch_size:
        custom_args["batch-size"] = args.batch_size
    if args.learning_rate:
        custom_args["learning-rate"] = args.learning_rate
    if args.curiosity_weight:
        custom_args["curiosity-weight"] = args.curiosity_weight
    if args.novelty_weight:
        custom_args["novelty-weight"] = args.novelty_weight
    if args.agent_count:
        custom_args["agent-count"] = args.agent_count
    if args.coordination_weight:
        custom_args["coordination-weight"] = args.coordination_weight
    
    # Run training based on arguments
    if args.demo:
        success = trainer.run_demo_training()
        if success:
            print("✅ Demo training completed successfully!")
        else:
            print("❌ Demo training failed!")
    
    elif args.all:
        results = trainer.run_full_training_suite()
        print("\nTraining Results:")
        for domain, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {domain}: {status}")
    
    elif args.domain:
        success = trainer.run_training(args.domain, custom_args)
        if success:
            print(f"✅ Training completed successfully for {args.domain}!")
        else:
            print(f"❌ Training failed for {args.domain}!")
    
    else:
        # Show help if no action specified
        trainer.show_training_summary()

if __name__ == "__main__":
    main()
