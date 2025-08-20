#!/usr/bin/env python3
"""
Comprehensive Training Orchestrator
===================================

Orchestrates all training domains and simulation tools for comprehensive
brain simulation training and fine-tuning.
"""

import os, sys
import json
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, field
from datetime import datetime
import subprocess
import signal
import psutil

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import training components
from integrated_brain_simulation_trainer import IntegratedBrainSimulationTrainer, BrainSimulationConfig
from domain_specific_trainers import (
    NeuroscienceTrainer, CognitiveLearningTrainer, 
    PhysicsSimulationTrainer, MultiAgentTrainer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TRAINING-ORCHESTRATOR] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingOrchestrationConfig:
    """Configuration for comprehensive training orchestration"""
    
    # Training domains to run
    domains: List[str] = field(default_factory=lambda: [
        "neuroscience", "cognitive", "physics", "agent"
    ])
    
    # Model configurations
    base_models: Dict[str, str] = field(default_factory=lambda: {
        "neuroscience": "microsoft/DialoGPT-medium",
        "cognitive": "microsoft/DialoGPT-medium", 
        "physics": "microsoft/DialoGPT-medium",
        "agent": "microsoft/DialoGPT-medium"
    })
    
    # Training parameters
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_length: int = 2048
    
    # Simulation settings
    enable_simulation: bool = True
    simulation_time: float = 500.0
    time_step: float = 0.1
    
    # Integration settings
    enable_curiosity: bool = True
    enable_exploration: bool = True
    enable_synthesis: bool = True
    enable_physics: bool = True
    enable_childlike_learning: bool = True
    
    # Orchestration settings
    parallel_training: bool = False
    max_parallel_jobs: int = 2
    checkpoint_interval: int = 5
    evaluation_interval: int = 10
    
    # Resource management
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    gpu_memory_fraction: float = 0.8

class TrainingOrchestrator:
    """
    Comprehensive training orchestrator that coordinates all training domains
    and simulation tools.
    """
    
    def __init__(self, config: TrainingOrchestrationConfig):
        self.config = config
        self.training_jobs = {}
        self.training_results = {}
        self.simulation_results = {}
        self.overall_progress = {
            "start_time": datetime.now().isoformat(),
            "completed_domains": [],
            "failed_domains": [],
            "total_progress": 0.0
        }
        
        # Setup output directories
        self._setup_output_directories()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_output_directories(self):
        """Setup output directories for training results"""
        output_dirs = [
            "training_outputs",
            "simulation_outputs", 
            "checkpoints",
            "reports",
            "logs"
        ]
        
        for dir_name in output_dirs:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            
        logger.info("✅ Output directories setup complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def check_system_resources(self) -> bool:
        """Check if system has sufficient resources"""
        try:
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb < self.config.max_memory_gb:
                logger.warning(f"Insufficient memory: {memory_gb:.1f}GB < {self.config.max_memory_gb}GB")
                return False
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.max_cpu_percent:
                logger.warning(f"High CPU usage: {cpu_percent}% > {self.config.max_cpu_percent}%")
                return False
            
            # Check GPU if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < 2.0:  # Minimum 2GB GPU memory
                    logger.warning(f"Insufficient GPU memory: {gpu_memory:.1f}GB")
                    return False
            
            logger.info("✅ System resources sufficient")
            return True
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return False
    
    def run_brain_simulation(self) -> Dict[str, Any]:
        """Run comprehensive brain simulation"""
        if not self.config.enable_simulation:
            logger.info("Simulation disabled, skipping...")
            return {}
        
        logger.info("🧠 Running comprehensive brain simulation...")
        
        try:
            # Create simulation config
            sim_config = BrainSimulationConfig(
                simulation_time=self.config.simulation_time,
                time_step=self.config.time_step,
                enable_physics=self.config.enable_physics,
                enable_curiosity=self.config.enable_curiosity,
                enable_exploration=self.config.enable_exploration,
                enable_synthesis=self.config.enable_synthesis
            )
            
            # Create integrated trainer
            trainer = IntegratedBrainSimulationTrainer(sim_config)
            
            # Run simulation
            simulation_results = trainer.run_brain_development_simulation()
            
            # Save simulation results
            sim_output_path = Path("simulation_outputs/brain_simulation_results.json")
            with open(sim_output_path, 'w') as f:
                json.dump(simulation_results, f, indent=2)
            
            self.simulation_results = simulation_results
            logger.info("✅ Brain simulation complete")
            return simulation_results
            
        except Exception as e:
            logger.error(f"Brain simulation failed: {e}")
            return {}
    
    def run_domain_training(self, domain: str) -> bool:
        """Run training for a specific domain"""
        logger.info(f"🎯 Starting {domain} domain training...")
        
        try:
            # Create domain-specific config
            config = BrainSimulationConfig(
                base_model=self.config.base_models.get(domain, "microsoft/DialoGPT-medium"),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                max_length=self.config.max_length,
                enable_curiosity=self.config.enable_curiosity,
                enable_exploration=self.config.enable_exploration,
                enable_synthesis=self.config.enable_synthesis,
                enable_physics=self.config.enable_physics,
                enable_childlike_learning=self.config.enable_childlike_learning
            )
            
            # Create domain-specific trainer
            if domain == "neuroscience":
                trainer = NeuroscienceTrainer(config)
                trainer.run_neuroscience_training()
            elif domain == "cognitive":
                trainer = CognitiveLearningTrainer(config)
                trainer.run_cognitive_training()
            elif domain == "physics":
                trainer = PhysicsSimulationTrainer(config)
                trainer.run_physics_training()
            elif domain == "agent":
                trainer = MultiAgentTrainer(config)
                trainer.run_agent_training()
            else:
                raise ValueError(f"Unknown domain: {domain}")
            
            # Record success
            self.overall_progress["completed_domains"].append(domain)
            self.training_results[domain] = {
                "status": "completed",
                "completion_time": datetime.now().isoformat(),
                "model_path": f"./{domain}_model"
            }
            
            logger.info(f"✅ {domain} training complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ {domain} training failed: {e}")
            
            # Record failure
            self.overall_progress["failed_domains"].append(domain)
            self.training_results[domain] = {
                "status": "failed",
                "error": str(e),
                "failure_time": datetime.now().isoformat()
            }
            
            return False
    
    def run_parallel_training(self):
        """Run training for all domains in parallel"""
        logger.info("🚀 Starting parallel training for all domains...")
        
        # Check system resources
        if not self.check_system_resources():
            logger.warning("Insufficient system resources, falling back to sequential training")
            self.run_sequential_training()
            return
        
        # Create training threads
        threads = []
        for domain in self.config.domains:
            thread = threading.Thread(
                target=self.run_domain_training,
                args=(domain,),
                name=f"training_{domain}"
            )
            threads.append(thread)
        
        # Start threads with limited concurrency
        active_threads = []
        for i, thread in enumerate(threads):
            if len(active_threads) >= self.config.max_parallel_jobs:
                # Wait for a thread to complete
                for active_thread in active_threads[:]:
                    if not active_thread.is_alive():
                        active_threads.remove(active_thread)
            
            thread.start()
            active_threads.append(thread)
            logger.info(f"Started {domain} training thread")
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        logger.info("✅ Parallel training complete")
    
    def run_sequential_training(self):
        """Run training for all domains sequentially"""
        logger.info("🚀 Starting sequential training for all domains...")
        
        for domain in self.config.domains:
            success = self.run_domain_training(domain)
            
            # Update progress
            completed = len(self.overall_progress["completed_domains"])
            total = len(self.config.domains)
            self.overall_progress["total_progress"] = (completed / total) * 100
            
            logger.info(f"Progress: {completed}/{total} domains complete ({self.overall_progress['total_progress']:.1f}%)")
            
            # Checkpoint if needed
            if completed % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
        
        logger.info("✅ Sequential training complete")
    
    def run_comprehensive_training(self):
        """Run comprehensive training pipeline"""
        logger.info("🎯 Starting comprehensive training orchestration...")
        
        start_time = datetime.now()
        
        try:
            # 1. Run brain simulation
            simulation_results = self.run_brain_simulation()
            
            # 2. Run domain training
            if self.config.parallel_training:
                self.run_parallel_training()
            else:
                self.run_sequential_training()
            
            # 3. Generate comprehensive report
            self.generate_comprehensive_report(simulation_results)
            
            # 4. Create model ensemble
            self.create_model_ensemble()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"🎉 Comprehensive training complete in {duration}")
            
        except Exception as e:
            logger.error(f"Comprehensive training failed: {e}")
            raise
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            "overall_progress": self.overall_progress,
            "training_results": self.training_results,
            "simulation_results": self.simulation_results,
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = Path("checkpoints/training_checkpoint.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def generate_comprehensive_report(self, simulation_results: Dict[str, Any]):
        """Generate comprehensive training report"""
        logger.info("📊 Generating comprehensive training report...")
        
        report = {
            "training_summary": {
                "start_time": self.overall_progress["start_time"],
                "end_time": datetime.now().isoformat(),
                "total_domains": len(self.config.domains),
                "completed_domains": len(self.overall_progress["completed_domains"]),
                "failed_domains": len(self.overall_progress["failed_domains"]),
                "success_rate": len(self.overall_progress["completed_domains"]) / len(self.config.domains) * 100
            },
            "domain_results": self.training_results,
            "simulation_results": simulation_results,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            "config": self.config.__dict__
        }
        
        # Save report
        report_path = Path("reports/comprehensive_training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary
        summary = f"""
🧠 Comprehensive Training Report
================================

Training Summary:
- Total Domains: {report['training_summary']['total_domains']}
- Completed: {report['training_summary']['completed_domains']}
- Failed: {report['training_summary']['failed_domains']}
- Success Rate: {report['training_summary']['success_rate']:.1f}%

Completed Domains:
{chr(10).join(f"- {domain}" for domain in self.overall_progress['completed_domains'])}

Failed Domains:
{chr(10).join(f"- {domain}" for domain in self.overall_progress['failed_domains'])}

System Information:
- CPU Cores: {report['system_info']['cpu_count']}
- Memory: {report['system_info']['memory_gb']:.1f}GB
- GPU Available: {report['system_info']['gpu_available']}
- GPU Count: {report['system_info']['gpu_count']}

Report saved to: {report_path}
        """
        
        # Save summary
        summary_path = Path("reports/training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"✅ Comprehensive report generated: {report_path}")
        print(summary)
    
    def create_model_ensemble(self):
        """Create ensemble of trained models"""
        logger.info("🤖 Creating model ensemble...")
        
        ensemble_config = {
            "models": {},
            "ensemble_type": "weighted_average",
            "weights": {},
            "created": datetime.now().isoformat()
        }
        
        # Add completed models to ensemble
        for domain, result in self.training_results.items():
            if result["status"] == "completed":
                model_path = result["model_path"]
                if Path(model_path).exists():
                    ensemble_config["models"][domain] = {
                        "path": model_path,
                        "type": domain,
                        "weight": 1.0 / len([r for r in self.training_results.values() if r["status"] == "completed"])
                    }
        
        # Save ensemble configuration
        ensemble_path = Path("training_outputs/model_ensemble.json")
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        logger.info(f"✅ Model ensemble created: {ensemble_path}")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down training orchestrator...")
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Generate final report
        self.generate_comprehensive_report(self.simulation_results)
        
        logger.info("✅ Shutdown complete")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive training orchestrator")
    parser.add_argument("--domains", nargs="+", 
                       choices=["neuroscience", "cognitive", "physics", "agent"],
                       default=["neuroscience", "cognitive", "physics", "agent"],
                       help="Domains to train")
    parser.add_argument("--parallel", action="store_true",
                       help="Run training in parallel")
    parser.add_argument("--max-parallel", type=int, default=2,
                       help="Maximum parallel training jobs")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs per domain")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--no-simulation", action="store_true",
                       help="Disable brain simulation")
    
    args = parser.parse_args()
    
    # Configuration
    config = TrainingOrchestrationConfig(
        domains=args.domains,
        parallel_training=args.parallel,
        max_parallel_jobs=args.max_parallel,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        enable_simulation=not args.no_simulation
    )
    
    print("🧠 Comprehensive Training Orchestrator")
    print("=" * 60)
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Parallel: {args.parallel}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Simulation: {not args.no_simulation}")
    print("=" * 60)
    
    try:
        # Create orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Run comprehensive training
        orchestrator.run_comprehensive_training()
        
        print("🎉 Comprehensive training orchestration complete!")
        print("Check the reports/ directory for detailed results.")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        orchestrator.shutdown()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
