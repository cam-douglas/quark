#!/usr/bin/env python3
"""
Small-Mind Simulation Runner

Comprehensive simulation management for brain development, neural networks,
and physics-based simulations using MuJoCo, NEST, and other frameworks.
"""

import os, sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import argparse

# Add the smallmind package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    import numpy as np
    import pandas as pd
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - some features will be limited")

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    logging.warning("PyBullet not available - physics simulation will be limited")

logger = logging.getLogger(__name__)

class BrainDevelopmentSimulation:
    """Brain development simulation using various frameworks"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        self.simulation_data = {}
        self.current_step = 0
        self.neural_network = None  # Initialize neural_network attribute
        
    def _default_config(self) -> Dict:
        """Default simulation configuration"""
        return {
            "simulation_type": "brain_development",
            "duration_steps": 1000,
            "time_step": 0.01,
            "output_frequency": 10,
            "physics_engine": "pybullet",
            "neural_network": True,
            "visualization": True,
            "save_results": True,
            "output_directory": "simulation_results"
        }
    
    def setup_logging(self):
        """Setup logging for the simulation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_simulation(self) -> bool:
        """Initialize the simulation environment"""
        self.logger.info("🧠 Initializing Brain Development Simulation")
        
        # Create output directory
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(exist_ok=True)
        
        # Initialize physics engine if available
        if self.config["physics_engine"] == "pybullet" and PYBULLET_AVAILABLE:
            self._init_pybullet()
        
        # Initialize neural network if enabled
        if self.config["neural_network"] and TORCH_AVAILABLE:
            self._init_neural_network()
        
        # Initialize visualization if enabled
        if self.config["visualization"]:
            self._init_visualization()
        
        self.logger.info("✅ Simulation initialized successfully")
        return True
    
    def _init_pybullet(self):
        """Initialize PyBullet physics engine"""
        if not PYBULLET_AVAILABLE:
            self.logger.warning("PyBullet not available - using mock physics engine")
            self.physics_client = None
            return
            
        try:
            self.logger.info("⚡ Initializing PyBullet physics engine")
            
            # Connect to PyBullet
            physics_client = p.connect(p.DIRECT)  # or p.GUI for visualization
            
            # Set physics parameters
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.config["time_step"])
            
            # Create ground plane
            ground_id = p.createCollisionShape(p.GEOM_PLANE)
            p.createMultiBody(0, ground_id)
            
            self.physics_client = physics_client
            self.logger.info("✅ PyBullet initialized")
        except Exception as e:
            self.logger.warning(f"PyBullet initialization failed: {e} - using mock physics")
            self.physics_client = None
    
    def _init_neural_network(self):
        """Initialize neural network for brain simulation"""
        self.logger.info("🧠 Initializing neural network")
        
        # Create a simple neural network for demonstration
        if TORCH_AVAILABLE:
            self.neural_network = self._create_brain_network()
            self.logger.info("✅ Neural network initialized")
        else:
            self.logger.warning("PyTorch not available - neural network disabled")
            self.neural_network = None
    
    def _create_brain_network(self):
        """Create a brain development neural network"""
        if not TORCH_AVAILABLE:
            return None
        
        # Simple feedforward network for brain development simulation
        network = torch.nn.Sequential(
            torch.nn.Linear(10, 64),  # Input: gene expression levels
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)   # Output: brain region development scores
        )
        
        return network
    
    def _init_visualization(self):
        """Initialize visualization components"""
        self.logger.info("🎨 Initializing visualization")
        
        # Initialize visualization data structures
        self.visualization_data = {
            "brain_regions": [],
            "development_scores": [],
            "time_steps": []
        }
        
        self.logger.info("✅ Visualization initialized")
    
    def run_simulation_step(self) -> Dict[str, Any]:
        """Run a single simulation step"""
        step_data = {
            "step": self.current_step,
            "time": self.current_step * self.config["time_step"],
            "physics": {},
            "neural": {},
            "development": {}
        }
        
        # Physics simulation step
        if hasattr(self, 'physics_client'):
            step_data["physics"] = self._physics_step()
        
        # Neural network step
        if self.neural_network is not None:
            step_data["neural"] = self._neural_step()
        
        # Brain development step
        step_data["development"] = self._development_step()
        
        # Store step data
        self.simulation_data[self.current_step] = step_data
        
        # Update visualization
        if self.config["visualization"]:
            self._update_visualization(step_data)
        
        self.current_step += 1
        return step_data
    
    def _physics_step(self) -> Dict[str, Any]:
        """Run physics simulation step"""
        if not hasattr(self, 'physics_client'):
            return {}
        
        # Step the physics simulation
        p.stepSimulation()
        
        # Get physics data
        physics_data = {
            "gravity": p.getGravity(),
            "time": p.getPhysicsEngineParameters()["physicsDeltaTime"]
        }
        
        return physics_data
    
    def _neural_step(self) -> Dict[str, Any]:
        """Run neural network step"""
        if self.neural_network is None:
            return {}
        
        # Generate synthetic input for this step
        input_data = torch.randn(1, 10)  # 10 gene expression levels
        
        # Forward pass
        with torch.no_grad():
            output = self.neural_network(input_data)
        
        neural_data = {
            "input": input_data.numpy().tolist(),
            "output": output.numpy().tolist(),
            "network_state": "active"
        }
        
        return neural_data
    
    def _development_step(self) -> Dict[str, Any]:
        """Run brain development step"""
        # Simulate brain development based on current step
        development_progress = self.current_step / self.config["duration_steps"]
        
        # Simulate different brain regions developing at different rates
        brain_regions = {
            "cerebral_cortex": min(1.0, development_progress * 1.2),
            "hippocampus": min(1.0, development_progress * 0.8),
            "cerebellum": min(1.0, development_progress * 1.5),
            "brainstem": min(1.0, development_progress * 0.6),
            "thalamus": min(1.0, development_progress * 1.0)
        }
        
        development_data = {
            "progress": development_progress,
            "brain_regions": brain_regions,
            "overall_development": np.mean(list(brain_regions.values()))
        }
        
        return development_data
    
    def _update_visualization(self, step_data: Dict[str, Any]):
        """Update visualization data"""
        if not self.config["visualization"]:
            return
        
        # Store data for visualization
        self.visualization_data["time_steps"].append(step_data["time"])
        
        if "development" in step_data:
            dev_data = step_data["development"]
            if "brain_regions" in dev_data:
                self.visualization_data["brain_regions"].append(dev_data["brain_regions"])
                self.visualization_data["development_scores"].append(dev_data["overall_development"])
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation"""
        self.logger.info(f"🚀 Starting simulation for {self.config['duration_steps']} steps")
        
        start_time = time.time()
        
        # Initialize simulation
        if not self.initialize_simulation():
            self.logger.error("Failed to initialize simulation")
            return {}
        
        # Run simulation steps
        for step in range(self.config["duration_steps"]):
            step_data = self.run_simulation_step()
            
            # Log progress
            if step % self.config["output_frequency"] == 0:
                self.logger.info(f"Step {step}/{self.config['duration_steps']} - "
                               f"Development: {step_data['development'].get('overall_development', 0):.3f}")
        
        # Finalize simulation
        simulation_summary = self._finalize_simulation()
        
        end_time = time.time()
        simulation_summary["execution_time"] = end_time - start_time
        
        self.logger.info(f"✅ Simulation completed in {simulation_summary['execution_time']:.2f} seconds")
        
        return simulation_summary
    
    def _finalize_simulation(self) -> Dict[str, Any]:
        """Finalize the simulation and prepare results"""
        self.logger.info("🏁 Finalizing simulation")
        
        # Calculate final statistics
        final_development = self.simulation_data[self.config["duration_steps"] - 1]["development"]
        
        summary = {
            "total_steps": self.config["duration_steps"],
            "final_development": final_development,
            "simulation_data": self.simulation_data,
            "visualization_data": self.visualization_data if self.config["visualization"] else {}
        }
        
        # Save results if enabled
        if self.config["save_results"]:
            self._save_simulation_results(summary)
        
        return summary
    
    def _save_simulation_results(self, results: Dict[str, Any]):
        """Save simulation results to files"""
        output_dir = Path(self.config["output_directory"])
        
        # Save main results
        results_file = output_dir / "simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save visualization data separately
        if "visualization_data" in results:
            viz_file = output_dir / "visualization_data.json"
            with open(viz_file, 'w') as f:
                json.dump(results["visualization_data"], f, indent=2, default=str)
        
        # Save CSV data for analysis
        self._save_csv_results(results, output_dir)
        
        self.logger.info(f"💾 Results saved to {output_dir}")
    
    def _save_csv_results(self, results: Dict[str, Any], output_dir: Path):
        """Save results in CSV format for analysis"""
        if "visualization_data" not in results:
            return
        
        viz_data = results["visualization_data"]
        
        # Create DataFrame for brain development
        if "brain_regions" in viz_data and "time_steps" in viz_data:
            df_data = []
            for i, (time_step, brain_regions) in enumerate(zip(viz_data["time_steps"], viz_data["brain_regions"])):
                row = {"time_step": time_step}
                row.update(brain_regions)
                if i < len(viz_data["development_scores"]):
                    row["overall_development"] = viz_data["development_scores"][i]
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = output_dir / "brain_development.csv"
            df.to_csv(csv_file, index=False)
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        return {
            "current_step": self.current_step,
            "total_steps": self.config["duration_steps"],
            "progress": self.current_step / self.config["duration_steps"] if self.config["duration_steps"] > 0 else 0,
            "config": self.config,
            "data_points": len(self.simulation_data)
        }

def create_simulation_config(simulation_type: str = "brain_development",
                           duration_steps: int = 1000,
                           physics_engine: str = "pybullet") -> Dict[str, Any]:
    """Create a simulation configuration"""
    return {
        "simulation_type": simulation_type,
        "duration_steps": duration_steps,
        "time_step": 0.01,
        "output_frequency": 10,
        "physics_engine": physics_engine,
        "neural_network": True,
        "visualization": True,
        "save_results": True,
        "output_directory": f"simulation_results_{simulation_type}"
    }

def main():
    """Main CLI interface for the simulation runner"""
    parser = argparse.ArgumentParser(description="Small-Mind Simulation Runner")
    parser.add_argument("--type", type=str, default="brain_development", 
                       help="Simulation type")
    parser.add_argument("--steps", type=int, default=1000, 
                       help="Number of simulation steps")
    parser.add_argument("--physics", type=str, default="pybullet", 
                       help="Physics engine to use")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create config from command line arguments
        config = create_simulation_config(
            simulation_type=args.type,
            duration_steps=args.steps,
            physics_engine=args.physics
        )
        
        if args.output:
            config["output_directory"] = args.output
    
    # Create and run simulation
    simulation = BrainDevelopmentSimulation(config)
    results = simulation.run_simulation()
    
    # Display summary
    if results:
        print("\n🚀 Simulation Summary:")
        print(f"Total Steps: {results['total_steps']}")
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        print(f"Final Development: {results['final_development']['overall_development']:.3f}")
        print(f"Results saved to: {config['output_directory']}")

if __name__ == "__main__":
    main()
