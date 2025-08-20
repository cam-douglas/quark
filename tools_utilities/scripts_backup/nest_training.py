#!/usr/bin/env python3
"""
NEST Training Script for Brain Simulation Framework
==================================================

This script integrates NEST (Neural Simulation Tool) for spiking neural network
simulations with the brain simulation framework.

NEST Documentation: https://github.com/nest/nest-simulator
NEST Website: https://www.nest-simulator.org

Author: Brain Simulation Framework Team
Date: 2024
License: Apache-2.0
"""

import os, sys
import asyncio
import logging
import numpy as np
import json
from datetime import datetime
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from typing import Dict, List, Any
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class NESTConfig:
    """Configuration for NEST training."""
    simulation_time: float = 1000.0  # ms
    num_neurons: int = 1000
    connectivity_density: float = 0.1
    output_dir: str = "nest_outputs"
    enable_brain_simulation: bool = True


class NESTTrainer:
    """NEST trainer for spiking neural network simulations."""
    
    def __init__(self, config: NESTConfig):
        self.config = config
        self.nest_available = False
        self.simulation_results = {}
        
        self._check_nest_installation()
    
    def _check_nest_installation(self):
        """Check NEST installation."""
        try:
            import nest
            self.nest_available = True
            console.print("✅ NEST is available")
        except ImportError:
            console.print("[bold red]NEST not available. Please install NEST.[/bold red]")
            console.print("See: https://www.nest-simulator.org/installation/")
            self.nest_available = False
    
    def create_network_model(self, model_name: str):
        """Create a NEST network model."""
        if not self.nest_available:
            return None
        
        try:
            import nest
            
            console.print(f"[bold green]Creating NEST network: {model_name}[/bold green]")
            
            # Reset NEST kernel
            nest.ResetKernel()
            
            # Set simulation parameters
            nest.SetKernelStatus({
                'resolution': 0.1,
                'overwrite_files': True,
                'print_time': True
            })
            
            # Create neurons
            neuron_params = {
                't_ref': 2.0,
                'V_reset': -70.0,
                'V_th': -55.0,
                'tau_m': 20.0,
                'C_m': 250.0,
                'E_L': -70.0
            }
            
            neurons = nest.Create('iaf_cond_alpha', self.config.num_neurons, params=neuron_params)
            
            # Create synapses and connect
            syn_params = {'weight': 1.0, 'delay': 1.0}
            nest.Connect(neurons, neurons, 
                        conn_spec={'rule': 'pairwise_bernoulli', 'p': self.config.connectivity_density},
                        syn_spec=syn_params)
            
            # Set up recording
            spike_recorder = nest.Create('spike_recorder')
            nest.Connect(neurons[:10], spike_recorder)
            
            return {
                "neurons": neurons,
                "spike_recorder": spike_recorder,
                "network_size": self.config.num_neurons
            }
            
        except Exception as e:
            logger.error(f"Error creating NEST model: {e}")
            return None
    
    async def run_simulation(self, model_name: str, model_data: Dict[str, Any]):
        """Run a NEST simulation."""
        if not self.nest_available:
            return None
        
        try:
            import nest
            
            console.print(f"[bold green]Running NEST simulation: {model_name}[/bold green]")
            
            start_time = datetime.now()
            
            # Run simulation
            nest.Simulate(self.config.simulation_time)
            
            end_time = datetime.now()
            simulation_time = (end_time - start_time).total_seconds()
            
            # Collect results
            spike_data = model_data["spike_recorder"].get('events')
            
            results = {
                "model_name": model_name,
                "simulation_time": simulation_time,
                "spike_data": spike_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate metrics
            if spike_data and len(spike_data["times"]) > 0:
                results["metrics"] = {
                    "total_spikes": len(spike_data["times"]),
                    "spike_rate": len(spike_data["times"]) / (self.config.simulation_time / 1000.0),
                    "unique_neurons": len(np.unique(spike_data["senders"]))
                }
            
            self.simulation_results[model_name] = results
            
            console.print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error running NEST simulation: {e}")
            return None
    
    async def train_network(self, model_name: str, epochs: int = 10):
        """Train a neural network using NEST simulations."""
        console.print(f"[bold green]Starting NEST training: {model_name}[/bold green]")
        
        # Create model
        model_data = self.create_network_model(model_name)
        if not model_data:
            return None
        
        training_results = []
        
        with Progress() as progress:
            task = progress.add_task("Training...", total=epochs)
            
            for epoch in range(epochs):
                try:
                    # Run simulation
                    results = await self.run_simulation(model_name, model_data)
                    
                    if results:
                        training_results.append({
                            "epoch": epoch,
                            "results": results,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error in epoch {epoch}: {e}")
                    progress.update(task, advance=1)
        
        # Save results
        self._save_results(training_results)
        
        console.print("[bold green]NEST training completed![/bold green]")
        return training_results
    
    def _save_results(self, training_results: List[Dict[str, Any]]):
        """Save training results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(self.config.output_dir, "nest_training_results.json")
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        console.print(f"✅ Results saved to {self.config.output_dir}")


async def main():
    """Main NEST training function."""
    console.print(Panel.fit(
        "[bold blue]NEST Training[/bold blue]\n"
        "Spiking neural network training with NEST",
        border_style="blue"
    ))
    
    # Configuration
    config = NESTConfig()
    
    # Initialize trainer
    trainer = NESTTrainer(config)
    
    if not trainer.nest_available:
        return
    
    # Train network
    results = await trainer.train_network("simple_network", epochs=5)
    
    if results:
        console.print(f"✅ Training completed with {len(results)} epochs")


if __name__ == "__main__":
    asyncio.run(main())
