#!/usr/bin/env python3
"""Core Biological Simulator Implementation.

Provides the main BiologicalSimulator class for AlphaGenome integration.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import time
from pathlib import Path

from .simulation_types import (
    SimulationMode, BiologicalProcess, MorphogenGradient,
    DevelopmentalEvent, SimulationParameters, DevelopmentalStage
)


class BiologicalSimulator:
    """Main biological simulation engine for AlphaGenome integration."""
    
    def __init__(self, simulation_params: Optional[SimulationParameters] = None):
        """Initialize biological simulator.
        
        Args:
            simulation_params: Simulation configuration parameters
        """
        self.params = simulation_params or SimulationParameters("default_sim")
        self.current_time = 0.0
        self.current_stage = DevelopmentalStage.NEURAL_INDUCTION
        self.simulation_running = False
        
        # Initialize morphogen gradients
        self.morphogen_gradients = {}
        for gradient in self.params.morphogen_gradients:
            self.morphogen_gradients[gradient.morphogen_name] = gradient
        
        # Initialize developmental events
        self.developmental_events = self.params.developmental_events
        
        # Initialize biological state
        self.total_cells = 1000  # Starting cell count
        self.brain_regions = {
            "neural_plate": {"cells": 200, "differentiation": 0.1},
            "cortex": {"cells": 300, "differentiation": 0.0},
            "hippocampus": {"cells": 150, "differentiation": 0.0},
            "cerebellum": {"cells": 200, "differentiation": 0.0},
            "brainstem": {"cells": 150, "differentiation": 0.0}
        }
        
        # Gene regulatory networks
        self.gene_networks = self._initialize_grns()
        
        print(f"🧬 BiologicalSimulator initialized with {len(self.morphogen_gradients)} morphogens")
        print(f"   - Total cells: {self.total_cells}")
        print(f"   - Brain regions: {len(self.brain_regions)}")
        print(f"   - Developmental events: {len(self.developmental_events)}")
    
    def _initialize_grns(self) -> Dict[str, Dict]:
        """Initialize gene regulatory networks."""
        return {
            "neural_induction": {
                "SOX2": {"expression": 0.8, "targets": ["NESTIN", "PAX6"]},
                "PAX6": {"expression": 0.6, "targets": ["NEUROD1"]},
                "NESTIN": {"expression": 0.7, "targets": ["TUBB3"]}
            },
            "neuronal_differentiation": {
                "NEUROD1": {"expression": 0.5, "targets": ["TUBB3", "MAP2"]},
                "TUBB3": {"expression": 0.4, "targets": ["SNAP25"]},
                "MAP2": {"expression": 0.3, "targets": ["SYN1"]}
            }
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            "current_time": self.current_time,
            "current_stage": self.current_stage.value,
            "total_cells": self.total_cells,
            "simulation_running": self.simulation_running,
            "brain_regions": self.brain_regions.copy(),
            "morphogen_levels": {
                name: np.random.exponential(0.5) 
                for name in self.morphogen_gradients.keys()
            }
        }
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Execute one simulation step.
        
        Args:
            dt: Time step size (uses params default if None)
            
        Returns:
            Updated simulation state
        """
        if dt is None:
            dt = self.params.time_step
        
        self.current_time += dt
        
        # Update morphogen gradients
        self._update_morphogens(dt)
        
        # Process developmental events
        self._process_events(dt)
        
        # Update cell populations
        self._update_cell_populations(dt)
        
        # Update developmental stage
        self._update_developmental_stage()
        
        return self.get_current_state()
    
    def _update_morphogens(self, dt: float):
        """Update morphogen gradient concentrations."""
        for name, gradient in self.morphogen_gradients.items():
            # Simple exponential decay with production
            current_level = np.random.exponential(gradient.production_rate * dt)
            # Apply diffusion and decay (simplified)
            current_level *= np.exp(-gradient.decay_rate * dt)
    
    def _process_events(self, dt: float):
        """Process active developmental events."""
        for event in self.developmental_events:
            if (event.start_time <= self.current_time <= 
                event.start_time + event.duration):
                self._execute_event(event, dt)
    
    def _execute_event(self, event: DevelopmentalEvent, dt: float):
        """Execute a developmental event."""
        if event.event_type == BiologicalProcess.NEURAL_INDUCTION:
            # Increase neural plate differentiation
            if "neural_plate" in self.brain_regions:
                self.brain_regions["neural_plate"]["differentiation"] += 0.01 * dt
        
        elif event.event_type == BiologicalProcess.NEURONAL_MIGRATION:
            # Migrate cells from neural plate to other regions
            for region in event.affected_regions:
                if region in self.brain_regions and region != "neural_plate":
                    migration_rate = 5 * dt  # cells per time step
                    self.brain_regions[region]["cells"] += migration_rate
                    if "neural_plate" in self.brain_regions:
                        self.brain_regions["neural_plate"]["cells"] -= migration_rate * 0.5
        
        elif event.event_type == BiologicalProcess.SYNAPTOGENESIS:
            # Increase synapse formation in affected regions
            for region in event.affected_regions:
                if region in self.brain_regions:
                    self.brain_regions[region]["differentiation"] += 0.02 * dt
    
    def _update_cell_populations(self, dt: float):
        """Update cell population counts."""
        # Simple growth model
        growth_rate = 0.1 * dt
        for region_data in self.brain_regions.values():
            region_data["cells"] *= (1 + growth_rate)
        
        # Update total cell count
        self.total_cells = sum(
            region["cells"] for region in self.brain_regions.values()
        )
    
    def _update_developmental_stage(self):
        """Update current developmental stage based on time."""
        if self.current_time < 24:
            self.current_stage = DevelopmentalStage.NEURAL_INDUCTION
        elif self.current_time < 96:
            self.current_stage = DevelopmentalStage.NEURONAL_MIGRATION
        elif self.current_time < 144:
            self.current_stage = DevelopmentalStage.SYNAPTOGENESIS
        else:
            self.current_stage = DevelopmentalStage.MATURATION
    
    def run_simulation(self, duration: Optional[float] = None, steps: Optional[int] = None) -> Dict[str, Any]:
        """Run simulation for specified duration or steps.
        
        Args:
            duration: Simulation duration in hours (uses params total_time if None)
            steps: Number of simulation steps (alternative to duration)
            
        Returns:
            Final simulation state
        """
        if steps is not None:
            # Convert steps to duration (assume each step = 1 hour)
            duration = float(steps)
        elif duration is None:
            duration = self.params.total_time
        
        self.simulation_running = True
        start_time = self.current_time
        
        print(f"🚀 Starting biological simulation for {duration} hours")
        
        try:
            while (self.current_time - start_time) < duration:
                self.step()
                
                # Save periodically if requested
                if (self.current_time % self.params.save_frequency) < self.params.time_step:
                    self._save_checkpoint()
            
            print(f"✅ Biological simulation completed at t={self.current_time:.1f}h")
            
        except KeyboardInterrupt:
            print(f"⏸️  Simulation paused at t={self.current_time:.1f}h")
        
        finally:
            self.simulation_running = False
        
        # Return brain specification format for BrainSimulator integration
        final_state = self.get_current_state()
        brain_spec = self.get_brain_specification()
        
        # Ensure proper formatting for brain initialization
        cell_distribution = brain_spec["cell_type_distribution"]
        
        return {
            "final_state": {
                "cell_type_distribution": {
                    "neuron": cell_distribution.get("NEURON", 100),  # lowercase for compatibility
                    "astrocyte": cell_distribution.get("ASTROCYTE", 20),
                    "oligodendrocyte": cell_distribution.get("OLIGODENDROCYTE", 5),
                    "microglia": cell_distribution.get("MICROGLIA", 2)
                },
                "total_tissues": len(brain_spec["brain_regions"]),
                "tissue_types": list(brain_spec["brain_regions"].keys())
            },
            "simulation_state": final_state,
            "brain_specification": brain_spec
        }
    
    def _save_checkpoint(self):
        """Save simulation checkpoint (placeholder)."""
        # In a full implementation, this would save state to disk
        pass
    
    def get_brain_specification(self) -> Dict[str, Any]:
        """Generate brain specification for BrainSimulator integration."""
        # Calculate cell type distributions based on current state
        total_neurons = sum(
            region["cells"] * region["differentiation"]
            for region in self.brain_regions.values()
        )
        
        return {
            "cell_type_distribution": {
                "NEURON": max(10, int(total_neurons * 0.8)),
                "ASTROCYTE": max(2, int(total_neurons * 0.15)),
                "OLIGODENDROCYTE": max(1, int(total_neurons * 0.04)),
                "MICROGLIA": max(1, int(total_neurons * 0.01))
            },
            "brain_regions": self.brain_regions.copy(),
            "developmental_stage": self.current_stage.value,
            "simulation_time": self.current_time,
            "morphogen_status": list(self.morphogen_gradients.keys())
        }


def create_biological_simulator(
    dna_controller=None, 
    cell_constructor=None, 
    genome_analyzer=None, 
    simulation_params=None
) -> BiologicalSimulator:
    """Factory function to create BiologicalSimulator instance.
    
    Args:
        dna_controller: DNA controller instance (optional)
        cell_constructor: Cell constructor instance (optional)
        genome_analyzer: Genome analyzer instance (optional)
        simulation_params: Simulation parameters (optional)
        
    Returns:
        Configured BiologicalSimulator instance
    """
    if simulation_params is None:
        simulation_params = SimulationParameters("integrated_simulation")
    
    simulator = BiologicalSimulator(simulation_params)
    
    # Store references to other components if provided
    if dna_controller:
        simulator.dna_controller = dna_controller
    if cell_constructor:
        simulator.cell_constructor = cell_constructor
    if genome_analyzer:
        simulator.genome_analyzer = genome_analyzer
    
    return simulator
