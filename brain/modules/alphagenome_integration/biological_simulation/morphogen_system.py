#!/usr/bin/env python3
"""Morphogen System Module - Morphogen gradient simulation and management.

Handles morphogen concentration gradients, diffusion, and spatial distribution.

Integration: Morphogen simulation for biological development workflows.
Rationale: Specialized morphogen gradient logic separate from main simulation.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .simulation_types import MorphogenGradient

class MorphogenSystem:
    """Manages morphogen gradients and their spatial/temporal evolution."""
    
    def __init__(self, spatial_dimensions: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)):
        self.spatial_dimensions = spatial_dimensions
        self.morphogen_gradients = {}
        self.gradient_history = []
    
    def setup_morphogen_gradients(self):
        """Set up initial morphogen gradients for neural development."""
        # SHH (Sonic Hedgehog) - ventral patterning
        self.create_morphogen_gradient(
            "SHH", 
            source_position=(500.0, 0.0, 500.0),  # Ventral midline
            initial_concentration=100.0,
            diffusion_rate=0.1,
            decay_rate=0.05
        )
        
        # BMP (Bone Morphogenetic Protein) - dorsal patterning
        self.create_morphogen_gradient(
            "BMP",
            source_position=(500.0, 1000.0, 500.0),  # Dorsal surface
            initial_concentration=80.0,
            diffusion_rate=0.08,
            decay_rate=0.04
        )
        
        # WNT - posterior patterning
        self.create_morphogen_gradient(
            "WNT",
            source_position=(0.0, 500.0, 500.0),  # Posterior end
            initial_concentration=90.0,
            diffusion_rate=0.12,
            decay_rate=0.06
        )
        
        # FGF (Fibroblast Growth Factor) - anterior patterning
        self.create_morphogen_gradient(
            "FGF",
            source_position=(1000.0, 500.0, 500.0),  # Anterior end
            initial_concentration=70.0,
            diffusion_rate=0.15,
            decay_rate=0.07
        )
    
    def create_morphogen_gradient(self, name: str, source_position: Tuple[float, float, float],
                                initial_concentration: float, diffusion_rate: float, 
                                decay_rate: float) -> MorphogenGradient:
        """Create a morphogen gradient with specified parameters."""
        gradient = MorphogenGradient(
            name=name,
            concentration=initial_concentration,
            position=source_position,
            diffusion_rate=diffusion_rate,
            decay_rate=decay_rate,
            source_strength=initial_concentration
        )
        
        self.morphogen_gradients[name] = gradient
        return gradient
    
    def update_morphogen_gradients(self, time_step: float):
        """Update all morphogen gradients for one time step."""
        for gradient_name, gradient in self.morphogen_gradients.items():
            # Simple diffusion and decay model
            gradient.concentration *= (1.0 - gradient.decay_rate * time_step)
            
            # Maintain minimum concentration from source
            if gradient.concentration < gradient.source_strength * 0.1:
                gradient.concentration = gradient.source_strength * 0.1
    
    def get_morphogen_concentrations_at_position(self, position: Tuple[float, float, float]) -> Dict[str, float]:
        """Get morphogen concentrations at a specific spatial position."""
        concentrations = {}
        
        for name, gradient in self.morphogen_gradients.items():
            # Calculate distance from gradient source
            distance = np.sqrt(
                (position[0] - gradient.position[0])**2 +
                (position[1] - gradient.position[1])**2 +
                (position[2] - gradient.position[2])**2
            )
            
            # Simple exponential decay with distance
            concentration = gradient.concentration * np.exp(-distance / 200.0)
            concentrations[name] = max(0.0, concentration)
        
        return concentrations
    
    def simulate_morphogen_diffusion(self) -> List[Dict[str, Any]]:
        """Simulate morphogen gradient diffusion for one step."""
        events = []
        
        for name, gradient in self.morphogen_gradients.items():
            # Simulate diffusion process
            diffusion_event = {
                "type": "morphogen_diffusion",
                "morphogen": name,
                "concentration": gradient.concentration,
                "position": gradient.position,
                "diffusion_rate": gradient.diffusion_rate
            }
            events.append(diffusion_event)
        
        return events
