#!/usr/bin/env python3
"""Biological Simulator - Clean interface to modular biological simulation.

Integration: This module participates in biological workflows via BiologicalSimulator.
Rationale: Streamlined interface to modular biological simulation components.
"""

# Import from modular biological simulation system
from .biological_simulation import (
    BiologicalSimulator,
    SimulationMode,
    BiologicalProcess,
    MorphogenGradient,
    DevelopmentalEvent,
    SimulationParameters
)

def create_biological_simulator(dna_controller=None, cell_constructor=None,
                               genome_analyzer=None, simulation_params=None):
    """Factory function to create a BiologicalSimulator with specified components."""
    return BiologicalSimulator(
        dna_controller=dna_controller,
        cell_constructor=cell_constructor, 
        genome_analyzer=genome_analyzer,
        simulation_params=simulation_params
    )

# Re-export for backward compatibility
__all__ = [
    'BiologicalSimulator', 'SimulationMode', 'BiologicalProcess', 
    'MorphogenGradient', 'DevelopmentalEvent', 'SimulationParameters',
    'create_biological_simulator'
]
