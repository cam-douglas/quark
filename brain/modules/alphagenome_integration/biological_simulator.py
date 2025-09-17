#!/usr/bin/env python3
"""AlphaGenome Biological Simulator - Main Interface.

Provides the primary interface for biological simulation functionality
within the AlphaGenome integration system.
"""

# Import from the modular biological_simulation package
from .biological_simulation.simulation_types import (
    SimulationMode, BiologicalProcess, MorphogenGradient,
    DevelopmentalEvent, SimulationParameters, DevelopmentalStage
)

from .biological_simulation.simulator_core import (
    BiologicalSimulator, create_biological_simulator
)

# Re-export for backward compatibility
__all__ = [
    'BiologicalSimulator',
    'SimulationMode',
    'BiologicalProcess',
    'MorphogenGradient',
    'DevelopmentalEvent',
    'DevelopmentalStage',
    'SimulationParameters',
    'create_biological_simulator'
]
