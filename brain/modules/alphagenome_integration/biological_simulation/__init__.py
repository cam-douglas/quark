#!/usr/bin/env python3
"""Biological Simulation Module - Main interface for biological simulation system.

Provides unified interface to biological simulation components with preserved integrations.

Integration: Main biological simulation interface for brain modules and AlphaGenome integration.
Rationale: Clean API abstraction maintaining all existing functionality.
"""

from .simulation_types import (
    SimulationMode, BiologicalProcess, MorphogenGradient,
    DevelopmentalEvent, SimulationParameters
)

# Import the main BiologicalSimulator class from the original file for now
# This maintains backward compatibility while we modularize
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from the new modular core
from .simulator_core import BiologicalSimulator

# Export main interface for backward compatibility
__all__ = [
    'BiologicalSimulator',
    'SimulationMode',
    'BiologicalProcess',
    'MorphogenGradient',
    'DevelopmentalEvent',
    'SimulationParameters'
]
