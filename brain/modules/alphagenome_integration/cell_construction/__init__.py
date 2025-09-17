#!/usr/bin/env python3
"""Cell Construction Module - Main interface for cell construction system.

Provides unified interface to cell construction components with preserved integrations.

Integration: Main cell construction interface for AlphaGenome biological workflows.
Rationale: Clean API abstraction maintaining all existing functionality.
"""

from .cell_types import CellType, DevelopmentalStage, CellularParameters, TissueParameters
from .constructor_core import CellConstructor

def create_cell_constructor(dna_controller=None):
    """Factory function to create a CellConstructor with specified components."""
    return CellConstructor(dna_controller=dna_controller)

# Export main interface for backward compatibility
__all__ = [
    'CellConstructor',
    'CellType',
    'DevelopmentalStage',
    'CellularParameters',
    'TissueParameters',
    'create_cell_constructor'
]
