#!/usr/bin/env python3
"""Morphogen Solver Module - Foundation Layer Implementation.

This module implements the biologically-accurate morphogen gradient solver 
that serves as the foundation for Quark's neural development system during 
embryonic development (weeks 3-8).

Key Components:
- SHH (Sonic Hedgehog): Dorsal-ventral neural tube patterning
- BMP (Bone Morphogenetic Protein): Dorsal neural specification
- WNT (Wingless-related): Anterior-posterior patterning
- FGF (Fibroblast Growth Factor): Neural induction and progenitor maintenance

Integration: Core foundation for Stage 1 Embryonic Development
Rationale: Establishes morphogen gradients for neural tube patterning
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from .morphogen_solver import MorphogenSolver
from .spatial_grid import SpatialGrid
from .biological_parameters import BiologicalParameters
from .cell_fate_specifier import CellFateSpecifier
from .shh_gradient_system import SHHGradientSystem
from .ra_gradient_solver import RAGradientSolver
from .fgf8_gradient_solver import FGF8GradientSolver

__all__ = [
    "MorphogenSolver",
    "SpatialGrid", 
    "BiologicalParameters",
    "CellFateSpecifier",
    "SHHGradientSystem"
]

__all__ += [
    "RAGradientSolver",
    "FGF8GradientSolver",
]

__version__ = "1.0.0"
__status__ = "Stage 1 Embryonic - Foundation Layer Implementation"
