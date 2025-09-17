#!/usr/bin/env python3
"""Neuroepithelial Cells Module - Lineage-tagged neural progenitor cells.

This module implements lineage-tagged neuroepithelial cells for neural tube
development, providing the cellular foundation for downstream proliferation.

Integration: Core cellular layer for neural development in brain architecture.
Rationale: Establishes neural progenitor populations with developmental lineage tracking.
"""

from .neuroepithelial_generator import NeuroepithelialGenerator
from .lineage_tracker import LineageTracker
from .cell_proliferation import CellProliferationManager
from .neural_progenitors import NeuralProgenitorPool

__version__ = "1.0.0"
__author__ = "Quark Brain Architecture"

__all__ = [
    'NeuroepithelialGenerator',
    'LineageTracker',
    'CellProliferationManager',
    'NeuralProgenitorPool'
]
