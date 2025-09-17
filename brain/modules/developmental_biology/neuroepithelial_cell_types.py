#!/usr/bin/env python3
"""Neuroepithelial Cell Type Definitions.

Core type definitions and molecular marker specifications for neuroepithelial
cells including cell states, competency windows, and marker expression patterns.

Integration: Type definitions for developmental biology system
Rationale: Centralized cell type definitions with biological accuracy
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

class NeuroepithelialCellType(Enum):
    """Types of neuroepithelial cells based on competency and position."""
    EARLY_MULTIPOTENT = "early_multipotent"         # Broad competency (E8.5-E9.5)
    LATE_MULTIPOTENT = "late_multipotent"           # Restricted competency (E9.5-E11.5)
    COMMITTED_PROGENITOR = "committed_progenitor"   # Lineage committed (E10.5+)
    TRANSITIONING = "transitioning"                 # In transition between states

class CellCyclePhase(Enum):
    """Cell cycle phases for neuroepithelial cells."""
    G1 = "g1"           # Gap 1 phase
    S = "s"             # Synthesis phase
    G2 = "g2"           # Gap 2 phase
    M = "m"             # Mitosis phase
    G0 = "g0"           # Quiescent phase

class DivisionType(Enum):
    """Types of cell division."""
    SYMMETRIC_PROLIFERATIVE = "symmetric_proliferative"     # 2 progenitors
    SYMMETRIC_DIFFERENTIATIVE = "symmetric_differentiative" # 2 committed cells
    ASYMMETRIC = "asymmetric"                               # 1 progenitor + 1 committed

@dataclass
class MolecularMarker:
    """Molecular marker definition for cell identification."""
    marker_name: str                    # Marker gene/protein name
    expression_level: float             # Expression level (0-1)
    temporal_window: tuple[float, float] # Developmental window (weeks)
    cellular_localization: str          # Subcellular localization
    functional_role: str                # Biological function

@dataclass
class CompetencyWindow:
    """Temporal competency window for cell fate specification."""
    fate_type: str                      # Target cell fate
    competency_start: float             # Start time (developmental weeks)
    competency_end: float               # End time (developmental weeks)
    competency_strength: float          # Competency level (0-1)
    required_signals: List[str]         # Required morphogen/signaling molecules

@dataclass
class CellStateProperties:
    """Properties defining a neuroepithelial cell state."""
    cell_type: NeuroepithelialCellType
    molecular_markers: Dict[str, MolecularMarker]
    competency_windows: List[CompetencyWindow]
    cell_cycle_length_hours: float     # Typical cell cycle length
    division_probabilities: Dict[DivisionType, float]  # Division type probabilities
    morphogen_responsiveness: Dict[str, float]  # Responsiveness to each morphogen
    spatial_constraints: Dict[str, float]       # Spatial positioning constraints
