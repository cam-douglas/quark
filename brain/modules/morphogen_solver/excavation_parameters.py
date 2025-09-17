#!/usr/bin/env python3
"""Excavation Parameters and Result Types.

Parameter definitions and result structures for voxel excavation algorithm
including validation parameters and excavation result data structures.

Integration: Parameter definitions for voxel excavation system
Rationale: Centralized parameter and result type definitions
"""

from typing import Dict
from dataclasses import dataclass
import numpy as np

from .ventricular_types import VentricleType

@dataclass
class ExcavationParameters:
    """Parameters for voxel excavation algorithm."""
    min_cavity_volume_mm3: float = 0.001      # Minimum cavity volume
    max_cavity_volume_mm3: float = 0.1        # Maximum cavity volume
    connectivity_radius_um: float = 5.0       # Connection detection radius
    smoothing_iterations: int = 2              # Cavity smoothing iterations
    validation_threshold: float = 0.8          # Volume validation threshold

@dataclass
class ExcavationResult:
    """Result of voxel excavation process."""
    excavated_mask: np.ndarray                 # Boolean mask of excavated voxels
    cavity_volumes: Dict[VentricleType, float] # Volume per ventricle (mm³)
    connectivity_validated: bool               # CSF pathway validation result
    total_excavated_volume: float              # Total volume excavated (mm³)
    validation_score: float                    # Validation against references
