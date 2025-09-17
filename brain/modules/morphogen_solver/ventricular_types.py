#!/usr/bin/env python3
"""Ventricular System Type Definitions.

Core type definitions and data structures for ventricular cavity topology
system including ventricle types, regions, and configuration parameters.

Integration: Type definitions for ventricular topology system
Rationale: Centralized type definitions following architecture patterns
"""

from typing import Tuple, List
from dataclasses import dataclass
from enum import Enum

class VentricleType(Enum):
    """Types of ventricular cavities in embryonic brain."""
    LATERAL_LEFT = "lateral_left"
    LATERAL_RIGHT = "lateral_right"
    THIRD = "third"
    FOURTH = "fourth"
    CEREBRAL_AQUEDUCT = "cerebral_aqueduct"

@dataclass
class VentricularRegion:
    """Definition of a ventricular cavity region."""
    ventricle_type: VentricleType
    center_position: Tuple[float, float, float]  # µm coordinates
    dimensions: Tuple[float, float, float]       # width, height, depth (µm)
    shape_type: str                              # "ellipsoid", "tube", "irregular"
    connectivity: List[VentricleType]            # Connected ventricles
    developmental_week: float                    # Embryonic week of formation
