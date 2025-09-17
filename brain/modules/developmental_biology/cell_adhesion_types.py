"""
Cell Adhesion Types and Parameters

Defines types and parameters for cell-cell adhesion and repulsion forces.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType


class AdhesionType(Enum):
    """Types of cell adhesion"""
    CADHERIN_ADHESION = "cadherin_adhesion"
    INTEGRIN_ADHESION = "integrin_adhesion"
    TIGHT_JUNCTION = "tight_junction"
    GAP_JUNCTION = "gap_junction"
    DESMOSOME = "desmosome"


class RepulsionType(Enum):
    """Types of cell repulsion"""
    STERIC_REPULSION = "steric_repulsion"
    ELECTROSTATIC_REPULSION = "electrostatic_repulsion"
    HYDROPHOBIC_REPULSION = "hydrophobic_repulsion"
    MECHANICAL_REPULSION = "mechanical_repulsion"


@dataclass
class AdhesionParameters:
    """Parameters for cell adhesion"""
    adhesion_strength: float
    adhesion_range: float
    adhesion_threshold: float
    zone_specific_strength: Dict[ZoneType, float]


@dataclass
class RepulsionParameters:
    """Parameters for cell repulsion"""
    repulsion_strength: float
    repulsion_range: float
    repulsion_threshold: float
    zone_specific_strength: Dict[ZoneType, float]


@dataclass
class AdhesionForce:
    """Calculated adhesion force between cells"""
    force_magnitude: float
    force_direction: tuple
    adhesion_type: AdhesionType
    interaction_range: float


@dataclass
class RepulsionForce:
    """Calculated repulsion force between cells"""
    force_magnitude: float
    force_direction: tuple
    repulsion_type: RepulsionType
    interaction_range: float
