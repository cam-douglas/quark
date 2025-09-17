"""
Density Inhibition Types

Types and data structures for density-dependent inhibition.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class InhibitionType(Enum):
    """Types of density inhibition"""
    CONTACT_INHIBITION = "contact_inhibition"
    CROWDING_INHIBITION = "crowding_inhibition"
    NUTRIENT_COMPETITION = "nutrient_competition"
    SPATIAL_CONSTRAINT = "spatial_constraint"


@dataclass
class DensityContext:
    """Context for density inhibition calculation"""
    local_density: float  # 0.0 to 1.0
    global_density: float  # 0.0 to 1.0
    cell_size: float  # Relative cell size
    tissue_elasticity: float  # 0.0 to 1.0
    nutrient_gradient: float  # 0.0 to 1.0
    spatial_position: Tuple[float, float, float]
    neighbor_count: int
    contact_area: float  # 0.0 to 1.0


@dataclass
class InhibitionResult:
    """Result of density inhibition calculation"""
    total_inhibition: float  # 0.0 to 1.0
    inhibition_components: Dict[InhibitionType, float]
    inhibition_strength: str  # "weak", "moderate", "strong"
    recovery_time: float  # Hours to recover


@dataclass
class InhibitionParameters:
    """Parameters for density inhibition calculations"""
    contact_threshold: float
    crowding_threshold: float
    nutrient_threshold: float
    spatial_threshold: float
    inhibition_strength: float
    recovery_rate: float
