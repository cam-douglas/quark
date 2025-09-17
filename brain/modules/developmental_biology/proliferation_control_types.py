"""
Proliferation Control Types

Types and data structures for proliferation rate control.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class ProliferationState(Enum):
    """Proliferation states"""
    ACTIVE = "active"
    QUIESCENT = "quiescent"
    ARRESTED = "arrested"
    DIFFERENTIATING = "differentiating"


@dataclass
class ProliferationContext:
    """Context for proliferation rate calculation"""
    cell_density: float  # 0.0 to 1.0
    growth_factor_levels: Dict[str, float]
    nutrient_availability: float  # 0.0 to 1.0
    developmental_stage: str
    tissue_type: str
    spatial_position: Tuple[float, float, float]
    cell_age: float  # Hours since last division


@dataclass
class ProliferationRate:
    """Proliferation rate parameters"""
    base_rate: float  # Divisions per hour
    current_rate: float  # Current effective rate
    inhibition_factor: float  # 0.0 to 1.0
    growth_factor_response: float  # 0.0 to 2.0
    density_inhibition: float  # 0.0 to 1.0
    nutrient_dependence: float  # 0.0 to 1.0


@dataclass
class ProliferationControlParameters:
    """Parameters for proliferation control"""
    base_proliferation_rate: float
    density_inhibition_threshold: float
    growth_factor_sensitivity: float
    nutrient_dependence_factor: float
    cell_cycle_checkpoint_strength: float
