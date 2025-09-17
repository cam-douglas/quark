#!/usr/bin/env python3
"""Core parameter types and data structures for morphogen systems.

Defines fundamental data structures used across all morphogen parameter modules.

Integration: Foundation types for all morphogen parameter systems
Rationale: Centralized type definitions ensure consistency across modules
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class MorphogenType(Enum):
    """Morphogen type enumeration."""
    SHH = "sonic_hedgehog"
    BMP = "bone_morphogenetic_protein"
    WNT = "wingless_related"
    FGF = "fibroblast_growth_factor"

@dataclass
class DiffusionParameters:
    """Diffusion parameters for morphogen transport."""
    diffusion_coefficient: float  # µm²/s
    degradation_rate: float      # 1/s
    production_rate: float       # nM/s
    half_life: float            # seconds
    
    def __post_init__(self):
        """Validate parameter ranges."""
        if self.diffusion_coefficient <= 0:
            raise ValueError("Diffusion coefficient must be positive")
        if self.degradation_rate <= 0:
            raise ValueError("Degradation rate must be positive")
        if self.production_rate < 0:
            raise ValueError("Production rate must be non-negative")
        if self.half_life <= 0:
            raise ValueError("Half-life must be positive")

@dataclass
class SourceParameters:
    """Source region parameters for morphogen production."""
    location: str               # Anatomical location
    intensity: float           # Production intensity (nM/s)
    spatial_extent: float      # Source size (µm)
    temporal_profile: str      # Time-dependent expression profile
    
    def __post_init__(self):
        """Validate parameter ranges."""
        if self.intensity < 0:
            raise ValueError("Source intensity must be non-negative")
        if self.spatial_extent <= 0:
            raise ValueError("Spatial extent must be positive")

@dataclass
class InteractionParameters:
    """Parameters for morphogen cross-regulation."""
    target_morphogen: str
    interaction_type: str      # 'inhibition', 'activation', 'competition', 'modulation'
    strength: float           # Interaction strength coefficient
    hill_coefficient: float   # Cooperativity parameter
    threshold: float          # Half-maximal concentration (nM)
    
    def __post_init__(self):
        """Validate parameter ranges."""
        valid_types = {'inhibition', 'activation', 'competition', 'modulation'}
        if self.interaction_type not in valid_types:
            raise ValueError(f"Interaction type must be one of {valid_types}")
        if self.strength <= 0:
            raise ValueError("Interaction strength must be positive")
        if self.hill_coefficient <= 0:
            raise ValueError("Hill coefficient must be positive")
        if self.threshold <= 0:
            raise ValueError("Threshold concentration must be positive")

@dataclass
class ParameterSet:
    """Complete parameter set for a morphogen."""
    morphogen_name: str
    diffusion: DiffusionParameters
    source: SourceParameters
    interactions: List[InteractionParameters]
    species: str = "mouse"
    developmental_stage: str = "E8.5-E10.5"
    
    def __post_init__(self):
        """Validate parameter set consistency."""
        if not self.morphogen_name:
            raise ValueError("Morphogen name cannot be empty")
        
        # Validate interaction target morphogens are different from source
        for interaction in self.interactions:
            if interaction.target_morphogen == self.morphogen_name:
                raise ValueError("Morphogen cannot interact with itself")

class ParameterValidationResult:
    """Results of parameter validation."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
    
    def add_info(self, key: str, value: Any) -> None:
        """Add validation info."""
        self.info[key] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info
        }

__all__ = [
    "MorphogenType",
    "DiffusionParameters", 
    "SourceParameters",
    "InteractionParameters",
    "ParameterSet",
    "ParameterValidationResult"
]
