#!/usr/bin/env python3
"""Literature Database Types - Core data structures and enums for morphogen parameter management.

This module defines the fundamental types, enums, and data classes used throughout
the literature database system for organizing morphogen research parameters.

Integration: Core types for literature_database package components.
Rationale: Centralized type definitions for consistent data modeling.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

class MorphogenType(Enum):
    """Types of morphogens in neural development."""
    SHH = "sonic_hedgehog"
    BMP = "bone_morphogenetic_protein"
    WNT = "wingless_related"
    FGF = "fibroblast_growth_factor"

class ParameterType(Enum):
    """Types of morphogen parameters from literature."""
    DIFFUSION_COEFFICIENT = "diffusion_coefficient"  # μm²/s
    DECAY_RATE = "decay_rate"  # 1/s
    PRODUCTION_RATE = "production_rate"  # molecules/s
    CONCENTRATION_RANGE = "concentration_range"  # nM or μM
    GRADIENT_LENGTH = "gradient_length"  # μm
    BINDING_AFFINITY = "binding_affinity"  # Kd in nM
    PROTEIN_HALF_LIFE = "protein_half_life"  # hours
    MRNA_HALF_LIFE = "mrna_half_life"  # hours
    EXPRESSION_DOMAIN = "expression_domain"  # spatial coordinates
    TEMPORAL_PROFILE = "temporal_profile"  # time series data

class DevelopmentalStage(Enum):
    """Embryonic developmental stages (Carnegie stages)."""
    CS9 = "carnegie_stage_9"   # Week 3, neural plate formation
    CS10 = "carnegie_stage_10"  # Week 3, neural groove
    CS11 = "carnegie_stage_11"  # Week 4, neural tube closure
    CS12 = "carnegie_stage_12"  # Week 4, somite formation
    CS13 = "carnegie_stage_13"  # Week 4-5, brain vesicles
    CS15 = "carnegie_stage_15"  # Week 5, brain segmentation
    CS17 = "carnegie_stage_17"  # Week 6, early organogenesis
    CS20 = "carnegie_stage_20"  # Week 7, neural differentiation
    CS23 = "carnegie_stage_23"  # Week 8, fetal transition

class ConfidenceLevel(Enum):
    """Confidence level in parameter measurement."""
    HIGH = "high"        # Multiple independent studies, consistent values
    MEDIUM = "medium"    # Few studies or some variability
    LOW = "low"         # Single study or high variability
    PRELIMINARY = "preliminary"  # In vitro only or model-based

@dataclass
class Parameter:
    """Represents a morphogen parameter from literature."""
    parameter_id: str
    morphogen: MorphogenType
    parameter_type: ParameterType
    value: float
    unit: str
    std_deviation: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    developmental_stage: Optional[DevelopmentalStage] = None
    species: str = "human"
    experimental_method: str = ""
    tissue_type: str = ""
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    notes: str = ""
    date_added: Optional[str] = None
    expert_validated: bool = False
    expert_comments: str = ""
    
    def __post_init__(self):
        if self.date_added is None:
            self.date_added = datetime.now().isoformat()

@dataclass
class Citation:
    """Represents a literature citation for parameters."""
    citation_id: str
    authors: str
    title: str
    journal: str
    year: int
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: str = ""
    experimental_methods: str = ""
    species_studied: str = ""
    relevance_score: float = 0.0  # 0-1 relevance to neural tube development
    quality_score: float = 0.0    # 0-1 experimental quality assessment

@dataclass
class ParameterSet:
    """Collection of related parameters for specific experimental conditions."""
    set_id: str
    morphogen: MorphogenType
    developmental_stage: DevelopmentalStage
    experimental_conditions: Dict[str, Any]
    parameters: List[str]  # parameter_ids
    citation_id: str
    confidence_assessment: str = ""
    expert_review_status: str = "pending"

# Utility functions for type validation and conversion
def validate_parameter_value(parameter_type: ParameterType, value: float, unit: str) -> bool:
    """Validate parameter value based on biological constraints."""
    constraints = {
        ParameterType.DIFFUSION_COEFFICIENT: (1e-6, 1.0),  # μm²/s reasonable range
        ParameterType.DECAY_RATE: (1e-6, 1.0),  # 1/s reasonable range
        ParameterType.PRODUCTION_RATE: (0.0, 1e12),  # molecules/s broad range
        ParameterType.CONCENTRATION_RANGE: (0.0, 1000.0),  # nM/μM range
        ParameterType.GRADIENT_LENGTH: (1.0, 10000.0),  # μm range
        ParameterType.BINDING_AFFINITY: (0.001, 1000.0),  # nM range
        ParameterType.PROTEIN_HALF_LIFE: (0.1, 168.0),  # hours (6 min to 1 week)
        ParameterType.MRNA_HALF_LIFE: (0.1, 48.0),  # hours (6 min to 2 days)
    }
    
    if parameter_type in constraints:
        min_val, max_val = constraints[parameter_type]
        return min_val <= value <= max_val
    
    return True  # No constraints for other parameter types

def get_standard_units(parameter_type: ParameterType) -> str:
    """Get standard units for parameter type."""
    standard_units = {
        ParameterType.DIFFUSION_COEFFICIENT: "μm²/s",
        ParameterType.DECAY_RATE: "1/s",
        ParameterType.PRODUCTION_RATE: "molecules/s",
        ParameterType.CONCENTRATION_RANGE: "nM",
        ParameterType.GRADIENT_LENGTH: "μm",
        ParameterType.BINDING_AFFINITY: "nM",
        ParameterType.PROTEIN_HALF_LIFE: "hours",
        ParameterType.MRNA_HALF_LIFE: "hours",
        ParameterType.EXPRESSION_DOMAIN: "μm",
        ParameterType.TEMPORAL_PROFILE: "hours"
    }
    
    return standard_units.get(parameter_type, "unknown")

def get_developmental_stage_timepoint(stage: DevelopmentalStage) -> float:
    """Get approximate timepoint in days for developmental stage."""
    timepoints = {
        DevelopmentalStage.CS9: 19.0,   # Day 19
        DevelopmentalStage.CS10: 22.0,  # Day 22
        DevelopmentalStage.CS11: 24.0,  # Day 24
        DevelopmentalStage.CS12: 26.0,  # Day 26
        DevelopmentalStage.CS13: 28.0,  # Day 28
        DevelopmentalStage.CS15: 33.0,  # Day 33
        DevelopmentalStage.CS17: 41.0,  # Day 41
        DevelopmentalStage.CS20: 47.0,  # Day 47
        DevelopmentalStage.CS23: 56.0,  # Day 56
    }
    
    return timepoints.get(stage, 30.0)  # Default to ~4 weeks

# Export all public types
__all__ = [
    "MorphogenType",
    "ParameterType", 
    "DevelopmentalStage",
    "ConfidenceLevel",
    "Parameter",
    "Citation",
    "ParameterSet",
    "validate_parameter_value",
    "get_standard_units",
    "get_developmental_stage_timepoint"
]