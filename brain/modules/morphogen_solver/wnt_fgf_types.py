#!/usr/bin/env python3
"""WNT/FGF Gradient System Type Definitions.

Type definitions and parameter structures for WNT and FGF morphogen systems
including gradient parameters, regional markers, and patterning specifications.

Integration: Type definitions for WNT/FGF gradient systems
Rationale: Centralized type definitions for posterior-anterior patterning
"""

from typing import Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

class WNTSignalingType(Enum):
    """Types of WNT signaling pathways."""
    CANONICAL = "canonical"           # β-catenin dependent
    NON_CANONICAL_PCP = "pcp"        # Planar cell polarity
    NON_CANONICAL_CA = "calcium"     # Calcium signaling

class FGFReceptorType(Enum):
    """Types of FGF receptors."""
    FGFR1 = "fgfr1"                  # Neural induction
    FGFR2 = "fgfr2"                  # Neural maintenance
    FGFR3 = "fgfr3"                  # Differentiation control
    FGFR4 = "fgfr4"                  # Regional specification

class RegionalMarker(Enum):
    """Regional specification markers for A-P axis."""
    FOREBRAIN = "forebrain"          # Prosencephalon
    MIDBRAIN = "midbrain"            # Mesencephalon
    HINDBRAIN = "hindbrain"          # Rhombencephalon
    SPINAL_CORD = "spinal_cord"      # Neural tube posterior

@dataclass
class WNTGradientParameters:
    """Parameters for WNT gradient system."""
    diffusion_coefficient: float     # µm²/s
    degradation_rate: float         # s⁻¹
    production_rate: float          # nM/s
    signaling_type: WNTSignalingType
    target_genes: List[str]         # Target gene names
    inhibitors: List[str]           # Inhibitor molecules

@dataclass
class FGFGradientParameters:
    """Parameters for FGF gradient system."""
    diffusion_coefficient: float     # µm²/s
    degradation_rate: float         # s⁻¹
    production_rate: float          # nM/s
    receptor_type: FGFReceptorType
    target_genes: List[str]         # Target gene names
    cofactors: List[str]            # Required cofactors

@dataclass
class RegionalSpecification:
    """Regional specification definition for A-P patterning."""
    region_marker: RegionalMarker
    wnt_concentration_range: Tuple[float, float]  # nM
    fgf_concentration_range: Tuple[float, float]  # nM
    anterior_boundary: float        # Position along A-P axis (0-1)
    posterior_boundary: float       # Position along A-P axis (0-1)
    characteristic_genes: List[str] # Region-specific genes
