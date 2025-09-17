#!/usr/bin/env python3
"""Meninges System Type Definitions.

Core type definitions and data structures for meningeal scaffold system
including layer types, mechanical properties, and attachment configurations.

Integration: Type definitions for meninges scaffold system
Rationale: Centralized type definitions following architecture patterns
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class MeningesLayerType(Enum):
    """Types of meningeal layers."""
    DURA_MATER = "dura_mater"           # Outer protective layer
    ARACHNOID_MATER = "arachnoid_mater" # Middle layer with CSF space
    PIA_MATER = "pia_mater"             # Inner layer on neural tissue

class AttachmentPointType(Enum):
    """Types of meningeal attachment points."""
    SKULL_PRIMORDIUM = "skull_primordium"     # Attachment to developing skull
    VERTEBRAL_COLUMN = "vertebral_column"     # Spinal attachment points
    CRANIAL_SUTURES = "cranial_sutures"      # Suture line attachments
    VASCULAR_ENTRY = "vascular_entry"        # Blood vessel entry points

@dataclass
class MechanicalProperties:
    """Mechanical properties of meningeal tissue."""
    elastic_modulus_pa: float           # Young's modulus (Pa)
    poisson_ratio: float               # Poisson's ratio (dimensionless)
    thickness_um: float                # Layer thickness (µm)
    density_kg_m3: float              # Tissue density (kg/m³)
    permeability_m2: float            # Hydraulic permeability (m²)
    tensile_strength_pa: float        # Ultimate tensile strength (Pa)

@dataclass
class AttachmentPoint:
    """Meningeal attachment point definition."""
    attachment_type: AttachmentPointType
    location: Tuple[float, float, float]  # µm coordinates
    attachment_strength_n: float          # Attachment force capacity (N)
    developmental_week: float             # Week of attachment formation
    region_radius_um: float              # Attachment region radius (µm)

@dataclass
class MeningesLayer:
    """Definition of a meningeal layer."""
    layer_type: MeningesLayerType
    mechanical_properties: MechanicalProperties
    attachment_points: List[AttachmentPoint]
    developmental_week_start: float      # Week when layer begins forming
    developmental_week_mature: float     # Week when layer is mature
    surface_mesh: Optional[np.ndarray]   # 3D surface mesh (if computed)
