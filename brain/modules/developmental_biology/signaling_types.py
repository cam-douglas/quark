"""
Signaling Types

This module defines shared types for growth factor signaling and
pathway management to avoid circular imports.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict
from dataclasses import dataclass
from enum import Enum


class GrowthFactorType(Enum):
    """Types of growth factors"""
    FGF = "FGF"  # Fibroblast Growth Factor
    EGF = "EGF"  # Epidermal Growth Factor
    WNT = "WNT"  # Wnt signaling
    SHH = "SHH"  # Sonic Hedgehog
    BMP = "BMP"  # Bone Morphogenetic Protein
    TGF_BETA = "TGF_BETA"  # Transforming Growth Factor Beta
    PDGF = "PDGF"  # Platelet-Derived Growth Factor
    VEGF = "VEGF"  # Vascular Endothelial Growth Factor


class SignalingPathway(Enum):
    """Signaling pathways"""
    MAPK = "MAPK"  # Mitogen-Activated Protein Kinase
    PI3K_AKT = "PI3K_AKT"  # Phosphoinositide 3-kinase/AKT
    WNT_BETA_CATENIN = "WNT_BETA_CATENIN"  # Wnt/β-catenin
    HEDGEHOG = "HEDGEHOG"  # Hedgehog signaling
    SMAD = "SMAD"  # SMAD signaling
    NOTCH = "NOTCH"  # Notch signaling


@dataclass
class GrowthFactorSignal:
    """Growth factor signal information"""
    factor_type: GrowthFactorType
    concentration: float  # 0.0 to 1.0
    receptor_expression: float  # 0.0 to 1.0
    pathway_activity: float  # 0.0 to 1.0
    downstream_effects: Dict[str, float]

