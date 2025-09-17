"""
Division Pattern Types

This module defines types and enums for cell division patterns,
including symmetric vs asymmetric division decisions and spindle orientation.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DivisionType(Enum):
    """Types of cell division patterns"""
    SYMMETRIC_PROLIFERATIVE = "symmetric_proliferative"  # Both daughters remain progenitors
    SYMMETRIC_DIFFERENTIATIVE = "symmetric_differentiative"  # Both daughters differentiate
    ASYMMETRIC = "asymmetric"  # One progenitor, one differentiated


class SpindleOrientation(Enum):
    """Spindle orientation relative to tissue axis"""
    PARALLEL = "parallel"  # Parallel to apical-basal axis
    PERPENDICULAR = "perpendicular"  # Perpendicular to apical-basal axis
    OBLIQUE = "oblique"  # At an angle to apical-basal axis


class CellPolarity(Enum):
    """Cell polarity states"""
    APICAL = "apical"  # Apical polarity
    BASAL = "basal"  # Basal polarity
    BIPOLAR = "bipolar"  # Both apical and basal
    UNPOLARIZED = "unpolarized"  # No clear polarity


@dataclass
class DivisionDecision:
    """Decision about cell division pattern"""
    division_type: DivisionType
    spindle_orientation: SpindleOrientation
    cell_polarity: CellPolarity
    division_plane: Tuple[float, float, float]  # Normal vector
    confidence: float
    regulatory_factors: Dict[str, float]


@dataclass
class SpindleState:
    """Current state of mitotic spindle"""
    orientation: SpindleOrientation
    angle_to_axis: float  # Angle in degrees
    stability: float  # 0.0 to 1.0
    centrosome_separation: float  # 0.0 to 1.0
    microtubule_density: float  # 0.0 to 1.0


@dataclass
class CellPolarityState:
    """Current cell polarity state"""
    polarity_type: CellPolarity
    apical_markers: Dict[str, float]  # Marker expression levels
    basal_markers: Dict[str, float]  # Marker expression levels
    polarity_strength: float  # 0.0 to 1.0
    orientation_vector: Tuple[float, float, float]  # Polarity axis


@dataclass
class DivisionPlaneState:
    """State of the division plane"""
    normal_vector: Tuple[float, float, float]
    position: Tuple[float, float, float]
    stability: float  # 0.0 to 1.0
    regulatory_signals: Dict[str, float]


class DivisionPatternPredictor:
    """
    Predicts division patterns based on cellular context,
    regulatory factors, and developmental stage.
    """
    
    def __init__(self):
        """Initialize division pattern predictor"""
        self.developmental_biases: Dict[str, Dict[DivisionType, float]] = {}
        self.regulatory_thresholds: Dict[str, float] = {}
        self._setup_developmental_biases()
        self._setup_regulatory_thresholds()
    
    def _setup_developmental_biases(self) -> None:
        """Setup developmental stage biases for division patterns"""
        # Early embryonic: mostly symmetric proliferative
        self.developmental_biases["early_embryonic"] = {
            DivisionType.SYMMETRIC_PROLIFERATIVE: 0.8,
            DivisionType.ASYMMETRIC: 0.15,
            DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.05
        }
        
        # Mid embryonic: balanced
        self.developmental_biases["mid_embryonic"] = {
            DivisionType.SYMMETRIC_PROLIFERATIVE: 0.5,
            DivisionType.ASYMMETRIC: 0.35,
            DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.15
        }
        
        # Late embryonic: more asymmetric
        self.developmental_biases["late_embryonic"] = {
            DivisionType.SYMMETRIC_PROLIFERATIVE: 0.3,
            DivisionType.ASYMMETRIC: 0.5,
            DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.2
        }
        
        # Fetal: mostly asymmetric and differentiative
        self.developmental_biases["fetal"] = {
            DivisionType.SYMMETRIC_PROLIFERATIVE: 0.1,
            DivisionType.ASYMMETRIC: 0.4,
            DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.5
        }
    
    def _setup_regulatory_thresholds(self) -> None:
        """Setup regulatory factor thresholds for division decisions"""
        self.regulatory_thresholds = {
            "notch_activity": 0.5,
            "wnt_signaling": 0.6,
            "shh_signaling": 0.4,
            "bmp_signaling": 0.7,
            "cell_density": 0.8,
            "nutrient_availability": 0.6,
            "differentiation_pressure": 0.3
        }
    
    def predict_division_type(self, developmental_stage: str, 
                            regulatory_factors: Dict[str, float]) -> DivisionType:
        """
        Predict division type based on developmental stage and regulatory factors
        
        Args:
            developmental_stage: Current developmental stage
            regulatory_factors: Current regulatory factor levels
            
        Returns:
            Predicted division type
        """
        # Get developmental bias
        if developmental_stage not in self.developmental_biases:
            developmental_stage = "mid_embryonic"
        
        base_probabilities = self.developmental_biases[developmental_stage].copy()
        
        # Adjust based on regulatory factors
        adjusted_probabilities = self._adjust_probabilities_for_regulatory_factors(
            base_probabilities, regulatory_factors
        )
        
        # Select division type based on probabilities
        division_types = list(adjusted_probabilities.keys())
        probabilities = list(adjusted_probabilities.values())
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        return np.random.choice(division_types, p=probabilities)
    
    def _adjust_probabilities_for_regulatory_factors(self, 
                                                   base_probabilities: Dict[DivisionType, float],
                                                   regulatory_factors: Dict[str, float]) -> Dict[DivisionType, float]:
        """Adjust division probabilities based on regulatory factors"""
        adjusted = base_probabilities.copy()
        
        # Notch activity promotes asymmetric division
        notch_activity = regulatory_factors.get("notch_activity", 0.5)
        if notch_activity > self.regulatory_thresholds["notch_activity"]:
            adjusted[DivisionType.ASYMMETRIC] *= 1.5
            adjusted[DivisionType.SYMMETRIC_PROLIFERATIVE] *= 0.8
        
        # Wnt signaling promotes symmetric proliferative
        wnt_signaling = regulatory_factors.get("wnt_signaling", 0.5)
        if wnt_signaling > self.regulatory_thresholds["wnt_signaling"]:
            adjusted[DivisionType.SYMMETRIC_PROLIFERATIVE] *= 1.3
            adjusted[DivisionType.SYMMETRIC_DIFFERENTIATIVE] *= 0.7
        
        # High cell density promotes differentiation
        cell_density = regulatory_factors.get("cell_density", 0.5)
        if cell_density > self.regulatory_thresholds["cell_density"]:
            adjusted[DivisionType.SYMMETRIC_DIFFERENTIATIVE] *= 1.4
            adjusted[DivisionType.SYMMETRIC_PROLIFERATIVE] *= 0.6
        
        # Differentiation pressure promotes asymmetric and differentiative
        diff_pressure = regulatory_factors.get("differentiation_pressure", 0.5)
        if diff_pressure > self.regulatory_thresholds["differentiation_pressure"]:
            adjusted[DivisionType.ASYMMETRIC] *= 1.2
            adjusted[DivisionType.SYMMETRIC_DIFFERENTIATIVE] *= 1.2
            adjusted[DivisionType.SYMMETRIC_PROLIFERATIVE] *= 0.8
        
        return adjusted
    
    def calculate_division_confidence(self, division_type: DivisionType,
                                    developmental_stage: str,
                                    regulatory_factors: Dict[str, float]) -> float:
        """Calculate confidence in division type prediction"""
        base_prob = self.developmental_biases.get(developmental_stage, {}).get(division_type, 0.33)
        
        # Adjust based on regulatory factor alignment
        regulatory_alignment = self._calculate_regulatory_alignment(division_type, regulatory_factors)
        
        # Combine base probability with regulatory alignment
        confidence = (base_prob + regulatory_alignment) / 2.0
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_regulatory_alignment(self, division_type: DivisionType,
                                      regulatory_factors: Dict[str, float]) -> float:
        """Calculate how well regulatory factors align with division type"""
        alignment_score = 0.5  # Base alignment
        
        if division_type == DivisionType.SYMMETRIC_PROLIFERATIVE:
            # Wnt signaling supports proliferative division
            wnt = regulatory_factors.get("wnt_signaling", 0.5)
            if wnt > 0.6:
                alignment_score += 0.2
            # Low differentiation pressure supports proliferative
            diff_pressure = regulatory_factors.get("differentiation_pressure", 0.5)
            if diff_pressure < 0.3:
                alignment_score += 0.2
        
        elif division_type == DivisionType.ASYMMETRIC:
            # Notch activity supports asymmetric division
            notch = regulatory_factors.get("notch_activity", 0.5)
            if notch > 0.5:
                alignment_score += 0.3
            # Moderate differentiation pressure supports asymmetric
            diff_pressure = regulatory_factors.get("differentiation_pressure", 0.5)
            if 0.3 <= diff_pressure <= 0.7:
                alignment_score += 0.1
        
        elif division_type == DivisionType.SYMMETRIC_DIFFERENTIATIVE:
            # High differentiation pressure supports differentiative
            diff_pressure = regulatory_factors.get("differentiation_pressure", 0.5)
            if diff_pressure > 0.7:
                alignment_score += 0.3
            # High cell density supports differentiative
            cell_density = regulatory_factors.get("cell_density", 0.5)
            if cell_density > 0.8:
                alignment_score += 0.2
        
        return min(1.0, alignment_score)
