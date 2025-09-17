"""
Spindle Orientation Controller

This module controls mitotic spindle orientation and positioning
based on cell polarity, tissue context, and regulatory signals.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .division_pattern_types import SpindleOrientation, SpindleState, CellPolarityState


@dataclass
class SpindleOrientationResult:
    """Result of spindle orientation calculation"""
    orientation: SpindleOrientation
    angle_to_axis: float
    stability: float
    centrosome_positions: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    microtubule_attachment_points: List[Tuple[float, float, float]]


class SpindleOrientationController:
    """
    Controls mitotic spindle orientation based on cell polarity,
    tissue context, and regulatory signals.
    """
    
    def __init__(self):
        """Initialize spindle orientation controller"""
        self.orientation_biases: Dict[str, Dict[SpindleOrientation, float]] = {}
        self.regulatory_influences: Dict[str, Dict[SpindleOrientation, float]] = {}
        self._setup_orientation_biases()
        self._setup_regulatory_influences()
    
    def _setup_orientation_biases(self) -> None:
        """Setup orientation biases for different developmental stages"""
        # Early embryonic: mostly parallel (symmetric division)
        self.orientation_biases["early_embryonic"] = {
            SpindleOrientation.PARALLEL: 0.7,
            SpindleOrientation.PERPENDICULAR: 0.2,
            SpindleOrientation.OBLIQUE: 0.1
        }
        
        # Mid embryonic: balanced
        self.orientation_biases["mid_embryonic"] = {
            SpindleOrientation.PARALLEL: 0.4,
            SpindleOrientation.PERPENDICULAR: 0.4,
            SpindleOrientation.OBLIQUE: 0.2
        }
        
        # Late embryonic: more perpendicular (asymmetric division)
        self.orientation_biases["late_embryonic"] = {
            SpindleOrientation.PARALLEL: 0.2,
            SpindleOrientation.PERPENDICULAR: 0.6,
            SpindleOrientation.OBLIQUE: 0.2
        }
        
        # Fetal: mostly perpendicular (differentiation)
        self.orientation_biases["fetal"] = {
            SpindleOrientation.PARALLEL: 0.1,
            SpindleOrientation.PERPENDICULAR: 0.7,
            SpindleOrientation.OBLIQUE: 0.2
        }
    
    def _setup_regulatory_influences(self) -> None:
        """Setup regulatory factor influences on spindle orientation"""
        # Notch signaling promotes perpendicular orientation
        self.regulatory_influences["notch_activity"] = {
            SpindleOrientation.PARALLEL: -0.3,
            SpindleOrientation.PERPENDICULAR: 0.4,
            SpindleOrientation.OBLIQUE: -0.1
        }
        
        # Wnt signaling promotes parallel orientation
        self.regulatory_influences["wnt_signaling"] = {
            SpindleOrientation.PARALLEL: 0.4,
            SpindleOrientation.PERPENDICULAR: -0.2,
            SpindleOrientation.OBLIQUE: -0.2
        }
        
        # Cell density promotes perpendicular orientation
        self.regulatory_influences["cell_density"] = {
            SpindleOrientation.PARALLEL: -0.2,
            SpindleOrientation.PERPENDICULAR: 0.3,
            SpindleOrientation.OBLIQUE: -0.1
        }
        
        # Differentiation pressure promotes perpendicular orientation
        self.regulatory_influences["differentiation_pressure"] = {
            SpindleOrientation.PARALLEL: -0.4,
            SpindleOrientation.PERPENDICULAR: 0.5,
            SpindleOrientation.OBLIQUE: -0.1
        }
    
    def calculate_spindle_orientation(self, cell_polarity: CellPolarityState,
                                    developmental_stage: str,
                                    regulatory_factors: Dict[str, float]) -> SpindleOrientationResult:
        """
        Calculate spindle orientation based on cell context
        
        Args:
            cell_polarity: Current cell polarity state
            developmental_stage: Current developmental stage
            regulatory_factors: Current regulatory factor levels
            
        Returns:
            SpindleOrientationResult with orientation details
        """
        # Get base orientation probabilities
        if developmental_stage not in self.orientation_biases:
            developmental_stage = "mid_embryonic"
        
        orientation_probs = self.orientation_biases[developmental_stage].copy()
        
        # Adjust based on cell polarity
        orientation_probs = self._adjust_for_cell_polarity(orientation_probs, cell_polarity)
        
        # Adjust based on regulatory factors
        orientation_probs = self._adjust_for_regulatory_factors(orientation_probs, regulatory_factors)
        
        # Select orientation
        orientations = list(orientation_probs.keys())
        probabilities = list(orientation_probs.values())
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        selected_orientation = np.random.choice(orientations, p=probabilities)
        
        # Calculate additional parameters
        angle_to_axis = self._calculate_angle_to_axis(selected_orientation)
        stability = self._calculate_spindle_stability(selected_orientation, cell_polarity, regulatory_factors)
        centrosome_positions = self._calculate_centrosome_positions(selected_orientation, cell_polarity)
        attachment_points = self._calculate_attachment_points(selected_orientation, cell_polarity)
        
        return SpindleOrientationResult(
            orientation=selected_orientation,
            angle_to_axis=angle_to_axis,
            stability=stability,
            centrosome_positions=centrosome_positions,
            microtubule_attachment_points=attachment_points
        )
    
    def _adjust_for_cell_polarity(self, orientation_probs: Dict[SpindleOrientation, float],
                                 cell_polarity: CellPolarityState) -> Dict[SpindleOrientation, float]:
        """Adjust orientation probabilities based on cell polarity"""
        adjusted = orientation_probs.copy()
        
        if cell_polarity.polarity_type.value == "bipolar":
            # Bipolar cells tend to have perpendicular spindles
            adjusted[SpindleOrientation.PERPENDICULAR] *= 1.3
            adjusted[SpindleOrientation.PARALLEL] *= 0.8
        
        elif cell_polarity.polarity_type.value == "apical":
            # Apical cells tend to have parallel spindles
            adjusted[SpindleOrientation.PARALLEL] *= 1.2
            adjusted[SpindleOrientation.PERPENDICULAR] *= 0.9
        
        elif cell_polarity.polarity_type.value == "basal":
            # Basal cells tend to have perpendicular spindles
            adjusted[SpindleOrientation.PERPENDICULAR] *= 1.2
            adjusted[SpindleOrientation.PARALLEL] *= 0.9
        
        # Strong polarity promotes more defined orientations
        polarity_strength = cell_polarity.polarity_strength
        if polarity_strength > 0.7:
            # Reduce oblique probability
            adjusted[SpindleOrientation.OBLIQUE] *= 0.5
        elif polarity_strength < 0.3:
            # Increase oblique probability
            adjusted[SpindleOrientation.OBLIQUE] *= 1.5
        
        return adjusted
    
    def _adjust_for_regulatory_factors(self, orientation_probs: Dict[SpindleOrientation, float],
                                     regulatory_factors: Dict[str, float]) -> Dict[SpindleOrientation, float]:
        """Adjust orientation probabilities based on regulatory factors"""
        adjusted = orientation_probs.copy()
        
        for factor_name, factor_value in regulatory_factors.items():
            if factor_name in self.regulatory_influences:
                influences = self.regulatory_influences[factor_name]
                
                # Normalize factor value to 0-1 range
                normalized_value = max(0.0, min(1.0, factor_value))
                
                for orientation, influence in influences.items():
                    # Apply influence proportional to factor value
                    adjustment = influence * normalized_value * 0.5
                    adjusted[orientation] *= (1.0 + adjustment)
        
        return adjusted
    
    def _calculate_angle_to_axis(self, orientation: SpindleOrientation) -> float:
        """Calculate angle between spindle and tissue axis"""
        if orientation == SpindleOrientation.PARALLEL:
            return 0.0
        elif orientation == SpindleOrientation.PERPENDICULAR:
            return 90.0
        else:  # OBLIQUE
            return np.random.uniform(30.0, 60.0)
    
    def _calculate_spindle_stability(self, orientation: SpindleOrientation,
                                   cell_polarity: CellPolarityState,
                                   regulatory_factors: Dict[str, float]) -> float:
        """Calculate spindle stability"""
        base_stability = 0.8
        
        # Polarity strength affects stability
        polarity_effect = cell_polarity.polarity_strength * 0.2
        
        # Regulatory factors affect stability
        regulatory_effect = 0.0
        for factor_name, factor_value in regulatory_factors.items():
            if factor_name in ["notch_activity", "wnt_signaling"]:
                # These factors promote stability
                regulatory_effect += factor_value * 0.1
        
        # Orientation affects stability
        orientation_effect = 0.0
        if orientation == SpindleOrientation.PARALLEL:
            orientation_effect = 0.1  # More stable
        elif orientation == SpindleOrientation.OBLIQUE:
            orientation_effect = -0.1  # Less stable
        
        stability = base_stability + polarity_effect + regulatory_effect + orientation_effect
        return max(0.0, min(1.0, stability))
    
    def _calculate_centrosome_positions(self, orientation: SpindleOrientation,
                                      cell_polarity: CellPolarityState) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate centrosome positions based on orientation"""
        # Base positions (centered)
        center = (0.0, 0.0, 0.0)
        separation = 1.0
        
        if orientation == SpindleOrientation.PARALLEL:
            # Centrosomes along apical-basal axis
            centrosome1 = (center[0], center[1] - separation/2, center[2])
            centrosome2 = (center[0], center[1] + separation/2, center[2])
        
        elif orientation == SpindleOrientation.PERPENDICULAR:
            # Centrosomes perpendicular to apical-basal axis
            centrosome1 = (center[0] - separation/2, center[1], center[2])
            centrosome2 = (center[0] + separation/2, center[1], center[2])
        
        else:  # OBLIQUE
            # Random oblique orientation
            angle = np.random.uniform(0, 2 * np.pi)
            offset_x = np.cos(angle) * separation / 2
            offset_y = np.sin(angle) * separation / 2
            centrosome1 = (center[0] - offset_x, center[1] - offset_y, center[2])
            centrosome2 = (center[0] + offset_x, center[1] + offset_y, center[2])
        
        return centrosome1, centrosome2
    
    def _calculate_attachment_points(self, orientation: SpindleOrientation,
                                   cell_polarity: CellPolarityState) -> List[Tuple[float, float, float]]:
        """Calculate microtubule attachment points"""
        attachment_points = []
        
        # Base attachment points
        base_points = [
            (0.0, 0.0, 0.0),  # Center
            (1.0, 0.0, 0.0),  # Right
            (-1.0, 0.0, 0.0),  # Left
            (0.0, 1.0, 0.0),  # Up
            (0.0, -1.0, 0.0),  # Down
        ]
        
        # Add orientation-specific points
        if orientation == SpindleOrientation.PARALLEL:
            # More points along apical-basal axis
            attachment_points.extend([
                (0.0, 1.5, 0.0),
                (0.0, -1.5, 0.0)
            ])
        elif orientation == SpindleOrientation.PERPENDICULAR:
            # More points perpendicular to axis
            attachment_points.extend([
                (1.5, 0.0, 0.0),
                (-1.5, 0.0, 0.0)
            ])
        
        attachment_points.extend(base_points)
        return attachment_points
    
    def update_spindle_state(self, spindle_state: SpindleState,
                           time_delta: float,
                           regulatory_factors: Dict[str, float]) -> SpindleState:
        """Update spindle state over time"""
        # Update stability based on regulatory factors
        stability_change = 0.0
        for factor_name, factor_value in regulatory_factors.items():
            if factor_name in ["notch_activity", "wnt_signaling"]:
                stability_change += factor_value * 0.01 * time_delta
        
        new_stability = max(0.0, min(1.0, spindle_state.stability + stability_change))
        
        # Update centrosome separation
        separation_change = 0.1 * time_delta
        new_separation = max(0.0, min(1.0, spindle_state.centrosome_separation + separation_change))
        
        # Update microtubule density
        density_change = 0.05 * time_delta
        new_density = max(0.0, min(1.0, spindle_state.microtubule_density + density_change))
        
        return SpindleState(
            orientation=spindle_state.orientation,
            angle_to_axis=spindle_state.angle_to_axis,
            stability=new_stability,
            centrosome_separation=new_separation,
            microtubule_density=new_density
        )
