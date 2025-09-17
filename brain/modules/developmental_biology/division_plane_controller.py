"""
Division Plane Controller

This module controls the positioning and orientation of the cell division plane
based on spindle orientation, cell geometry, and regulatory signals.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .division_pattern_types import SpindleOrientation, DivisionPlaneState, CellPolarityState
from .spindle_orientation_controller import SpindleOrientationResult


@dataclass
class DivisionPlaneResult:
    """Result of division plane calculation"""
    normal_vector: Tuple[float, float, float]
    position: Tuple[float, float, float]
    stability: float
    regulatory_signals: Dict[str, float]
    daughter_cell_volumes: Tuple[float, float]  # Relative volumes


class DivisionPlaneController:
    """
    Controls the positioning and orientation of the cell division plane
    based on spindle orientation, cell geometry, and regulatory signals.
    """
    
    def __init__(self):
        """Initialize division plane controller"""
        self.plane_biases: Dict[SpindleOrientation, Dict[str, float]] = {}
        self.regulatory_effects: Dict[str, Dict[str, float]] = {}
        self._setup_plane_biases()
        self._setup_regulatory_effects()
    
    def _setup_plane_biases(self) -> None:
        """Setup division plane biases for different spindle orientations"""
        # Parallel spindle: symmetric division plane
        self.plane_biases[SpindleOrientation.PARALLEL] = {
            "symmetric_division": 0.9,
            "volume_ratio": 0.5,  # Equal volumes
            "position_offset": 0.0,  # Centered
            "stability_bonus": 0.2
        }
        
        # Perpendicular spindle: asymmetric division plane
        self.plane_biases[SpindleOrientation.PERPENDICULAR] = {
            "symmetric_division": 0.1,
            "volume_ratio": 0.3,  # Unequal volumes (30:70)
            "position_offset": 0.2,  # Offset from center
            "stability_bonus": 0.1
        }
        
        # Oblique spindle: variable division plane
        self.plane_biases[SpindleOrientation.OBLIQUE] = {
            "symmetric_division": 0.5,
            "volume_ratio": 0.4,  # Slightly unequal
            "position_offset": 0.1,  # Slight offset
            "stability_bonus": 0.0
        }
    
    def _setup_regulatory_effects(self) -> None:
        """Setup regulatory factor effects on division plane"""
        # Notch activity promotes asymmetric division
        self.regulatory_effects["notch_activity"] = {
            "symmetric_division": -0.3,
            "volume_ratio": -0.2,  # More unequal
            "position_offset": 0.1,  # More offset
            "stability_bonus": 0.1
        }
        
        # Wnt signaling promotes symmetric division
        self.regulatory_effects["wnt_signaling"] = {
            "symmetric_division": 0.3,
            "volume_ratio": 0.2,  # More equal
            "position_offset": -0.1,  # More centered
            "stability_bonus": 0.1
        }
        
        # Cell density affects division plane positioning
        self.regulatory_effects["cell_density"] = {
            "symmetric_division": -0.1,
            "volume_ratio": -0.1,
            "position_offset": 0.05,
            "stability_bonus": 0.0
        }
        
        # Differentiation pressure promotes asymmetric division
        self.regulatory_effects["differentiation_pressure"] = {
            "symmetric_division": -0.4,
            "volume_ratio": -0.3,
            "position_offset": 0.2,
            "stability_bonus": 0.0
        }
    
    def calculate_division_plane(self, spindle_result: SpindleOrientationResult,
                               cell_polarity: CellPolarityState,
                               cell_geometry: Dict[str, float],
                               regulatory_factors: Dict[str, float]) -> DivisionPlaneResult:
        """
        Calculate division plane based on spindle orientation and context
        
        Args:
            spindle_result: Result from spindle orientation calculation
            cell_polarity: Current cell polarity state
            cell_geometry: Cell geometry parameters (volume, aspect_ratio, etc.)
            regulatory_factors: Current regulatory factor levels
            
        Returns:
            DivisionPlaneResult with division plane details
        """
        # Get base plane parameters
        base_params = self.plane_biases[spindle_result.orientation].copy()
        
        # Adjust based on regulatory factors
        adjusted_params = self._adjust_for_regulatory_factors(base_params, regulatory_factors)
        
        # Calculate division plane normal vector
        normal_vector = self._calculate_normal_vector(spindle_result, cell_polarity, adjusted_params)
        
        # Calculate division plane position
        position = self._calculate_plane_position(cell_geometry, adjusted_params, cell_polarity)
        
        # Calculate stability
        stability = self._calculate_plane_stability(spindle_result, adjusted_params, regulatory_factors)
        
        # Calculate daughter cell volumes
        volume_ratio = adjusted_params["volume_ratio"]
        daughter_volumes = self._calculate_daughter_volumes(volume_ratio, cell_geometry)
        
        return DivisionPlaneResult(
            normal_vector=normal_vector,
            position=position,
            stability=stability,
            regulatory_signals=regulatory_factors.copy(),
            daughter_cell_volumes=daughter_volumes
        )
    
    def _adjust_for_regulatory_factors(self, base_params: Dict[str, float],
                                     regulatory_factors: Dict[str, float]) -> Dict[str, float]:
        """Adjust division plane parameters based on regulatory factors"""
        adjusted = base_params.copy()
        
        for factor_name, factor_value in regulatory_factors.items():
            if factor_name in self.regulatory_effects:
                effects = self.regulatory_effects[factor_name]
                normalized_value = max(0.0, min(1.0, factor_value))
                
                for param_name, effect in effects.items():
                    if param_name in adjusted:
                        adjustment = effect * normalized_value * 0.5
                        adjusted[param_name] += adjustment
        
        # Clamp values to valid ranges
        adjusted["symmetric_division"] = max(0.0, min(1.0, adjusted["symmetric_division"]))
        adjusted["volume_ratio"] = max(0.1, min(0.9, adjusted["volume_ratio"]))
        adjusted["position_offset"] = max(-0.5, min(0.5, adjusted["position_offset"]))
        adjusted["stability_bonus"] = max(0.0, min(0.5, adjusted["stability_bonus"]))
        
        return adjusted
    
    def _calculate_normal_vector(self, spindle_result: SpindleOrientationResult,
                               cell_polarity: CellPolarityState,
                               params: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate division plane normal vector"""
        # Start with spindle orientation
        if spindle_result.orientation == SpindleOrientation.PARALLEL:
            # Perpendicular to apical-basal axis
            base_normal = (1.0, 0.0, 0.0)
        elif spindle_result.orientation == SpindleOrientation.PERPENDICULAR:
            # Parallel to apical-basal axis
            base_normal = (0.0, 1.0, 0.0)
        else:  # OBLIQUE
            # Random oblique orientation
            angle = np.random.uniform(0, 2 * np.pi)
            base_normal = (np.cos(angle), np.sin(angle), 0.0)
        
        # Adjust based on cell polarity
        if cell_polarity.polarity_type.value == "bipolar":
            # Use polarity axis as reference
            polarity_axis = cell_polarity.orientation_vector
            # Make normal perpendicular to polarity axis
            base_normal = self._perpendicular_vector(polarity_axis)
        
        # Add some randomness based on stability
        stability = spindle_result.stability
        noise_scale = (1.0 - stability) * 0.1
        
        noise = np.random.normal(0, noise_scale, 3)
        normal_vector = tuple(np.array(base_normal) + noise)
        
        # Normalize
        normal_magnitude = np.sqrt(sum(x**2 for x in normal_vector))
        if normal_magnitude > 0:
            normal_vector = tuple(x / normal_magnitude for x in normal_vector)
        
        return normal_vector
    
    def _perpendicular_vector(self, vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Find a vector perpendicular to the given vector"""
        x, y, z = vector
        
        # Find the component with the smallest absolute value
        if abs(x) <= abs(y) and abs(x) <= abs(z):
            # x is smallest, use (0, -z, y)
            return (0.0, -z, y)
        elif abs(y) <= abs(z):
            # y is smallest, use (-z, 0, x)
            return (-z, 0.0, x)
        else:
            # z is smallest, use (-y, x, 0)
            return (-y, x, 0.0)
    
    def _calculate_plane_position(self, cell_geometry: Dict[str, float],
                                params: Dict[str, float],
                                cell_polarity: CellPolarityState) -> Tuple[float, float, float]:
        """Calculate division plane position"""
        # Base position (cell center)
        base_position = (0.0, 0.0, 0.0)
        
        # Apply position offset
        offset = params["position_offset"]
        
        # Adjust based on cell polarity
        if cell_polarity.polarity_type.value == "apical":
            # Offset towards basal side
            position = (base_position[0], base_position[1] + offset, base_position[2])
        elif cell_polarity.polarity_type.value == "basal":
            # Offset towards apical side
            position = (base_position[0], base_position[1] - offset, base_position[2])
        else:
            # Random offset
            angle = np.random.uniform(0, 2 * np.pi)
            offset_x = np.cos(angle) * offset
            offset_y = np.sin(angle) * offset
            position = (base_position[0] + offset_x, base_position[1] + offset_y, base_position[2])
        
        return position
    
    def _calculate_plane_stability(self, spindle_result: SpindleOrientationResult,
                                 params: Dict[str, float],
                                 regulatory_factors: Dict[str, float]) -> float:
        """Calculate division plane stability"""
        base_stability = 0.7
        
        # Spindle stability contributes
        spindle_contribution = spindle_result.stability * 0.3
        
        # Regulatory factors contribute
        regulatory_contribution = 0.0
        for factor_name, factor_value in regulatory_factors.items():
            if factor_name in ["notch_activity", "wnt_signaling"]:
                regulatory_contribution += factor_value * 0.1
        
        # Stability bonus from parameters
        stability_bonus = params["stability_bonus"]
        
        total_stability = base_stability + spindle_contribution + regulatory_contribution + stability_bonus
        
        return max(0.0, min(1.0, total_stability))
    
    def _calculate_daughter_volumes(self, volume_ratio: float,
                                  cell_geometry: Dict[str, float]) -> Tuple[float, float]:
        """Calculate relative volumes of daughter cells"""
        total_volume = cell_geometry.get("volume", 1.0)
        
        # Volume ratio determines asymmetry
        # volume_ratio = 0.5 means equal volumes
        # volume_ratio < 0.5 means first daughter smaller
        # volume_ratio > 0.5 means first daughter larger
        
        daughter1_volume = total_volume * volume_ratio
        daughter2_volume = total_volume * (1.0 - volume_ratio)
        
        return (daughter1_volume, daughter2_volume)
    
    def update_division_plane_state(self, plane_state: DivisionPlaneState,
                                  time_delta: float,
                                  regulatory_factors: Dict[str, float]) -> DivisionPlaneState:
        """Update division plane state over time"""
        # Update stability based on regulatory factors
        stability_change = 0.0
        for factor_name, factor_value in regulatory_factors.items():
            if factor_name in ["notch_activity", "wnt_signaling"]:
                stability_change += factor_value * 0.01 * time_delta
        
        new_stability = max(0.0, min(1.0, plane_state.stability + stability_change))
        
        # Update regulatory signals
        new_regulatory_signals = regulatory_factors.copy()
        
        return DivisionPlaneState(
            normal_vector=plane_state.normal_vector,
            position=plane_state.position,
            stability=new_stability,
            regulatory_signals=new_regulatory_signals
        )
    
    def validate_division_plane(self, plane_result: DivisionPlaneResult,
                              cell_geometry: Dict[str, float]) -> bool:
        """Validate that the division plane is physically reasonable"""
        # Check that normal vector is valid
        normal_magnitude = np.sqrt(sum(x**2 for x in plane_result.normal_vector))
        if normal_magnitude < 0.1:
            return False
        
        # Check that daughter volumes are reasonable
        vol1, vol2 = plane_result.daughter_cell_volumes
        total_vol = vol1 + vol2
        if abs(total_vol - cell_geometry.get("volume", 1.0)) > 0.1:
            return False
        
        # Check that volumes are positive
        if vol1 <= 0 or vol2 <= 0:
            return False
        
        # Check that stability is reasonable
        if plane_result.stability < 0 or plane_result.stability > 1:
            return False
        
        return True
