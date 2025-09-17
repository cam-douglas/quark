"""
Morphogen Exposure Calculator

This module calculates morphogen exposure for cells based on their position
and integrates with the foundation layer morphogen gradient system.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType, SpatialPosition, CellArchitecture
from .gene_expression_calculator import MorphogenType


@dataclass
class MorphogenGradient:
    """Morphogen gradient information"""
    morphogen_type: MorphogenType
    concentration: float  # 0.0 to 1.0
    gradient_direction: Tuple[float, float, float]
    gradient_strength: float
    spatial_position: Tuple[float, float, float]


class MorphogenExposureCalculator:
    """
    Calculates morphogen exposure for cells based on their position
    and integrates with the foundation layer morphogen gradient system.
    """
    
    def __init__(self):
        """Initialize morphogen exposure calculator"""
        self.morphogen_gradients: Dict[str, MorphogenGradient] = {}
        self._initialize_morphogen_gradients()
    
    def _initialize_morphogen_gradients(self) -> None:
        """Initialize morphogen gradients from foundation layer"""
        # This would integrate with the completed morphogen gradient system
        # For now, create representative gradients
        
        gradients = [
            ("shh_gradient", MorphogenType.SHH, (0.0, 0.0, 1.0), 0.8),
            ("bmp_gradient", MorphogenType.BMP, (0.0, 0.0, -1.0), 0.6),
            ("wnt_gradient", MorphogenType.WNT, (1.0, 0.0, 0.0), 0.7),
            ("fgf_gradient", MorphogenType.FGF, (0.0, 1.0, 0.0), 0.5)
        ]
        
        for gradient_id, morphogen_type, direction, strength in gradients:
            gradient = MorphogenGradient(
                morphogen_type=morphogen_type,
                concentration=0.5,  # Base concentration
                gradient_direction=direction,
                gradient_strength=strength,
                spatial_position=(0.0, 0.0, 0.0)
            )
            
            self.morphogen_gradients[gradient_id] = gradient
    
    def calculate_morphogen_exposure(self, cell_architecture: CellArchitecture) -> Dict[MorphogenType, float]:
        """
        Calculate morphogen exposure for cell
        
        Args:
            cell_architecture: Cell architecture information
            
        Returns:
            Dictionary mapping morphogen types to exposure levels
        """
        morphogen_exposure = {}
        
        for gradient_id, gradient in self.morphogen_gradients.items():
            # Calculate distance-based exposure
            distance = self._calculate_distance_to_gradient(
                cell_architecture.position, gradient
            )
            
            # Calculate concentration at cell position
            concentration = self._calculate_morphogen_concentration(
                cell_architecture.position, gradient
            )
            
            # Calculate exposure based on cell zone and polarity
            exposure = self._calculate_cell_exposure(
                concentration, cell_architecture, gradient.morphogen_type
            )
            
            morphogen_exposure[gradient.morphogen_type] = exposure
        
        return morphogen_exposure
    
    def _calculate_distance_to_gradient(self, position: SpatialPosition,
                                      gradient: MorphogenGradient) -> float:
        """Calculate distance from cell to morphogen gradient"""
        dx = position.x - gradient.spatial_position[0]
        dy = position.y - gradient.spatial_position[1]
        dz = position.z - gradient.spatial_position[2]
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _calculate_morphogen_concentration(self, position: SpatialPosition,
                                         gradient: MorphogenGradient) -> float:
        """Calculate morphogen concentration at cell position"""
        # Simplified concentration calculation
        # In reality, this would integrate with the spatial grid system
        
        # Base concentration
        base_concentration = gradient.concentration
        
        # Distance-based decay
        distance = self._calculate_distance_to_gradient(position, gradient)
        decay_factor = np.exp(-distance / 5.0)  # Decay over 5 units
        
        # Gradient direction influence
        direction_factor = self._calculate_direction_influence(position, gradient)
        
        concentration = base_concentration * decay_factor * direction_factor
        
        return min(1.0, max(0.0, concentration))
    
    def _calculate_direction_influence(self, position: SpatialPosition,
                                     gradient: MorphogenGradient) -> float:
        """Calculate influence of gradient direction on concentration"""
        # Dot product of position vector with gradient direction
        position_vector = np.array([position.x, position.y, position.z])
        gradient_direction = np.array(gradient.gradient_direction)
        
        dot_product = np.dot(position_vector, gradient_direction)
        
        # Normalize to 0-1 range
        direction_influence = (dot_product + 1.0) / 2.0
        
        return direction_influence
    
    def _calculate_cell_exposure(self, concentration: float,
                               cell_architecture: CellArchitecture,
                               morphogen_type: MorphogenType) -> float:
        """Calculate cell exposure to morphogen"""
        # Base exposure from concentration
        base_exposure = concentration
        
        # Zone-specific sensitivity
        zone_sensitivity = self._get_zone_sensitivity(
            cell_architecture.position.zone_type, morphogen_type
        )
        
        # Polarity-specific exposure
        polarity_factor = self._get_polarity_factor(
            cell_architecture.polarity_state, morphogen_type
        )
        
        # Surface area influence
        surface_factor = self._get_surface_factor(cell_architecture)
        
        exposure = base_exposure * zone_sensitivity * polarity_factor * surface_factor
        
        return min(1.0, exposure)
    
    def _get_zone_sensitivity(self, zone_type: ZoneType, morphogen_type: MorphogenType) -> float:
        """Get zone-specific sensitivity to morphogen"""
        sensitivities = {
            ZoneType.VENTRICULAR_ZONE: {
                MorphogenType.SHH: 1.0,
                MorphogenType.BMP: 0.8,
                MorphogenType.WNT: 0.9,
                MorphogenType.FGF: 1.2
            },
            ZoneType.SUBVENTRICULAR_ZONE: {
                MorphogenType.SHH: 0.8,
                MorphogenType.BMP: 1.0,
                MorphogenType.WNT: 1.1,
                MorphogenType.FGF: 1.0
            },
            ZoneType.INTERMEDIATE_ZONE: {
                MorphogenType.SHH: 0.6,
                MorphogenType.BMP: 1.2,
                MorphogenType.WNT: 0.8,
                MorphogenType.FGF: 0.8
            },
            ZoneType.MANTLE_ZONE: {
                MorphogenType.SHH: 0.4,
                MorphogenType.BMP: 1.4,
                MorphogenType.WNT: 0.6,
                MorphogenType.FGF: 0.6
            }
        }
        
        zone_sens = sensitivities.get(zone_type, sensitivities[ZoneType.VENTRICULAR_ZONE])
        return zone_sens.get(morphogen_type, 1.0)
    
    def _get_polarity_factor(self, polarity_state, morphogen_type: MorphogenType) -> float:
        """Get polarity-specific factor for morphogen exposure"""
        # Apical cells are more sensitive to some morphogens
        if polarity_state.value == "apical":
            if morphogen_type in [MorphogenType.SHH, MorphogenType.FGF]:
                return 1.2
            else:
                return 0.8
        elif polarity_state.value == "basal":
            if morphogen_type in [MorphogenType.BMP, MorphogenType.WNT]:
                return 1.2
            else:
                return 0.8
        else:
            return 1.0
    
    def _get_surface_factor(self, cell_architecture: CellArchitecture) -> float:
        """Get surface area factor for morphogen exposure"""
        total_surface = cell_architecture.apical_surface_area + cell_architecture.basal_surface_area
        return min(1.5, total_surface / 2.0)  # Normalize to typical surface area
    
    def get_morphogen_exposure_statistics(self, morphogen_exposures: List[Dict[MorphogenType, float]]) -> Dict[str, float]:
        """Get statistics about morphogen exposure patterns"""
        if not morphogen_exposures:
            return {}
        
        # Calculate average exposure for each morphogen
        morphogen_averages = {}
        for morphogen_type in MorphogenType:
            exposures = [exposure.get(morphogen_type, 0.0) for exposure in morphogen_exposures]
            morphogen_averages[morphogen_type.value] = np.mean(exposures)
        
        # Calculate exposure variability
        morphogen_stds = {}
        for morphogen_type in MorphogenType:
            exposures = [exposure.get(morphogen_type, 0.0) for exposure in morphogen_exposures]
            morphogen_stds[morphogen_type.value] = np.std(exposures)
        
        return {
            "total_cells": len(morphogen_exposures),
            "morphogen_averages": morphogen_averages,
            "morphogen_std": morphogen_stds,
            "total_morphogens": len(MorphogenType)
        }
