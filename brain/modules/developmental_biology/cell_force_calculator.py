"""
Cell Force Calculator

Calculates adhesion and repulsion forces between cells.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Tuple

from .cell_adhesion_types import (
    AdhesionType, RepulsionType, AdhesionParameters, RepulsionParameters,
    AdhesionForce, RepulsionForce
)
from .neuroepithelial_cells import NeuroepithelialCell
from .ventricular_zone_organizer import ZoneType


class CellForceCalculator:
    """Calculates forces between cells for tissue mechanics"""
    
    def __init__(self, adhesion_params: AdhesionParameters, repulsion_params: RepulsionParameters):
        """Initialize force calculator"""
        self.adhesion_params = adhesion_params
        self.repulsion_params = repulsion_params
    
    def calculate_adhesion_force(self,
                                cell1: NeuroepithelialCell,
                                cell2: NeuroepithelialCell,
                                distance: float) -> AdhesionForce:
        """Calculate adhesion force between two cells"""
        if distance > self.adhesion_params.adhesion_range:
            return AdhesionForce(0.0, (0.0, 0.0, 0.0), AdhesionType.CADHERIN_ADHESION, 0.0)
        
        # Get zone-specific strength
        zone1 = self._get_cell_zone(cell1)
        zone2 = self._get_cell_zone(cell2)
        
        strength1 = self.adhesion_params.zone_specific_strength.get(zone1, 1.0)
        strength2 = self.adhesion_params.zone_specific_strength.get(zone2, 1.0)
        zone_strength = (strength1 + strength2) / 2.0
        
        # Calculate force magnitude (inverse distance relationship)
        base_strength = self.adhesion_params.adhesion_strength * zone_strength
        force_magnitude = base_strength * (1.0 / (distance + 0.1))  # Avoid division by zero
        
        # Force direction (from cell1 toward cell2)
        direction_vector = np.array(cell2.position) - np.array(cell1.position)
        direction_norm = np.linalg.norm(direction_vector)
        
        if direction_norm > 0:
            force_direction = tuple(direction_vector / direction_norm)
        else:
            force_direction = (0.0, 0.0, 0.0)
        
        return AdhesionForce(
            force_magnitude=force_magnitude,
            force_direction=force_direction,
            adhesion_type=AdhesionType.CADHERIN_ADHESION,
            interaction_range=distance
        )
    
    def calculate_repulsion_force(self,
                                 cell1: NeuroepithelialCell,
                                 cell2: NeuroepithelialCell,
                                 distance: float) -> RepulsionForce:
        """Calculate repulsion force between two cells"""
        if distance > self.repulsion_params.repulsion_range:
            return RepulsionForce(0.0, (0.0, 0.0, 0.0), RepulsionType.STERIC_REPULSION, 0.0)
        
        # Get zone-specific strength
        zone1 = self._get_cell_zone(cell1)
        zone2 = self._get_cell_zone(cell2)
        
        strength1 = self.repulsion_params.zone_specific_strength.get(zone1, 1.0)
        strength2 = self.repulsion_params.zone_specific_strength.get(zone2, 1.0)
        zone_strength = (strength1 + strength2) / 2.0
        
        # Calculate force magnitude (strong at close distance)
        base_strength = self.repulsion_params.repulsion_strength * zone_strength
        force_magnitude = base_strength * (1.0 / (distance**2 + 0.01))  # Inverse square law
        
        # Force direction (from cell2 away from cell1)
        direction_vector = np.array(cell1.position) - np.array(cell2.position)
        direction_norm = np.linalg.norm(direction_vector)
        
        if direction_norm > 0:
            force_direction = tuple(direction_vector / direction_norm)
        else:
            force_direction = (0.0, 0.0, 0.0)
        
        return RepulsionForce(
            force_magnitude=force_magnitude,
            force_direction=force_direction,
            repulsion_type=RepulsionType.STERIC_REPULSION,
            interaction_range=distance
        )
    
    def calculate_net_force(self,
                           target_cell: NeuroepithelialCell,
                           neighboring_cells: List[NeuroepithelialCell]) -> Tuple[float, float, float]:
        """Calculate net force on target cell from all neighbors"""
        net_force = np.array([0.0, 0.0, 0.0])
        
        for neighbor in neighboring_cells:
            if neighbor.cell_id == target_cell.cell_id:
                continue
            
            # Calculate distance
            distance = np.linalg.norm(np.array(neighbor.position) - np.array(target_cell.position))
            
            # Calculate adhesion force
            adhesion_force = self.calculate_adhesion_force(target_cell, neighbor, distance)
            adhesion_vector = np.array(adhesion_force.force_direction) * adhesion_force.force_magnitude
            
            # Calculate repulsion force
            repulsion_force = self.calculate_repulsion_force(target_cell, neighbor, distance)
            repulsion_vector = np.array(repulsion_force.force_direction) * repulsion_force.force_magnitude
            
            # Add to net force
            net_force += adhesion_vector - repulsion_vector  # Adhesion attracts, repulsion repels
        
        return tuple(net_force)
    
    def _get_cell_zone(self, cell: NeuroepithelialCell) -> ZoneType:
        """Determine cell's zone type based on position"""
        # Simple zone assignment based on z-coordinate (apical-basal axis)
        z_pos = cell.position[2]
        
        if z_pos < 0.3:
            return ZoneType.VENTRICULAR_ZONE
        elif z_pos < 0.7:
            return ZoneType.INTERMEDIATE_ZONE
        else:
            return ZoneType.MARGINAL_ZONE
