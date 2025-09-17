"""
Cell Adhesion Manager

Manages cell-cell adhesion and repulsion forces for tissue architecture.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .cell_adhesion_types import (
    AdhesionType, RepulsionType, AdhesionParameters, RepulsionParameters,
    AdhesionForce, RepulsionForce
)
from .cell_force_calculator import CellForceCalculator
from .neuroepithelial_cells import NeuroepithelialCell
from .ventricular_zone_organizer import ZoneType


class CellAdhesionManager:
    """
    Manages cell-cell adhesion and repulsion forces for tissue architecture
    """
    
    def __init__(self):
        """Initialize cell adhesion manager"""
        self.adhesion_parameters: Dict[str, AdhesionParameters] = {}
        self.repulsion_parameters: Dict[str, RepulsionParameters] = {}
        self.force_calculator: Optional[CellForceCalculator] = None
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup adhesion and repulsion parameters"""
        # Setup adhesion parameters
        self.adhesion_parameters = {
            "ventricular_zone": AdhesionParameters(
                adhesion_strength=1.0,
                adhesion_range=0.5,
                adhesion_threshold=0.3,
                zone_specific_strength={
                    ZoneType.VENTRICULAR_ZONE: 1.0,
                    ZoneType.SUBVENTRICULAR_ZONE: 0.8,
                    ZoneType.INTERMEDIATE_ZONE: 0.6,
                    ZoneType.MANTLE_ZONE: 0.4
                }
            ),
            "subventricular_zone": AdhesionParameters(
                adhesion_strength=0.8,
                adhesion_range=0.4,
                adhesion_threshold=0.25,
                zone_specific_strength={
                    ZoneType.VENTRICULAR_ZONE: 0.8,
                    ZoneType.SUBVENTRICULAR_ZONE: 1.0,
                    ZoneType.INTERMEDIATE_ZONE: 0.7,
                    ZoneType.MANTLE_ZONE: 0.5
                }
            )
        }
        
        # Setup repulsion parameters
        self.repulsion_parameters = {
            "ventricular_zone": RepulsionParameters(
                repulsion_strength=2.0,
                repulsion_range=0.2,
                repulsion_threshold=0.1,
                zone_specific_strength={
                    ZoneType.VENTRICULAR_ZONE: 1.0,
                    ZoneType.SUBVENTRICULAR_ZONE: 0.9,
                    ZoneType.INTERMEDIATE_ZONE: 0.8,
                    ZoneType.MANTLE_ZONE: 0.7
                }
            )
        }
        
        # Initialize force calculator
        default_adhesion = self.adhesion_parameters["ventricular_zone"]
        default_repulsion = self.repulsion_parameters["ventricular_zone"]
        self.force_calculator = CellForceCalculator(default_adhesion, default_repulsion)
    
    def manage_cell_adhesion(self,
                           cells: Dict[str, NeuroepithelialCell],
                           target_cell_id: str) -> List[AdhesionForce]:
        """
        Manage adhesion forces for a target cell
        
        Args:
            cells: Dictionary of all cells
            target_cell_id: ID of target cell
        
        Returns:
            List of adhesion forces acting on target cell
        """
        if target_cell_id not in cells or self.force_calculator is None:
            return []
        
        target_cell = cells[target_cell_id]
        neighboring_cells = [cell for cell_id, cell in cells.items() if cell_id != target_cell_id]
        
        adhesion_forces = []
        
        for neighbor in neighboring_cells:
            distance = np.linalg.norm(np.array(neighbor.position) - np.array(target_cell.position))
            
            if distance <= self.adhesion_parameters["ventricular_zone"].adhesion_range:
                force = self.force_calculator.calculate_adhesion_force(target_cell, neighbor, distance)
                
                adhesion_forces.append(AdhesionForce(
                    cell_id=target_cell_id,
                    target_cell_id=neighbor.cell_id,
                    force_magnitude=force.force_magnitude,
                    force_direction=force.force_direction,
                    adhesion_type=force.adhesion_type
                ))
        
        return adhesion_forces
    
    def manage_cell_repulsion(self,
                            cells: Dict[str, NeuroepithelialCell],
                            target_cell_id: str) -> List[RepulsionForce]:
        """
        Manage repulsion forces for a target cell
        
        Args:
            cells: Dictionary of all cells
            target_cell_id: ID of target cell
        
        Returns:
            List of repulsion forces acting on target cell
        """
        if target_cell_id not in cells or self.force_calculator is None:
            return []
        
        target_cell = cells[target_cell_id]
        neighboring_cells = [cell for cell_id, cell in cells.items() if cell_id != target_cell_id]
        
        repulsion_forces = []
        
        for neighbor in neighboring_cells:
            distance = np.linalg.norm(np.array(neighbor.position) - np.array(target_cell.position))
            
            if distance <= self.repulsion_parameters["ventricular_zone"].repulsion_range:
                force = self.force_calculator.calculate_repulsion_force(target_cell, neighbor, distance)
                
                repulsion_forces.append(RepulsionForce(
                    cell_id=target_cell_id,
                    target_cell_id=neighbor.cell_id,
                    force_magnitude=force.force_magnitude,
                    force_direction=force.force_direction,
                    repulsion_type=force.repulsion_type
                ))
        
        return repulsion_forces
    
    def calculate_net_cell_force(self,
                                cells: Dict[str, NeuroepithelialCell],
                                target_cell_id: str) -> Tuple[float, float, float]:
        """
        Calculate net force on target cell from adhesion and repulsion
        
        Args:
            cells: Dictionary of all cells
            target_cell_id: ID of target cell
        
        Returns:
            Net force vector (x, y, z)
        """
        if self.force_calculator is None:
            return (0.0, 0.0, 0.0)
        
        target_cell = cells[target_cell_id]
        neighboring_cells = [cell for cell_id, cell in cells.items() if cell_id != target_cell_id]
        
        return self.force_calculator.calculate_net_force(target_cell, neighboring_cells)
    
    def update_adhesion_parameters(self,
                                 zone_type: str,
                                 parameters: AdhesionParameters) -> None:
        """Update adhesion parameters for a zone"""
        self.adhesion_parameters[zone_type] = parameters
        
        # Update force calculator if this is the active zone
        if zone_type == "ventricular_zone" and self.force_calculator is not None:
            repulsion_params = self.repulsion_parameters["ventricular_zone"]
            self.force_calculator = CellForceCalculator(parameters, repulsion_params)
    
    def update_repulsion_parameters(self,
                                  zone_type: str,
                                  parameters: RepulsionParameters) -> None:
        """Update repulsion parameters for a zone"""
        self.repulsion_parameters[zone_type] = parameters
        
        # Update force calculator if this is the active zone
        if zone_type == "ventricular_zone" and self.force_calculator is not None:
            adhesion_params = self.adhesion_parameters["ventricular_zone"]
            self.force_calculator = CellForceCalculator(adhesion_params, parameters)
    
    def get_adhesion_statistics(self) -> Dict[str, float]:
        """Get adhesion force statistics"""
        return {
            'total_zones': len(self.adhesion_parameters),
            'average_adhesion_strength': np.mean([
                params.adhesion_strength for params in self.adhesion_parameters.values()
            ]),
            'average_adhesion_range': np.mean([
                params.adhesion_range for params in self.adhesion_parameters.values()
            ])
        }
    
    def get_repulsion_statistics(self) -> Dict[str, float]:
        """Get repulsion force statistics"""
        return {
            'total_zones': len(self.repulsion_parameters),
            'average_repulsion_strength': np.mean([
                params.repulsion_strength for params in self.repulsion_parameters.values()
            ]),
            'average_repulsion_range': np.mean([
                params.repulsion_range for params in self.repulsion_parameters.values()
            ])
        }