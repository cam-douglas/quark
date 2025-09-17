"""
Meninges Constraint Manager

This module manages mechanical constraints from the meninges scaffold
and integrates with the foundation layer meninges system.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .cell_positioning_algorithms import CellPosition


class MeningesLayer(Enum):
    """Meninges layers from foundation layer"""
    DURA_MATER = "dura_mater"
    ARACHNOID = "arachnoid"
    PIA_MATER = "pia_mater"


class MechanicalConstraint(Enum):
    """Types of mechanical constraints"""
    BOUNDARY_CONSTRAINT = "boundary_constraint"
    ELASTIC_CONSTRAINT = "elastic_constraint"
    GROWTH_CONSTRAINT = "growth_constraint"
    PRESSURE_CONSTRAINT = "pressure_constraint"


@dataclass
class MeningesConstraint:
    """Meninges constraint information"""
    layer: MeningesLayer
    position: Tuple[float, float, float]
    stiffness: float
    thickness: float
    constraint_radius: float


class MeningesConstraintManager:
    """
    Manages mechanical constraints from the meninges scaffold
    and integrates with the foundation layer meninges system.
    """
    
    def __init__(self):
        """Initialize meninges constraint manager"""
        self.meninges_constraints: Dict[str, MeningesConstraint] = {}
        self._setup_meninges_constraints()
    
    def _setup_meninges_constraints(self) -> None:
        """Setup meninges constraints from foundation layer"""
        # This would integrate with the completed meninges scaffold system
        # For now, create representative constraints
        
        self.meninges_constraints = {
            "dura_mater_outer": MeningesConstraint(
                layer=MeningesLayer.DURA_MATER,
                position=(0.0, 0.0, 1.1),  # Outside neural tube
                stiffness=3.0,
                thickness=0.1,
                constraint_radius=5.0
            ),
            "arachnoid_middle": MeningesConstraint(
                layer=MeningesLayer.ARACHNOID,
                position=(0.0, 0.0, 1.05),
                stiffness=1.5,
                thickness=0.05,
                constraint_radius=4.5
            ),
            "pia_mater_inner": MeningesConstraint(
                layer=MeningesLayer.PIA_MATER,
                position=(0.0, 0.0, 1.0),  # At neural tube surface
                stiffness=1.0,
                thickness=0.02,
                constraint_radius=4.0
            )
        }
    
    def calculate_constraint_forces(self, cell_position: CellPosition) -> Dict[MechanicalConstraint, float]:
        """Calculate constraint forces from meninges layers"""
        constraint_forces = {}
        
        for constraint_id, constraint in self.meninges_constraints.items():
            # Calculate distance to constraint
            distance = self._calculate_distance_to_constraint(cell_position, constraint)
            
            if distance < constraint.constraint_radius:
                # Calculate constraint force
                force = self._calculate_constraint_force(cell_position, constraint, distance)
                
                # Determine constraint type
                if constraint.layer == MeningesLayer.DURA_MATER:
                    constraint_type = MechanicalConstraint.BOUNDARY_CONSTRAINT
                elif constraint.layer == MeningesLayer.ARACHNOID:
                    constraint_type = MechanicalConstraint.ELASTIC_CONSTRAINT
                else:  # PIA_MATER
                    constraint_type = MechanicalConstraint.GROWTH_CONSTRAINT
                
                constraint_forces[constraint_type] = force
        
        return constraint_forces
    
    def _calculate_distance_to_constraint(self, cell_position: CellPosition, 
                                        constraint: MeningesConstraint) -> float:
        """Calculate distance from cell to meninges constraint"""
        dx = cell_position.x - constraint.position[0]
        dy = cell_position.y - constraint.position[1]
        dz = cell_position.z - constraint.position[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _calculate_constraint_force(self, cell_position: CellPosition, 
                                  constraint: MeningesConstraint, distance: float) -> float:
        """Calculate force from meninges constraint"""
        if distance == 0:
            return constraint.stiffness * 10.0  # Strong force at contact
        
        # Force decreases with distance
        force = constraint.stiffness / (distance + 0.1)
        
        # Adjust based on constraint thickness
        thickness_factor = 1.0 + constraint.thickness * 2.0
        
        return force * thickness_factor
    
    def get_constraint_statistics(self, cell_positions: List[CellPosition]) -> Dict[str, float]:
        """Get constraint statistics for multiple cells"""
        if not cell_positions:
            return {}
        
        constraint_counts = {constraint: 0 for constraint in MechanicalConstraint}
        total_forces = {constraint: 0.0 for constraint in MechanicalConstraint}
        
        for cell_position in cell_positions:
            forces = self.calculate_constraint_forces(cell_position)
            for constraint_type, force in forces.items():
                constraint_counts[constraint_type] += 1
                total_forces[constraint_type] += force
        
        total_cells = len(cell_positions)
        
        return {
            "total_cells": total_cells,
            "constraint_counts": {constraint.value: count for constraint, count in constraint_counts.items()},
            "average_forces": {constraint.value: total_forces[constraint]/max(1, constraint_counts[constraint]) 
                             for constraint in MechanicalConstraint},
            "total_constraints": len(self.meninges_constraints)
        }
    
    def update_constraint_positions(self, time_delta: float) -> None:
        """Update constraint positions over time (for tissue growth)"""
        # This would integrate with tissue growth from foundation layer
        # For now, just maintain current positions
        pass
    
    def get_constraint_info(self, constraint_id: str) -> Optional[MeningesConstraint]:
        """Get information about specific constraint"""
        return self.meninges_constraints.get(constraint_id)
    
    def add_constraint(self, constraint_id: str, constraint: MeningesConstraint) -> None:
        """Add new meninges constraint"""
        self.meninges_constraints[constraint_id] = constraint
    
    def remove_constraint(self, constraint_id: str) -> None:
        """Remove meninges constraint"""
        if constraint_id in self.meninges_constraints:
            del self.meninges_constraints[constraint_id]
