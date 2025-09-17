"""
Cell Positioning Algorithms

This module coordinates cell packing and spacing algorithms, models cell-cell
adhesion and repulsion, and maintains tissue architecture integrity.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType
from .cell_packing_algorithms import CellPackingAlgorithms, PackingAlgorithm
from .cell_adhesion_manager import CellAdhesionManager, AdhesionForce, RepulsionForce


class InteractionType(Enum):
    """Types of cell interactions"""
    ADHESION = "adhesion"
    REPULSION = "repulsion"
    NEUTRAL = "neutral"


@dataclass
class CellPosition:
    """Cell position information"""
    cell_id: str
    x: float
    y: float
    z: float
    radius: float
    zone_type: ZoneType
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    force: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CellInteraction:
    """Cell interaction information"""
    cell_id: str
    target_cell_id: str
    interaction_type: InteractionType
    force_magnitude: float
    force_direction: Tuple[float, float, float]
    distance: float


class CellPositioningAlgorithms:
    """
    Coordinates cell packing and spacing algorithms, models cell-cell
    adhesion and repulsion, and maintains tissue architecture integrity.
    """
    
    def __init__(self):
        """Initialize cell positioning algorithms"""
        self.packing_algorithms = CellPackingAlgorithms()
        self.adhesion_manager = CellAdhesionManager()
    
    def position_cells(self, cell_positions: Dict[str, CellPosition], 
                      zone_type: ZoneType, algorithm: PackingAlgorithm) -> Dict[str, CellPosition]:
        """
        Position cells using specified packing algorithm
        
        Args:
            cell_positions: Dictionary of cell positions
            zone_type: Zone type for positioning
            algorithm: Packing algorithm to use
            
        Returns:
            Dictionary of positioned cells
        """
        # Convert to list format for packing algorithms
        position_list = [(pos.x, pos.y, pos.z) for pos in cell_positions.values()]
        
        # Apply packing algorithm
        packed_positions = self.packing_algorithms.optimize_packing(position_list, zone_type, algorithm)
        
        # Update cell positions
        updated_positions = {}
        cell_ids = list(cell_positions.keys())
        
        for i, (new_x, new_y, new_z) in enumerate(packed_positions):
            if i < len(cell_ids):
                cell_id = cell_ids[i]
                original_pos = cell_positions[cell_id]
                
                updated_pos = CellPosition(
                    cell_id=cell_id,
                    x=new_x,
                    y=new_y,
                    z=new_z,
                    radius=original_pos.radius,
                    zone_type=original_pos.zone_type,
                    velocity=original_pos.velocity,
                    force=original_pos.force
                )
                
                updated_positions[cell_id] = updated_pos
        
        return updated_positions
    
    def calculate_cell_interactions(self, cell_positions: Dict[str, CellPosition], 
                                  zone_type: ZoneType) -> List[CellInteraction]:
        """
        Calculate cell-cell interactions (adhesion and repulsion)
        
        Args:
            cell_positions: Dictionary of cell positions
            zone_type: Zone type for interactions
            
        Returns:
            List of cell interactions
        """
        # Convert to position format for adhesion manager
        position_dict = {cell_id: (pos.x, pos.y, pos.z) 
                        for cell_id, pos in cell_positions.items()}
        
        # Calculate adhesion forces
        adhesion_forces = self.adhesion_manager.calculate_adhesion_forces(position_dict, zone_type)
        
        # Calculate repulsion forces
        repulsion_forces = self.adhesion_manager.calculate_repulsion_forces(position_dict, zone_type)
        
        # Convert to interactions
        interactions = []
        
        # Add adhesion interactions
        for force in adhesion_forces:
            interaction = CellInteraction(
                cell_id=force.cell_id,
                target_cell_id=force.target_cell_id,
                interaction_type=InteractionType.ADHESION,
                force_magnitude=force.force_magnitude,
                force_direction=force.force_direction,
                distance=self._calculate_distance(
                    position_dict[force.cell_id], 
                    position_dict[force.target_cell_id]
                )
            )
            interactions.append(interaction)
        
        # Add repulsion interactions
        for force in repulsion_forces:
            interaction = CellInteraction(
                cell_id=force.cell_id,
                target_cell_id=force.target_cell_id,
                interaction_type=InteractionType.REPULSION,
                force_magnitude=force.force_magnitude,
                force_direction=force.force_direction,
                distance=self._calculate_distance(
                    position_dict[force.cell_id], 
                    position_dict[force.target_cell_id]
                )
            )
            interactions.append(interaction)
        
        return interactions
    
    def apply_interaction_forces(self, cell_positions: Dict[str, CellPosition], 
                               interactions: List[CellInteraction]) -> Dict[str, CellPosition]:
        """
        Apply interaction forces to cell positions
        
        Args:
            cell_positions: Dictionary of cell positions
            interactions: List of cell interactions
            
        Returns:
            Dictionary of updated cell positions
        """
        updated_positions = {}
        
        for cell_id, position in cell_positions.items():
            # Calculate net force from interactions
            net_force = self._calculate_net_force(cell_id, interactions)
            
            # Update position based on force
            new_position = self._update_position_from_force(position, net_force)
            updated_positions[cell_id] = new_position
        
        return updated_positions
    
    def maintain_tissue_architecture(self, cell_positions: Dict[str, CellPosition], 
                                   zone_type: ZoneType) -> Dict[str, CellPosition]:
        """
        Maintain tissue architecture integrity
        
        Args:
            cell_positions: Dictionary of cell positions
            zone_type: Zone type for architecture
            
        Returns:
            Dictionary of cells with maintained architecture
        """
        # Calculate interactions
        interactions = self.calculate_cell_interactions(cell_positions, zone_type)
        
        # Apply forces
        updated_positions = self.apply_interaction_forces(cell_positions, interactions)
        
        # Apply packing constraints
        final_positions = self.position_cells(updated_positions, zone_type, PackingAlgorithm.DENSITY_GRADIENT_PACKING)
        
        return final_positions
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate distance between two positions"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _calculate_net_force(self, cell_id: str, interactions: List[CellInteraction]) -> Tuple[float, float, float]:
        """Calculate net force on cell from interactions"""
        net_fx = 0.0
        net_fy = 0.0
        net_fz = 0.0
        
        for interaction in interactions:
            if interaction.cell_id == cell_id:
                # Add force contribution
                fx = interaction.force_magnitude * interaction.force_direction[0]
                fy = interaction.force_magnitude * interaction.force_direction[1]
                fz = interaction.force_magnitude * interaction.force_direction[2]
                
                net_fx += fx
                net_fy += fy
                net_fz += fz
        
        return (net_fx, net_fy, net_fz)
    
    def _update_position_from_force(self, position: CellPosition, 
                                  net_force: Tuple[float, float, float]) -> CellPosition:
        """Update cell position based on net force"""
        # Simple force-based position update
        force_scale = 0.01  # Scale factor for force application
        
        new_x = position.x + net_force[0] * force_scale
        new_y = position.y + net_force[1] * force_scale
        new_z = position.z + net_force[2] * force_scale
        
        # Apply boundary constraints
        new_x = max(-10.0, min(10.0, new_x))
        new_y = max(-10.0, min(10.0, new_y))
        new_z = max(0.0, min(1.0, new_z))
        
        # Update velocity based on force
        new_velocity = (
            position.velocity[0] + net_force[0] * 0.1,
            position.velocity[1] + net_force[1] * 0.1,
            position.velocity[2] + net_force[2] * 0.1
        )
        
        return CellPosition(
            cell_id=position.cell_id,
            x=new_x,
            y=new_y,
            z=new_z,
            radius=position.radius,
            zone_type=position.zone_type,
            velocity=new_velocity,
            force=net_force
        )
    
    def get_positioning_statistics(self, cell_positions: Dict[str, CellPosition], 
                                 zone_type: ZoneType) -> Dict[str, float]:
        """Get positioning statistics for cells"""
        if not cell_positions:
            return {}
        
        # Get packing statistics
        position_list = [(pos.x, pos.y, pos.z) for pos in cell_positions.values()]
        packing_stats = self.packing_algorithms.get_packing_statistics(position_list, zone_type)
        
        # Get interaction statistics
        interactions = self.calculate_cell_interactions(cell_positions, zone_type)
        adhesion_interactions = [i for i in interactions if i.interaction_type == InteractionType.ADHESION]
        repulsion_interactions = [i for i in interactions if i.interaction_type == InteractionType.REPULSION]
        
        return {
            "total_cells": len(cell_positions),
            "packing_efficiency": packing_stats.get("packing_efficiency", 0.0),
            "current_density": packing_stats.get("current_density", 0.0),
            "target_density": packing_stats.get("target_density", 0.0),
            "total_adhesion_interactions": len(adhesion_interactions),
            "total_repulsion_interactions": len(repulsion_interactions),
            "average_adhesion_force": np.mean([i.force_magnitude for i in adhesion_interactions]) if adhesion_interactions else 0.0,
            "average_repulsion_force": np.mean([i.force_magnitude for i in repulsion_interactions]) if repulsion_interactions else 0.0
        }