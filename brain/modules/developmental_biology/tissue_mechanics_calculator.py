"""
Tissue Mechanics Calculator

This module calculates tissue mechanics including stress, pressure,
growth rates, and deformation for tissue mechanics integration.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .cell_positioning_algorithms import CellPosition
from .ventricular_zone_organizer import ZoneType
from .meninges_constraint_manager import MechanicalConstraint


@dataclass
class TissueMechanicsParameters:
    """Parameters for tissue mechanics"""
    tissue_stiffness: float
    growth_rate: float
    pressure_sensitivity: float
    boundary_stiffness: float
    elastic_modulus: float
    poisson_ratio: float


@dataclass
class TissueMechanicsState:
    """Tissue mechanics state information"""
    cell_id: str
    position: CellPosition
    mechanical_stress: float
    constraint_forces: Dict[MechanicalConstraint, float]
    tissue_pressure: float
    growth_rate: float
    deformation: Tuple[float, float, float]  # (dx, dy, dz)


class TissueMechanicsCalculator:
    """
    Calculates tissue mechanics including stress, pressure,
    growth rates, and deformation for tissue mechanics integration.
    """
    
    def __init__(self):
        """Initialize tissue mechanics calculator"""
        self.mechanics_parameters: Dict[str, TissueMechanicsParameters] = {}
        self._setup_mechanics_parameters()
    
    def _setup_mechanics_parameters(self) -> None:
        """Setup tissue mechanics parameters for different zones"""
        self.mechanics_parameters = {
            "ventricular_zone": TissueMechanicsParameters(
                tissue_stiffness=1.0,
                growth_rate=0.1,
                pressure_sensitivity=0.8,
                boundary_stiffness=2.0,
                elastic_modulus=1.5,
                poisson_ratio=0.3
            ),
            "subventricular_zone": TissueMechanicsParameters(
                tissue_stiffness=0.8,
                growth_rate=0.08,
                pressure_sensitivity=0.6,
                boundary_stiffness=1.5,
                elastic_modulus=1.2,
                poisson_ratio=0.3
            ),
            "intermediate_zone": TissueMechanicsParameters(
                tissue_stiffness=0.6,
                growth_rate=0.06,
                pressure_sensitivity=0.4,
                boundary_stiffness=1.0,
                elastic_modulus=0.8,
                poisson_ratio=0.3
            ),
            "mantle_zone": TissueMechanicsParameters(
                tissue_stiffness=0.4,
                growth_rate=0.04,
                pressure_sensitivity=0.2,
                boundary_stiffness=0.8,
                elastic_modulus=0.5,
                poisson_ratio=0.3
            )
        }
    
    def calculate_initial_stress(self, cell_position: CellPosition) -> float:
        """Calculate initial mechanical stress for cell"""
        zone_name = cell_position.zone_type.value
        params = self.mechanics_parameters.get(zone_name, 
                                             self.mechanics_parameters["ventricular_zone"])
        
        # Base stress from tissue stiffness
        base_stress = params.tissue_stiffness * 0.1
        
        # Adjust based on position in tissue
        position_factor = 1.0 + cell_position.z * 0.5  # Higher stress deeper in tissue
        
        # Adjust based on zone type
        zone_factors = {
            ZoneType.VENTRICULAR_ZONE: 1.0,
            ZoneType.SUBVENTRICULAR_ZONE: 0.8,
            ZoneType.INTERMEDIATE_ZONE: 0.6,
            ZoneType.MANTLE_ZONE: 0.4
        }
        
        zone_factor = zone_factors.get(cell_position.zone_type, 1.0)
        
        return base_stress * position_factor * zone_factor
    
    def calculate_tissue_pressure(self, cell_position: CellPosition) -> float:
        """Calculate tissue pressure at cell position"""
        zone_name = cell_position.zone_type.value
        params = self.mechanics_parameters.get(zone_name, 
                                             self.mechanics_parameters["ventricular_zone"])
        
        # Base pressure from tissue mechanics
        base_pressure = params.pressure_sensitivity * 0.5
        
        # Adjust based on position depth
        depth_factor = 1.0 + cell_position.z * 0.3
        
        # Adjust based on cell density (simplified)
        density_factor = 1.0 + cell_position.radius * 0.2
        
        return base_pressure * depth_factor * density_factor
    
    def calculate_growth_rate(self, cell_position: CellPosition) -> float:
        """Calculate tissue growth rate at cell position"""
        zone_name = cell_position.zone_type.value
        params = self.mechanics_parameters.get(zone_name, 
                                             self.mechanics_parameters["ventricular_zone"])
        
        # Base growth rate from parameters
        base_growth = params.growth_rate
        
        # Adjust based on mechanical stress
        stress_factor = 1.0 - cell_position.radius * 0.1  # Higher stress reduces growth
        
        # Adjust based on tissue pressure
        pressure_factor = 1.0 - params.pressure_sensitivity * 0.2
        
        return base_growth * stress_factor * pressure_factor
    
    def update_mechanical_stress(self, state: TissueMechanicsState, time_delta: float) -> float:
        """Update mechanical stress over time"""
        # Stress changes based on constraint forces
        total_constraint_force = sum(state.constraint_forces.values())
        
        # Stress relaxation
        stress_relaxation = 0.1 * time_delta
        
        # New stress
        new_stress = state.mechanical_stress + total_constraint_force * time_delta
        new_stress = max(0.0, new_stress - stress_relaxation)
        
        return new_stress
    
    def update_tissue_pressure(self, state: TissueMechanicsState, time_delta: float) -> float:
        """Update tissue pressure over time"""
        zone_name = state.position.zone_type.value
        params = self.mechanics_parameters.get(zone_name, 
                                             self.mechanics_parameters["ventricular_zone"])
        
        # Pressure changes based on growth and mechanical stress
        growth_pressure = state.growth_rate * time_delta
        stress_pressure = state.mechanical_stress * params.pressure_sensitivity * time_delta
        
        new_pressure = state.tissue_pressure + growth_pressure + stress_pressure
        
        return max(0.0, new_pressure)
    
    def update_growth_rate(self, state: TissueMechanicsState, time_delta: float) -> float:
        """Update growth rate over time"""
        zone_name = state.position.zone_type.value
        params = self.mechanics_parameters.get(zone_name, 
                                             self.mechanics_parameters["ventricular_zone"])
        
        # Growth rate changes based on mechanical stress and pressure
        stress_inhibition = state.mechanical_stress * 0.1
        pressure_inhibition = state.tissue_pressure * 0.05
        
        new_growth_rate = params.growth_rate - stress_inhibition - pressure_inhibition
        
        return max(0.0, new_growth_rate)
    
    def update_deformation(self, state: TissueMechanicsState, time_delta: float) -> Tuple[float, float, float]:
        """Update tissue deformation over time"""
        # Deformation based on constraint forces and mechanical stress
        deformation_rate = state.mechanical_stress * 0.01 * time_delta
        
        # Random deformation components
        dx = deformation_rate * (np.random.random() - 0.5)
        dy = deformation_rate * (np.random.random() - 0.5)
        dz = deformation_rate * (np.random.random() - 0.5)
        
        # Add to existing deformation
        new_deformation = (
            state.deformation[0] + dx,
            state.deformation[1] + dy,
            state.deformation[2] + dz
        )
        
        return new_deformation
    
    def apply_mechanical_constraints(self, position: CellPosition, 
                                   state: TissueMechanicsState) -> CellPosition:
        """Apply mechanical constraints to cell position"""
        # Calculate total constraint force
        total_force = sum(state.constraint_forces.values())
        
        if total_force == 0:
            return position
        
        # Apply force as position adjustment
        force_factor = total_force * 0.01  # Scale factor
        
        # Adjust position based on constraint forces
        new_x = position.x + state.deformation[0] * force_factor
        new_y = position.y + state.deformation[1] * force_factor
        new_z = position.z + state.deformation[2] * force_factor
        
        # Apply boundary constraints
        new_x = max(-10.0, min(10.0, new_x))
        new_y = max(-10.0, min(10.0, new_y))
        new_z = max(0.0, min(1.0, new_z))
        
        # Create new position
        constrained_position = CellPosition(
            cell_id=position.cell_id,
            x=new_x,
            y=new_y,
            z=new_z,
            radius=position.radius,
            zone_type=position.zone_type,
            velocity=position.velocity,
            force=position.force
        )
        
        return constrained_position
    
    def get_mechanics_statistics(self, states: List[TissueMechanicsState]) -> Dict[str, float]:
        """Get tissue mechanics statistics for multiple cells"""
        if not states:
            return {}
        
        stresses = [state.mechanical_stress for state in states]
        pressures = [state.tissue_pressure for state in states]
        growth_rates = [state.growth_rate for state in states]
        constraint_forces = [sum(state.constraint_forces.values()) for state in states]
        
        return {
            "total_cells": len(states),
            "average_stress": np.mean(stresses),
            "average_pressure": np.mean(pressures),
            "average_growth_rate": np.mean(growth_rates),
            "average_constraint_force": np.mean(constraint_forces),
            "max_stress": np.max(stresses),
            "max_pressure": np.max(pressures),
            "max_growth_rate": np.max(growth_rates)
        }
