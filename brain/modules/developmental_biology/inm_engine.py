"""Interkinetic Nuclear Migration Engine

Lightweight wrapper around the original INM system that integrates
phase-dependent velocities from inm_parameters.py.

Author: Quark AI
Date: 2025-01-30
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from .inm_parameters import INMParameters, CellCyclePhase


@dataclass
class INMState:
    """Simplified INM state for the engine"""
    cell_id: str
    current_position: float  # 0.0 (apical) to 1.0 (basal)
    target_position: float
    velocity: float
    phase: CellCyclePhase
    migration_active: bool


class INMEngine:
    """Engine for interkinetic nuclear migration with phase-dependent velocities"""
    
    def __init__(self):
        """Initialize INM engine"""
        self.parameters = INMParameters()
        self.cell_states: Dict[str, INMState] = {}
    
    def initialize_cell(self, cell_id: str, initial_position: float, 
                       phase: CellCyclePhase) -> INMState:
        """Initialize a cell's INM state"""
        target_pos = self.parameters.get_target_position(phase)
        velocity = self.parameters.get_velocity_for_phase(phase)
        
        state = INMState(
            cell_id=cell_id,
            current_position=initial_position,
            target_position=target_pos,
            velocity=velocity,
            phase=phase,
            migration_active=True
        )
        
        self.cell_states[cell_id] = state
        return state
    
    def update_cell_phase(self, cell_id: str, new_phase: CellCyclePhase) -> None:
        """Update cell cycle phase and adjust velocity/target"""
        if cell_id not in self.cell_states:
            return
        
        state = self.cell_states[cell_id]
        state.phase = new_phase
        state.target_position = self.parameters.get_target_position(new_phase)
        state.velocity = self.parameters.get_velocity_for_phase(new_phase)
        state.migration_active = True
    
    def update_positions(self, dt_hours: float) -> Dict[str, float]:
        """Update all cell positions for time step"""
        position_changes = {}
        
        for cell_id, state in self.cell_states.items():
            if not state.migration_active:
                continue
            
            # Calculate direction
            direction = 1.0 if state.target_position > state.current_position else -1.0
            
            # Calculate position change
            change = state.velocity * dt_hours * direction
            new_position = state.current_position + change
            
            # Clamp to valid range
            new_position = max(0.0, min(1.0, new_position))
            
            # Check if target reached
            if abs(new_position - state.target_position) < self.parameters.position_tolerance:
                new_position = state.target_position
                state.migration_active = False
            
            position_changes[cell_id] = new_position - state.current_position
            state.current_position = new_position
        
        return position_changes
    
    def get_phase_velocities(self) -> Dict[CellCyclePhase, float]:
        """Get current phase velocities for validation"""
        return {phase: self.parameters.get_velocity_for_phase(phase) 
                for phase in CellCyclePhase}
    
    def validate_velocity_ordering(self, tolerance: float = 0.15) -> bool:
        """Validate that phase velocities follow expected ordering"""
        velocities = self.get_phase_velocities()
        return self.parameters.validate_velocity_match(velocities, tolerance)
