"""
Interkinetic Nuclear Migration

This module models nuclear movement during cell cycle with apical-basal
nuclear shuttling and cell cycle timing control.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType, SpatialPosition, PolarityState


class CellCyclePhase(Enum):
    """Cell cycle phases"""
    G1 = "G1"
    S = "S"
    G2 = "G2"
    M = "M"


class NuclearPosition(Enum):
    """Nuclear position states"""
    APICAL = "apical"
    BASAL = "basal"
    MIGRATING_APICAL = "migrating_apical"
    MIGRATING_BASAL = "migrating_basal"
    STATIONARY = "stationary"


@dataclass
class NuclearMigrationState:
    """Nuclear migration state information"""
    cell_id: str
    current_position: float  # 0.0 (apical) to 1.0 (basal)
    target_position: float
    migration_velocity: float  # Position units per hour
    migration_direction: str  # "apical" or "basal"
    cell_cycle_phase: CellCyclePhase
    migration_active: bool
    time_in_phase: float  # Hours


@dataclass
class MigrationParameters:
    """Parameters for nuclear migration"""
    g1_apical_position: float  # 0.0 to 1.0
    s_basal_position: float  # 0.0 to 1.0
    g2_apical_position: float  # 0.0 to 1.0
    m_apical_position: float  # 0.0 to 1.0
    migration_velocity: float  # Position units per hour
    position_tolerance: float  # Tolerance for position matching


class InterkineticNuclearMigration:
    """
    Models nuclear movement during cell cycle with apical-basal
    nuclear shuttling and cell cycle timing control.
    """
    
    def __init__(self):
        """Initialize interkinetic nuclear migration system"""
        self.migration_parameters: Dict[str, MigrationParameters] = {}
        self.nuclear_states: Dict[str, NuclearMigrationState] = {}
        self._setup_migration_parameters()
    
    def _setup_migration_parameters(self) -> None:
        """Setup migration parameters for different developmental stages"""
        self.migration_parameters = {
            "early_embryonic": MigrationParameters(
                g1_apical_position=0.1,  # Near apical surface
                s_basal_position=0.8,   # Near basal surface
                g2_apical_position=0.2,  # Moving back to apical
                m_apical_position=0.05,  # At apical surface for division
                migration_velocity=0.3,  # Fast migration
                position_tolerance=0.05
            ),
            "mid_embryonic": MigrationParameters(
                g1_apical_position=0.15,
                s_basal_position=0.75,
                g2_apical_position=0.25,
                m_apical_position=0.1,
                migration_velocity=0.25,
                position_tolerance=0.05
            ),
            "late_embryonic": MigrationParameters(
                g1_apical_position=0.2,
                s_basal_position=0.7,
                g2_apical_position=0.3,
                m_apical_position=0.15,
                migration_velocity=0.2,
                position_tolerance=0.05
            ),
            "fetal": MigrationParameters(
                g1_apical_position=0.25,
                s_basal_position=0.65,
                g2_apical_position=0.35,
                m_apical_position=0.2,
                migration_velocity=0.15,
                position_tolerance=0.05
            )
        }
    
    def initialize_nuclear_migration(self, cell_id: str, initial_position: float,
                                  cell_cycle_phase: CellCyclePhase,
                                  developmental_stage: str = "mid_embryonic") -> NuclearMigrationState:
        """
        Initialize nuclear migration state for cell
        
        Args:
            cell_id: Unique cell identifier
            initial_position: Initial nuclear position (0.0 to 1.0)
            cell_cycle_phase: Current cell cycle phase
            developmental_stage: Developmental stage for parameter selection
            
        Returns:
            NuclearMigrationState with initial migration state
        """
        params = self.migration_parameters.get(developmental_stage, 
                                             self.migration_parameters["mid_embryonic"])
        
        # Determine target position based on cell cycle phase
        target_position = self._get_target_position_for_phase(cell_cycle_phase, params)
        
        # Determine migration direction
        migration_direction = self._determine_migration_direction(initial_position, target_position)
        
        # Calculate migration velocity
        migration_velocity = self._calculate_migration_velocity(
            initial_position, target_position, params
        )
        
        nuclear_state = NuclearMigrationState(
            cell_id=cell_id,
            current_position=initial_position,
            target_position=target_position,
            migration_velocity=migration_velocity,
            migration_direction=migration_direction,
            cell_cycle_phase=cell_cycle_phase,
            migration_active=True,
            time_in_phase=0.0
        )
        
        self.nuclear_states[cell_id] = nuclear_state
        return nuclear_state
    
    def _get_target_position_for_phase(self, phase: CellCyclePhase, 
                                     params: MigrationParameters) -> float:
        """Get target nuclear position for cell cycle phase"""
        phase_positions = {
            CellCyclePhase.G1: params.g1_apical_position,
            CellCyclePhase.S: params.s_basal_position,
            CellCyclePhase.G2: params.g2_apical_position,
            CellCyclePhase.M: params.m_apical_position
        }
        return phase_positions.get(phase, params.g1_apical_position)
    
    def _determine_migration_direction(self, current_position: float, 
                                     target_position: float) -> str:
        """Determine migration direction based on current and target positions"""
        if abs(current_position - target_position) < 0.01:
            return "stationary"
        elif current_position < target_position:
            return "basal"
        else:
            return "apical"
    
    def _calculate_migration_velocity(self, current_position: float,
                                   target_position: float,
                                   params: MigrationParameters) -> float:
        """Calculate migration velocity based on distance to target"""
        distance = abs(current_position - target_position)
        
        # Base velocity from parameters
        base_velocity = params.migration_velocity
        
        # Adjust velocity based on distance (closer = slower)
        if distance < 0.1:
            velocity_factor = 0.5  # Slow down when close to target
        elif distance < 0.3:
            velocity_factor = 0.8
        else:
            velocity_factor = 1.0  # Full speed when far from target
        
        return base_velocity * velocity_factor
    
    def update_nuclear_migration(self, cell_id: str, time_delta: float,
                               new_cell_cycle_phase: Optional[CellCyclePhase] = None) -> NuclearMigrationState:
        """
        Update nuclear migration state over time
        
        Args:
            cell_id: Unique cell identifier
            time_delta: Time step in hours
            new_cell_cycle_phase: New cell cycle phase (if changed)
            
        Returns:
            Updated NuclearMigrationState
        """
        if cell_id not in self.nuclear_states:
            raise ValueError(f"Cell {cell_id} not found in nuclear migration states")
        
        state = self.nuclear_states[cell_id]
        
        # Update time in phase
        state.time_in_phase += time_delta
        
        # Check for cell cycle phase change
        if new_cell_cycle_phase is not None and new_cell_cycle_phase != state.cell_cycle_phase:
            state.cell_cycle_phase = new_cell_cycle_phase
            state.time_in_phase = 0.0
            
            # Update target position for new phase
            developmental_stage = "mid_embryonic"  # Default
            params = self.migration_parameters[developmental_stage]
            state.target_position = self._get_target_position_for_phase(new_cell_cycle_phase, params)
            
            # Update migration direction and velocity
            state.migration_direction = self._determine_migration_direction(
                state.current_position, state.target_position
            )
            state.migration_velocity = self._calculate_migration_velocity(
                state.current_position, state.target_position, params
            )
        
        # Update nuclear position if migration is active
        if state.migration_active:
            self._update_nuclear_position(state, time_delta)
        
        return state
    
    def _update_nuclear_position(self, state: NuclearMigrationState, time_delta: float) -> None:
        """Update nuclear position based on migration"""
        if state.migration_direction == "stationary":
            return
        
        # Calculate position change
        position_change = state.migration_velocity * time_delta
        
        if state.migration_direction == "basal":
            new_position = state.current_position + position_change
        else:  # apical
            new_position = state.current_position - position_change
        
        # Clamp position to valid range
        new_position = max(0.0, min(1.0, new_position))
        
        # Check if target reached
        params = self.migration_parameters["mid_embryonic"]  # Default
        if abs(new_position - state.target_position) < params.position_tolerance:
            state.current_position = state.target_position
            state.migration_active = False
            state.migration_direction = "stationary"
        else:
            state.current_position = new_position
    
    def get_nuclear_position_state(self, cell_id: str) -> NuclearPosition:
        """Get current nuclear position state"""
        if cell_id not in self.nuclear_states:
            return NuclearPosition.STATIONARY
        
        state = self.nuclear_states[cell_id]
        
        if not state.migration_active:
            return NuclearPosition.STATIONARY
        elif state.migration_direction == "apical":
            return NuclearPosition.MIGRATING_APICAL
        elif state.migration_direction == "basal":
            return NuclearPosition.MIGRATING_BASAL
        else:
            # Determine if at apical or basal position
            if state.current_position < 0.2:
                return NuclearPosition.APICAL
            elif state.current_position > 0.8:
                return NuclearPosition.BASAL
            else:
                return NuclearPosition.STATIONARY
    
    def get_migration_statistics(self, cell_ids: List[str]) -> Dict[str, float]:
        """Get migration statistics for multiple cells"""
        if not cell_ids:
            return {}
        
        positions = []
        velocities = []
        active_migrations = 0
        
        for cell_id in cell_ids:
            if cell_id in self.nuclear_states:
                state = self.nuclear_states[cell_id]
                positions.append(state.current_position)
                velocities.append(state.migration_velocity)
                if state.migration_active:
                    active_migrations += 1
        
        if not positions:
            return {}
        
        return {
            "total_cells": len(cell_ids),
            "active_migrations": active_migrations,
            "migration_activity": active_migrations / len(cell_ids),
            "average_position": np.mean(positions),
            "position_std": np.std(positions),
            "average_velocity": np.mean(velocities),
            "velocity_std": np.std(velocities)
        }
    
    def pause_migration(self, cell_id: str) -> None:
        """Pause nuclear migration for cell"""
        if cell_id in self.nuclear_states:
            self.nuclear_states[cell_id].migration_active = False
    
    def resume_migration(self, cell_id: str) -> None:
        """Resume nuclear migration for cell"""
        if cell_id in self.nuclear_states:
            self.nuclear_states[cell_id].migration_active = True
    
    def force_nuclear_position(self, cell_id: str, position: float) -> None:
        """Force nuclear to specific position"""
        if cell_id in self.nuclear_states:
            state = self.nuclear_states[cell_id]
            state.current_position = max(0.0, min(1.0, position))
            state.migration_active = False
            state.migration_direction = "stationary"
