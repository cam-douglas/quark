"""
Differentiation Timing Parameters

This module manages timing parameters for differentiation timing control
including competency windows, triggers, and phase transitions.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType


class DifferentiationTrigger(Enum):
    """Triggers for differentiation"""
    COMPETENCY_WINDOW = "competency_window"
    CELL_CYCLE_EXIT = "cell_cycle_exit"
    MORPHOGEN_THRESHOLD = "morphogen_threshold"
    TEMPORAL_SIGNAL = "temporal_signal"
    NEIGHBOR_INFLUENCE = "neighbor_influence"


class DifferentiationTimingPhase(Enum):
    """Phases of differentiation timing"""
    COMPETENCY_ACQUISITION = "competency_acquisition"
    COMPETENCY_MAINTENANCE = "competency_maintenance"
    COMPETENCY_LOSS = "competency_loss"
    DIFFERENTIATION_INITIATION = "differentiation_initiation"
    DIFFERENTIATION_PROGRESSION = "differentiation_progression"


@dataclass
class TimingParameters:
    """Parameters for differentiation timing"""
    competency_acquisition_rate: float
    competency_maintenance_rate: float
    competency_loss_rate: float
    differentiation_threshold: float
    cell_cycle_exit_probability: float
    morphogen_sensitivity: float


class DifferentiationTimingParameters:
    """
    Manages timing parameters for differentiation timing control
    including competency windows, triggers, and phase transitions.
    """
    
    def __init__(self):
        """Initialize differentiation timing parameters"""
        self.timing_parameters: Dict[str, TimingParameters] = {}
        self._setup_timing_parameters()
    
    def _setup_timing_parameters(self) -> None:
        """Setup timing parameters for different zones and cell types"""
        self.timing_parameters = {
            "ventricular_zone": TimingParameters(
                competency_acquisition_rate=0.1,
                competency_maintenance_rate=0.05,
                competency_loss_rate=0.02,
                differentiation_threshold=0.8,
                cell_cycle_exit_probability=0.3,
                morphogen_sensitivity=0.7
            ),
            "subventricular_zone": TimingParameters(
                competency_acquisition_rate=0.15,
                competency_maintenance_rate=0.08,
                competency_loss_rate=0.03,
                differentiation_threshold=0.7,
                cell_cycle_exit_probability=0.4,
                morphogen_sensitivity=0.6
            ),
            "intermediate_zone": TimingParameters(
                competency_acquisition_rate=0.2,
                competency_maintenance_rate=0.1,
                competency_loss_rate=0.05,
                differentiation_threshold=0.6,
                cell_cycle_exit_probability=0.5,
                morphogen_sensitivity=0.5
            ),
            "mantle_zone": TimingParameters(
                competency_acquisition_rate=0.25,
                competency_maintenance_rate=0.12,
                competency_loss_rate=0.08,
                differentiation_threshold=0.5,
                cell_cycle_exit_probability=0.6,
                morphogen_sensitivity=0.4
            )
        }
    
    def get_timing_parameters(self, zone_type: ZoneType) -> TimingParameters:
        """Get timing parameters for specific zone"""
        zone_name = zone_type.value
        return self.timing_parameters.get(zone_name, self.timing_parameters["ventricular_zone"])
    
    def update_competency_level(self, current_competency: float, phase: DifferentiationTimingPhase, 
                              params: TimingParameters, time_delta: float) -> float:
        """Update competency level based on current phase and timing"""
        if phase == DifferentiationTimingPhase.COMPETENCY_ACQUISITION:
            # Increase competency
            rate = params.competency_acquisition_rate
            new_competency = min(1.0, current_competency + rate * time_delta)
        elif phase == DifferentiationTimingPhase.COMPETENCY_MAINTENANCE:
            # Maintain competency with slight decay
            rate = params.competency_maintenance_rate
            new_competency = max(0.0, current_competency - rate * time_delta * 0.1)
        elif phase == DifferentiationTimingPhase.COMPETENCY_LOSS:
            # Decrease competency
            rate = params.competency_loss_rate
            new_competency = max(0.0, current_competency - rate * time_delta)
        else:
            # During differentiation, competency may change
            new_competency = current_competency
        
        return new_competency
    
    def calculate_differentiation_readiness(self, competency_level: float, cell_state: str) -> float:
        """Calculate differentiation readiness based on competency and cell state"""
        # Base readiness from competency
        base_readiness = competency_level
        
        # Adjust based on cell state
        state_factors = {
            "progenitor": 0.3,
            "committed": 0.6,
            "differentiating": 0.9,
            "differentiated": 1.0
        }
        
        state_factor = state_factors.get(cell_state, 0.5)
        
        # Combined readiness
        readiness = base_readiness * state_factor
        
        return min(1.0, max(0.0, readiness))
    
    def identify_active_triggers(self, competency_level: float, differentiation_readiness: float,
                               params: TimingParameters, cell_state: str) -> List[DifferentiationTrigger]:
        """Identify active differentiation triggers"""
        active_triggers = []
        
        # Check competency window trigger
        if competency_level >= params.differentiation_threshold:
            active_triggers.append(DifferentiationTrigger.COMPETENCY_WINDOW)
        
        # Check cell cycle exit trigger
        if self._should_exit_cell_cycle(params, differentiation_readiness):
            active_triggers.append(DifferentiationTrigger.CELL_CYCLE_EXIT)
        
        # Check morphogen threshold trigger
        if differentiation_readiness >= params.morphogen_sensitivity:
            active_triggers.append(DifferentiationTrigger.MORPHOGEN_THRESHOLD)
        
        # Check temporal signal trigger
        if self._check_temporal_signal():
            active_triggers.append(DifferentiationTrigger.TEMPORAL_SIGNAL)
        
        # Check neighbor influence trigger
        if self._check_neighbor_influence(differentiation_readiness):
            active_triggers.append(DifferentiationTrigger.NEIGHBOR_INFLUENCE)
        
        return active_triggers
    
    def _should_exit_cell_cycle(self, params: TimingParameters, differentiation_readiness: float) -> bool:
        """Determine if cell should exit cell cycle for differentiation"""
        # Adjust probability based on differentiation readiness
        adjusted_probability = params.cell_cycle_exit_probability * differentiation_readiness
        
        return np.random.random() < adjusted_probability
    
    def _check_temporal_signal(self) -> bool:
        """Check if temporal signal indicates differentiation timing"""
        # Time-based differentiation signal (simplified)
        return np.random.random() < 0.1  # 10% chance per check
    
    def _check_neighbor_influence(self, differentiation_readiness: float) -> bool:
        """Check if neighbor influence triggers differentiation"""
        neighbor_threshold = 0.7
        return differentiation_readiness >= neighbor_threshold
    
    def get_next_phase(self, current_phase: DifferentiationTimingPhase, 
                      trigger: DifferentiationTrigger) -> DifferentiationTimingPhase:
        """Get next phase based on current phase and trigger"""
        phase_transitions = {
            DifferentiationTimingPhase.COMPETENCY_ACQUISITION: DifferentiationTimingPhase.COMPETENCY_MAINTENANCE,
            DifferentiationTimingPhase.COMPETENCY_MAINTENANCE: DifferentiationTimingPhase.DIFFERENTIATION_INITIATION,
            DifferentiationTimingPhase.DIFFERENTIATION_INITIATION: DifferentiationTimingPhase.DIFFERENTIATION_PROGRESSION,
            DifferentiationTimingPhase.DIFFERENTIATION_PROGRESSION: DifferentiationTimingPhase.DIFFERENTIATION_PROGRESSION,  # Terminal
            DifferentiationTimingPhase.COMPETENCY_LOSS: DifferentiationTimingPhase.COMPETENCY_ACQUISITION  # Can reacquire
        }
        
        return phase_transitions.get(current_phase, current_phase)
    
    def should_change_phase(self, trigger: DifferentiationTrigger) -> bool:
        """Determine if trigger should cause phase change"""
        phase_change_triggers = {
            DifferentiationTrigger.COMPETENCY_WINDOW: True,
            DifferentiationTrigger.CELL_CYCLE_EXIT: True,
            DifferentiationTrigger.MORPHOGEN_THRESHOLD: True,
            DifferentiationTrigger.TEMPORAL_SIGNAL: False,
            DifferentiationTrigger.NEIGHBOR_INFLUENCE: False
        }
        
        return phase_change_triggers.get(trigger, False)
