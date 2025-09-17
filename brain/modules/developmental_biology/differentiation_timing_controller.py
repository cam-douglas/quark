"""
Differentiation Timing Controller

This module models timing of neuronal differentiation, implements competency
window restrictions, and controls exit from cell cycle for proper differentiation.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType
from .competency_curve_calculator import CompetencyWindow
from .ventricular_zone_organizer import ZoneType
from .differentiation_timing_parameters import (
    DifferentiationTimingParameters, DifferentiationTrigger, 
    DifferentiationTimingPhase, TimingParameters
)


@dataclass
class DifferentiationTimingState:
    """State of differentiation timing for a cell"""
    cell_id: str
    current_phase: DifferentiationTimingPhase
    competency_level: float
    competency_window: CompetencyWindow
    differentiation_readiness: float
    time_in_phase: float
    active_triggers: List[DifferentiationTrigger]


@dataclass
class DifferentiationEvent:
    """Event representing differentiation timing change"""
    cell_id: str
    event_type: DifferentiationTrigger
    phase: DifferentiationTimingPhase
    timestamp: float
    confidence: float


class DifferentiationTimingController:
    """
    Models timing of neuronal differentiation, implements competency
    window restrictions, and controls exit from cell cycle for proper differentiation.
    """
    
    def __init__(self):
        """Initialize differentiation timing controller"""
        self.timing_states: Dict[str, DifferentiationTimingState] = {}
        self.differentiation_events: List[DifferentiationEvent] = []
        self.timing_parameters_manager = DifferentiationTimingParameters()
    
    
    def control_differentiation_timing(self, cells: Dict[str, NeuroepithelialCell], 
                                     competency_windows: Dict[str, CompetencyWindow],
                                     time_delta: float) -> Dict[str, DifferentiationTimingState]:
        """
        Control differentiation timing for cells
        
        Args:
            cells: Dictionary of neuroepithelial cells
            competency_windows: Dictionary of competency windows for cells
            time_delta: Time step in hours
            
        Returns:
            Dictionary of updated differentiation timing states
        """
        updated_states = {}
        
        for cell_id, cell in cells.items():
            # Get or create timing state
            if cell_id not in self.timing_states:
                state = self._initialize_timing_state(cell_id, cell, competency_windows.get(cell_id))
                self.timing_states[cell_id] = state
            else:
                state = self.timing_states[cell_id]
            
            # Update timing state
            updated_state = self._update_timing_state(state, cell, time_delta)
            updated_states[cell_id] = updated_state
            
            # Check for differentiation triggers
            triggers = self._check_differentiation_triggers(updated_state, cell)
            if triggers:
                for trigger in triggers:
                    event = self._process_differentiation_trigger(updated_state, cell, trigger)
                    if event:
                        self.differentiation_events.append(event)
        
        return updated_states
    
    def _initialize_timing_state(self, cell_id: str, cell: NeuroepithelialCell, 
                               competency_window: Optional[CompetencyWindow]) -> DifferentiationTimingState:
        """Initialize differentiation timing state for a cell"""
        # Determine initial phase based on cell type
        if cell.cell_type == NeuroepithelialCellType.EARLY_MULTIPOTENT:
            initial_phase = DifferentiationTimingPhase.COMPETENCY_ACQUISITION
        elif cell.cell_type == NeuroepithelialCellType.LATE_MULTIPOTENT:
            initial_phase = DifferentiationTimingPhase.COMPETENCY_MAINTENANCE
        elif cell.cell_type == NeuroepithelialCellType.COMMITTED_PROGENITOR:
            initial_phase = DifferentiationTimingPhase.DIFFERENTIATION_PROGRESSION
        else:
            initial_phase = DifferentiationTimingPhase.DIFFERENTIATION_PROGRESSION
        
        # Get competency window
        if competency_window is None:
            # Create default competency window
            competency_window = CompetencyWindow(
                fate_type="default",
                competency_start=0.0,
                competency_end=10.0,
                competency_strength=1.0,
                required_signals=[]
            )
        
        # Calculate initial competency level
        competency_level = self._calculate_competency_level(competency_window, 0.0)
        
        # Calculate differentiation readiness
        differentiation_readiness = self.timing_parameters_manager.calculate_differentiation_readiness(
            competency_level, cell.cell_type.value
        )
        
        return DifferentiationTimingState(
            cell_id=cell_id,
            current_phase=initial_phase,
            competency_level=competency_level,
            competency_window=competency_window,
            differentiation_readiness=differentiation_readiness,
            time_in_phase=0.0,
            active_triggers=[]
        )
    
    def _update_timing_state(self, state: DifferentiationTimingState, cell: NeuroepithelialCell, 
                           time_delta: float) -> DifferentiationTimingState:
        """Update differentiation timing state over time"""
        # Update time in current phase
        state.time_in_phase += time_delta
        
        # Get timing parameters
        zone_type = cell.zone_type if hasattr(cell, 'zone_type') else ZoneType.VENTRICULAR_ZONE
        params = self.timing_parameters_manager.get_timing_parameters(zone_type)
        
        # Update competency level based on current phase
        state.competency_level = self.timing_parameters_manager.update_competency_level(
            state.competency_level, state.current_phase, params, time_delta
        )
        
        # Update differentiation readiness
        state.differentiation_readiness = self.timing_parameters_manager.calculate_differentiation_readiness(
            state.competency_level, cell.cell_type.value
        )
        
        # Update active triggers
        state.active_triggers = self.timing_parameters_manager.identify_active_triggers(
            state.competency_level, state.differentiation_readiness, params, cell.cell_type.value
        )
        
        return state
    
    def _check_differentiation_triggers(self, state: DifferentiationTimingState, cell: NeuroepithelialCell) -> List[DifferentiationTrigger]:
        """Check for differentiation triggers"""
        return state.active_triggers
    
    def _calculate_competency_level(self, competency_window: CompetencyWindow, time: float) -> float:
        """Calculate competency level from competency window"""
        if time < competency_window.competency_start:
            return 0.0
        elif time < competency_window.competency_end:
            # Linear increase to peak, then linear decrease
            mid_time = (competency_window.competency_start + competency_window.competency_end) / 2
            if time < mid_time:
                progress = (time - competency_window.competency_start) / (mid_time - competency_window.competency_start)
                return progress * competency_window.competency_strength
            else:
                progress = (time - mid_time) / (competency_window.competency_end - mid_time)
                return competency_window.competency_strength * (1.0 - progress)
        else:
            return 0.0
    
    def _process_differentiation_trigger(self, state: DifferentiationTimingState, cell: NeuroepithelialCell, 
                                       trigger: DifferentiationTrigger) -> Optional[DifferentiationEvent]:
        """Process differentiation trigger and return event"""
        # Determine if trigger should cause phase change
        if self.timing_parameters_manager.should_change_phase(trigger):
            # Update phase
            new_phase = self.timing_parameters_manager.get_next_phase(state.current_phase, trigger)
            state.current_phase = new_phase
            state.time_in_phase = 0.0
            
            # Create event
            event = DifferentiationEvent(
                cell_id=state.cell_id,
                event_type=trigger,
                phase=new_phase,
                timestamp=0.0,
                confidence=state.differentiation_readiness
            )
            
            return event
        
        return None
    
    def get_differentiation_timing_statistics(self, cell_ids: List[str]) -> Dict[str, float]:
        """Get differentiation timing statistics for multiple cells"""
        if not cell_ids:
            return {}
        
        phases = []
        competency_levels = []
        readiness_levels = []
        trigger_counts = {}
        
        for cell_id in cell_ids:
            if cell_id in self.timing_states:
                state = self.timing_states[cell_id]
                phases.append(state.current_phase.value)
                competency_levels.append(state.competency_level)
                readiness_levels.append(state.differentiation_readiness)
                
                # Count triggers
                for trigger in state.active_triggers:
                    trigger_counts[trigger.value] = trigger_counts.get(trigger.value, 0) + 1
        
        if not phases:
            return {}
        
        # Count phases
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        return {
            "total_cells": len(cell_ids),
            "average_competency_level": np.mean(competency_levels),
            "average_readiness_level": np.mean(readiness_levels),
            "phase_distribution": phase_counts,
            "active_trigger_counts": trigger_counts,
            "total_differentiation_events": len(self.differentiation_events)
        }
    
