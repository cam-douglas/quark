#!/usr/bin/env python3
"""Progenitor Competency Management System.

Main coordinator for progenitor competency modeling including temporal
window management, restriction mechanisms, and fate competition for
neuroepithelial cells during embryonic development.

Integration: Main competency coordinator for developmental biology system
Rationale: Centralized competency management with focused responsibilities
"""

from typing import Dict, List, Optional, Set
import numpy as np
import random
import logging

from .competency_window_types import (
    CompetencyWindow, CompetencyState, FateRestrictionEvent, 
    FateType, CompetencyProfile, RestrictionMechanism
)
from .competency_curve_calculator import CompetencyCurveCalculator

logger = logging.getLogger(__name__)

class ProgenitorCompetencyManager:
    """Manager for progenitor cell competency modeling.
    
    Main coordinator for temporal competency windows, restriction
    mechanisms, and fate specification for neuroepithelial cells
    during embryonic neural tube development.
    
    Key Components:
    - Temporal competency window definitions
    - Competency curve calculations
    - Restriction mechanism implementation
    - Fate competition modeling
    """
    
    def __init__(self):
        """Initialize progenitor competency manager."""
        self.curve_calculator = CompetencyCurveCalculator()
        self.competency_windows = self._initialize_competency_windows()
        self.cell_competency_states: Dict[str, CompetencyState] = {}
        self.restriction_events: List[FateRestrictionEvent] = []
        
        logger.info("Initialized ProgenitorCompetencyManager")
        logger.info(f"Competency windows: {len(self.competency_windows)}")
    
    def _initialize_competency_windows(self) -> Dict[FateType, CompetencyWindow]:
        """Initialize temporal competency windows for all fate types."""
        windows = {}
        
        # Motor neuron competency (ventral, SHH-dependent)
        windows[FateType.MOTOR_NEURON] = CompetencyWindow(
            fate_type=FateType.MOTOR_NEURON,
            competency_start=8.5,
            competency_peak=9.5,
            competency_end=11.0,
            max_competency_strength=0.9,
            competency_profile=CompetencyProfile.BELL_CURVE,
            required_morphogens=['SHH'],
            inhibitory_morphogens=['BMP'],
            restriction_mechanisms=[RestrictionMechanism.TEMPORAL_DECAY, 
                                  RestrictionMechanism.MORPHOGEN_INHIBITION]
        )
        
        # V0 Interneuron competency
        windows[FateType.INTERNEURON_V0] = CompetencyWindow(
            fate_type=FateType.INTERNEURON_V0,
            competency_start=8.5,
            competency_peak=9.0,
            competency_end=10.5,
            max_competency_strength=0.8,
            competency_profile=CompetencyProfile.EARLY_PEAK,
            required_morphogens=['SHH'],
            inhibitory_morphogens=[],
            restriction_mechanisms=[RestrictionMechanism.TEMPORAL_DECAY]
        )
        
        # Dorsal interneuron competency (BMP-dependent)
        windows[FateType.INTERNEURON_DORSAL] = CompetencyWindow(
            fate_type=FateType.INTERNEURON_DORSAL,
            competency_start=8.0,
            competency_peak=10.0,
            competency_end=12.0,
            max_competency_strength=0.8,
            competency_profile=CompetencyProfile.SUSTAINED,
            required_morphogens=['BMP'],
            inhibitory_morphogens=['SHH'],
            restriction_mechanisms=[RestrictionMechanism.MORPHOGEN_INHIBITION,
                                  RestrictionMechanism.EPIGENETIC_SILENCING]
        )
        
        # Neural crest competency (early, BMP/WNT-dependent)
        windows[FateType.NEURAL_CREST] = CompetencyWindow(
            fate_type=FateType.NEURAL_CREST,
            competency_start=8.0,
            competency_peak=8.5,
            competency_end=10.0,
            max_competency_strength=0.7,
            competency_profile=CompetencyProfile.EARLY_PEAK,
            required_morphogens=['BMP', 'WNT'],
            inhibitory_morphogens=['SHH'],
            restriction_mechanisms=[RestrictionMechanism.TEMPORAL_DECAY,
                                  RestrictionMechanism.TRANSCRIPTIONAL_REPRESSION]
        )
        
        # Oligodendrocyte competency (late, SHH-dependent)
        windows[FateType.OLIGODENDROCYTE] = CompetencyWindow(
            fate_type=FateType.OLIGODENDROCYTE,
            competency_start=10.0,
            competency_peak=12.0,
            competency_end=14.0,
            max_competency_strength=0.6,
            competency_profile=CompetencyProfile.LATE_ONSET,
            required_morphogens=['SHH'],
            inhibitory_morphogens=['BMP'],
            restriction_mechanisms=[RestrictionMechanism.TEMPORAL_DECAY]
        )
        
        # Astrocyte competency (latest, BMP-dependent)
        windows[FateType.ASTROCYTE] = CompetencyWindow(
            fate_type=FateType.ASTROCYTE,
            competency_start=11.0,
            competency_peak=13.0,
            competency_end=16.0,
            max_competency_strength=0.5,
            competency_profile=CompetencyProfile.LATE_ONSET,
            required_morphogens=['BMP'],
            inhibitory_morphogens=['SHH'],
            restriction_mechanisms=[RestrictionMechanism.TEMPORAL_DECAY,
                                  RestrictionMechanism.METABOLIC_CONSTRAINT]
        )
        
        return windows
    
    def initialize_cell_competency(self, cell_id: str, developmental_time: float) -> CompetencyState:
        """Initialize competency state for a new cell.
        
        Args:
            cell_id: Cell identifier
            developmental_time: Current developmental time
            
        Returns:
            Initial competency state
        """
        # Calculate initial competencies for all fates
        initial_competencies = {}
        
        for fate_type, window in self.competency_windows.items():
            # Start with base competency if within window
            if window.competency_start <= developmental_time <= window.competency_end:
                competency = self.curve_calculator.calculate_competency_level(
                    window, developmental_time, {})
                initial_competencies[fate_type] = competency
            else:
                initial_competencies[fate_type] = 0.0
        
        # Create competency state
        competency_state = CompetencyState(
            cell_id=cell_id,
            current_competencies=initial_competencies,
            competency_history=[initial_competencies.copy()],
            restricted_fates=set(),
            competency_last_update=developmental_time
        )
        
        self.cell_competency_states[cell_id] = competency_state
        
        logger.debug(f"Initialized competency for cell {cell_id}")
        
        return competency_state
    
    def update_cell_competency(self, cell_id: str, developmental_time: float,
                              morphogen_levels: Dict[str, float]) -> CompetencyState:
        """Update competency state for a cell.
        
        Args:
            cell_id: Cell identifier
            developmental_time: Current developmental time
            morphogen_levels: Current morphogen levels
            
        Returns:
            Updated competency state
        """
        if cell_id not in self.cell_competency_states:
            return self.initialize_cell_competency(cell_id, developmental_time)
        
        competency_state = self.cell_competency_states[cell_id]
        dt = developmental_time - competency_state.competency_last_update
        
        # Update competencies for each fate
        updated_competencies = {}
        
        for fate_type, window in self.competency_windows.items():
            if fate_type in competency_state.restricted_fates:
                # Fate is permanently restricted
                updated_competencies[fate_type] = 0.0
            else:
                # Calculate new competency level
                new_competency = self.curve_calculator.calculate_competency_level(
                    window, developmental_time, morphogen_levels)
                
                # Apply restriction
                restriction_rate = self.curve_calculator.calculate_competency_restriction_rate(
                    window, developmental_time, morphogen_levels)
                
                # Reduce competency based on restriction rate
                current_competency = competency_state.current_competencies.get(fate_type, 0.0)
                restricted_competency = current_competency * (1.0 - restriction_rate * dt)
                
                # Take minimum of new calculation and restricted level
                final_competency = min(new_competency, restricted_competency)
                updated_competencies[fate_type] = max(0.0, final_competency)
                
                # Check for permanent restriction
                if final_competency < 0.05 and developmental_time > window.competency_end:
                    competency_state.restricted_fates.add(fate_type)
                    self._record_restriction_event(fate_type, developmental_time, cell_id)
        
        # Update competency state
        competency_state.current_competencies = updated_competencies
        competency_state.competency_history.append(updated_competencies.copy())
        competency_state.competency_last_update = developmental_time
        
        return competency_state
    
    def _record_restriction_event(self, fate_type: FateType, restriction_time: float,
                                 cell_id: str) -> None:
        """Record a fate restriction event."""
        # Determine primary restriction mechanism
        window = self.competency_windows[fate_type]
        primary_mechanism = window.restriction_mechanisms[0] if window.restriction_mechanisms else RestrictionMechanism.TEMPORAL_DECAY
        
        restriction_event = FateRestrictionEvent(
            fate_type=fate_type,
            restriction_time=restriction_time,
            restriction_mechanism=primary_mechanism,
            triggering_signal=None,  # Could be determined from morphogen levels
            irreversible=True
        )
        
        self.restriction_events.append(restriction_event)
        logger.debug(f"Fate {fate_type.value} restricted for cell {cell_id} at E{restriction_time}")
    
    def get_competency_summary(self, cell_id: str) -> Optional[Dict[str, any]]:
        """Get competency summary for a cell.
        
        Args:
            cell_id: Cell identifier
            
        Returns:
            Competency summary or None if cell not found
        """
        if cell_id not in self.cell_competency_states:
            return None
        
        state = self.cell_competency_states[cell_id]
        
        return {
            'cell_id': cell_id,
            'current_competencies': state.current_competencies,
            'active_competencies': {fate: comp for fate, comp in state.current_competencies.items() if comp > 0.1},
            'restricted_fates': [fate.value for fate in state.restricted_fates],
            'total_competency': sum(state.current_competencies.values()),
            'most_competent_fate': max(state.current_competencies.items(), key=lambda x: x[1]) if state.current_competencies else None,
            'competency_diversity': len([comp for comp in state.current_competencies.values() if comp > 0.1]),
            'last_update_time': state.competency_last_update
        }
