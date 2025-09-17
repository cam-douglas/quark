"""
Cell Fate Decision Integrator

This module integrates the developmental biology lineage system with the
foundation layer cell fate specifier, implementing progenitor → committed
cell transitions and controlling timing of fate commitment.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .neuroepithelial_cells import NeuroepithelialCell
from .morphogen_responsiveness_integrator import CellFateDecision
from .fate_commitment_manager import (
    FateCommitmentManager, FateCommitmentState, 
    FateTransitionEvent, FateCommitmentStage
)


class CellFateDecisionIntegrator:
    """
    Integrates the developmental biology lineage system with the
    foundation layer cell fate specifier, implementing progenitor → committed
    cell transitions and controlling timing of fate commitment.
    """
    
    def __init__(self):
        """Initialize cell fate decision integrator"""
        self.fate_commitment_states: Dict[str, FateCommitmentState] = {}
        self.fate_transition_events: List[FateTransitionEvent] = []
        self.commitment_manager = FateCommitmentManager()
    
    
    def integrate_with_fate_specifier(self, cells: Dict[str, NeuroepithelialCell], 
                                    fate_decisions: Dict[str, CellFateDecision],
                                    time_delta: float) -> Dict[str, FateCommitmentState]:
        """Integrate with foundation layer cell fate specifier using commitment manager"""
        updated_states = {}
        
        for cell_id, cell in cells.items():
            # Get or create fate commitment state
            if cell_id not in self.fate_commitment_states:
                fate_decision = fate_decisions.get(cell_id)
                target_fate = fate_decision.fate_type if fate_decision else "unknown"
                commitment_strength = fate_decision.decision_confidence if fate_decision else 0.0
                
                state = self.commitment_manager.initialize_fate_commitment_state(
                    cell_id, cell, target_fate, commitment_strength
                )
                self.fate_commitment_states[cell_id] = state
            else:
                state = self.fate_commitment_states[cell_id]
            
            # Update fate commitment based on foundation layer decisions
            fate_decision = fate_decisions.get(cell_id)
            commitment_strength = fate_decision.decision_confidence if fate_decision else 0.0
            
            updated_state = self.commitment_manager.update_fate_commitment_state(
                state, cell, commitment_strength, time_delta
            )
            updated_states[cell_id] = updated_state
            
            # Check for fate transitions
            if self.commitment_manager.should_transition_fate(updated_state, cell):
                transition_event = self.commitment_manager.execute_fate_transition(updated_state, cell)
                if transition_event:
                    self.fate_transition_events.append(transition_event)
        
        return updated_states
    
    def get_fate_commitment_statistics(self, cell_ids: List[str]) -> Dict[str, float]:
        """Get fate commitment statistics for multiple cells"""
        if not cell_ids:
            return {}
        
        stages = []
        commitment_strengths = []
        transition_probabilities = []
        
        for cell_id in cell_ids:
            if cell_id in self.fate_commitment_states:
                state = self.fate_commitment_states[cell_id]
                stages.append(state.current_stage.value)
                commitment_strengths.append(state.commitment_strength)
                transition_probabilities.append(state.transition_probability)
        
        if not stages:
            return {}
        
        # Count stages
        stage_counts = {}
        for stage in stages:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return {
            "total_cells": len(cell_ids),
            "average_commitment_strength": np.mean(commitment_strengths),
            "average_transition_probability": np.mean(transition_probabilities),
            "stage_distribution": stage_counts,
            "total_transitions": len(self.fate_transition_events)
        }
    
    
