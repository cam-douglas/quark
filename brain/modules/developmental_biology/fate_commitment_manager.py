"""
Fate Commitment Manager

This module manages fate commitment states and transitions for
cell fate decision integration.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType
from .ventricular_zone_organizer import ZoneType


class FateCommitmentStage(Enum):
    """Stages of fate commitment"""
    PROGENITOR = "progenitor"
    COMMITTED = "committed"
    DIFFERENTIATING = "differentiating"
    DIFFERENTIATED = "differentiated"


class FateTransitionType(Enum):
    """Types of fate transitions"""
    PROGENITOR_TO_COMMITTED = "progenitor_to_committed"
    COMMITTED_TO_DIFFERENTIATING = "committed_to_differentiating"
    DIFFERENTIATING_TO_DIFFERENTIATED = "differentiating_to_differentiated"
    REVERSAL = "reversal"


@dataclass
class FateCommitmentState:
    """State of fate commitment for a cell"""
    cell_id: str
    current_stage: FateCommitmentStage
    target_fate: str
    commitment_strength: float
    transition_probability: float
    time_in_stage: float
    commitment_factors: Dict[str, float]


@dataclass
class FateTransitionEvent:
    """Event representing a fate transition"""
    cell_id: str
    from_stage: FateCommitmentStage
    to_stage: FateCommitmentStage
    transition_type: FateTransitionType
    timestamp: float
    confidence: float


class FateCommitmentManager:
    """
    Manages fate commitment states and transitions for
    cell fate decision integration.
    """
    
    def __init__(self):
        """Initialize fate commitment manager"""
        self.commitment_parameters: Dict[str, Dict[str, float]] = {}
        self._setup_commitment_parameters()
    
    def _setup_commitment_parameters(self) -> None:
        """Setup commitment parameters for different zones and stages"""
        self.commitment_parameters = {
            "ventricular_zone": {
                "progenitor_commitment_rate": 0.1,
                "commitment_threshold": 0.7,
                "reversal_probability": 0.05,
                "time_to_commitment": 2.0
            },
            "subventricular_zone": {
                "progenitor_commitment_rate": 0.15,
                "commitment_threshold": 0.6,
                "reversal_probability": 0.03,
                "time_to_commitment": 1.5
            },
            "intermediate_zone": {
                "progenitor_commitment_rate": 0.2,
                "commitment_threshold": 0.5,
                "reversal_probability": 0.01,
                "time_to_commitment": 1.0
            },
            "mantle_zone": {
                "progenitor_commitment_rate": 0.25,
                "commitment_threshold": 0.4,
                "reversal_probability": 0.005,
                "time_to_commitment": 0.8
            }
        }
    
    def initialize_fate_commitment_state(self, cell_id: str, cell: NeuroepithelialCell, 
                                       target_fate: str, commitment_strength: float) -> FateCommitmentState:
        """Initialize fate commitment state for a cell"""
        # Determine initial stage based on cell type
        if cell.cell_type == NeuroepithelialCellType.EARLY_MULTIPOTENT:
            initial_stage = FateCommitmentStage.PROGENITOR
        elif cell.cell_type == NeuroepithelialCellType.LATE_MULTIPOTENT:
            initial_stage = FateCommitmentStage.COMMITTED
        elif cell.cell_type == NeuroepithelialCellType.COMMITTED_PROGENITOR:
            initial_stage = FateCommitmentStage.DIFFERENTIATING
        else:
            initial_stage = FateCommitmentStage.DIFFERENTIATED
        
        # Get zone-specific parameters
        zone_name = cell.zone_type.value if hasattr(cell, 'zone_type') else "ventricular_zone"
        params = self.commitment_parameters.get(zone_name, 
                                              self.commitment_parameters["ventricular_zone"])
        
        # Calculate transition probability
        transition_probability = self._calculate_transition_probability(commitment_strength, params)
        
        # Initialize commitment factors
        commitment_factors = {
            "morphogen_exposure": commitment_strength,
            "cell_cycle_position": 0.5,
            "neighbor_influence": 0.0,
            "temporal_factors": 0.0
        }
        
        return FateCommitmentState(
            cell_id=cell_id,
            current_stage=initial_stage,
            target_fate=target_fate,
            commitment_strength=commitment_strength,
            transition_probability=transition_probability,
            time_in_stage=0.0,
            commitment_factors=commitment_factors
        )
    
    def update_fate_commitment_state(self, state: FateCommitmentState, cell: NeuroepithelialCell,
                                   commitment_strength: float, time_delta: float) -> FateCommitmentState:
        """Update fate commitment state over time"""
        # Update time in current stage
        state.time_in_stage += time_delta
        
        # Update commitment strength
        state.commitment_strength = commitment_strength
        
        # Update commitment factors
        state.commitment_factors["morphogen_exposure"] = commitment_strength
        state.commitment_factors["cell_cycle_position"] = self._get_cell_cycle_position(cell)
        state.commitment_factors["neighbor_influence"] = self._calculate_neighbor_influence(cell)
        state.commitment_factors["temporal_factors"] = self._calculate_temporal_factors(state)
        
        # Recalculate transition probability
        zone_name = cell.zone_type.value if hasattr(cell, 'zone_type') else "ventricular_zone"
        params = self.commitment_parameters.get(zone_name, 
                                              self.commitment_parameters["ventricular_zone"])
        state.transition_probability = self._calculate_transition_probability(state.commitment_strength, params)
        
        return state
    
    def should_transition_fate(self, state: FateCommitmentState, cell: NeuroepithelialCell) -> bool:
        """Determine if cell should transition to next fate stage"""
        # Check if commitment strength exceeds threshold
        zone_name = cell.zone_type.value if hasattr(cell, 'zone_type') else "ventricular_zone"
        params = self.commitment_parameters.get(zone_name, 
                                              self.commitment_parameters["ventricular_zone"])
        
        if state.commitment_strength >= params["commitment_threshold"]:
            # Check transition probability
            if np.random.random() < state.transition_probability:
                return True
        
        return False
    
    def execute_fate_transition(self, state: FateCommitmentState, cell: NeuroepithelialCell) -> Optional[FateTransitionEvent]:
        """Execute fate transition and return transition event"""
        # Determine next stage
        next_stage = self._get_next_stage(state.current_stage)
        if next_stage is None:
            return None
        
        # Determine transition type
        transition_type = self._get_transition_type(state.current_stage, next_stage)
        
        # Update cell state
        self._update_cell_state(cell, next_stage)
        
        # Reset time in stage
        state.time_in_stage = 0.0
        state.current_stage = next_stage
        
        # Create transition event
        transition_event = FateTransitionEvent(
            cell_id=state.cell_id,
            from_stage=state.current_stage,
            to_stage=next_stage,
            transition_type=transition_type,
            timestamp=0.0,
            confidence=state.commitment_strength
        )
        
        return transition_event
    
    def _calculate_transition_probability(self, commitment_strength: float, 
                                        params: Dict[str, float]) -> float:
        """Calculate probability of fate transition"""
        # Base probability from commitment strength
        base_probability = commitment_strength * params["progenitor_commitment_rate"]
        
        # Time-based factor
        time_factor = min(1.0, params["time_to_commitment"] / 10.0)
        
        # Combined probability
        transition_probability = base_probability * time_factor
        
        return min(1.0, max(0.0, transition_probability))
    
    def _get_cell_cycle_position(self, cell: NeuroepithelialCell) -> float:
        """Get cell cycle position as a factor for commitment"""
        # This would integrate with cell cycle timing engine
        return 0.5
    
    def _calculate_neighbor_influence(self, cell: NeuroepithelialCell) -> float:
        """Calculate influence of neighboring cells on fate commitment"""
        # This would integrate with cell positioning and adhesion systems
        return 0.0
    
    def _calculate_temporal_factors(self, state: FateCommitmentState) -> float:
        """Calculate temporal factors affecting commitment"""
        # Time in stage affects commitment
        time_factor = min(1.0, state.time_in_stage / 5.0)
        return time_factor
    
    def _get_next_stage(self, current_stage: FateCommitmentStage) -> Optional[FateCommitmentStage]:
        """Get next stage in fate commitment progression"""
        stage_progression = {
            FateCommitmentStage.PROGENITOR: FateCommitmentStage.COMMITTED,
            FateCommitmentStage.COMMITTED: FateCommitmentStage.DIFFERENTIATING,
            FateCommitmentStage.DIFFERENTIATING: FateCommitmentStage.DIFFERENTIATED,
            FateCommitmentStage.DIFFERENTIATED: None  # Terminal stage
        }
        return stage_progression.get(current_stage)
    
    def _get_transition_type(self, from_stage: FateCommitmentStage, to_stage: FateCommitmentStage) -> FateTransitionType:
        """Get transition type for stage change"""
        if from_stage == FateCommitmentStage.PROGENITOR and to_stage == FateCommitmentStage.COMMITTED:
            return FateTransitionType.PROGENITOR_TO_COMMITTED
        elif from_stage == FateCommitmentStage.COMMITTED and to_stage == FateCommitmentStage.DIFFERENTIATING:
            return FateTransitionType.COMMITTED_TO_DIFFERENTIATING
        elif from_stage == FateCommitmentStage.DIFFERENTIATING and to_stage == FateCommitmentStage.DIFFERENTIATED:
            return FateTransitionType.DIFFERENTIATING_TO_DIFFERENTIATED
        else:
            return FateTransitionType.REVERSAL
    
    def _update_cell_state(self, cell: NeuroepithelialCell, new_stage: FateCommitmentStage) -> None:
        """Update cell state based on new fate stage"""
        stage_to_type = {
            FateCommitmentStage.PROGENITOR: NeuroepithelialCellType.EARLY_MULTIPOTENT,
            FateCommitmentStage.COMMITTED: NeuroepithelialCellType.LATE_MULTIPOTENT,
            FateCommitmentStage.DIFFERENTIATING: NeuroepithelialCellType.COMMITTED_PROGENITOR,
            FateCommitmentStage.DIFFERENTIATED: NeuroepithelialCellType.TRANSITIONING
        }
        
        if new_stage in stage_to_type:
            cell.cell_type = stage_to_type[new_stage]
