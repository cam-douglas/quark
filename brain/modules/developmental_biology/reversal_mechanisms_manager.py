"""
Reversal Mechanisms Manager

This module manages commitment reversal mechanisms including competency
reacquisition, fate reprogramming, dedifferentiation, and transdifferentiation.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType
from .cell_fate_decision_integrator import FateCommitmentState, FateCommitmentStage
from .validation_metrics_calculator import ValidationResult


class ReversalMechanism(Enum):
    """Types of commitment reversal mechanisms"""
    COMPETENCY_REACQUISITION = "competency_reacquisition"
    FATE_REPROGRAMMING = "fate_reprogramming"
    DEDifferentiation = "dedifferentiation"
    TRANSDifferentiation = "transdifferentiation"


@dataclass
class ReversalEvent:
    """Event representing commitment reversal"""
    cell_id: str
    reversal_mechanism: ReversalMechanism
    from_fate: str
    to_fate: str
    reversal_probability: float
    timestamp: float
    confidence: float


@dataclass
class ReversalParameters:
    """Parameters for reversal mechanisms"""
    success_probability: float
    time_to_completion: float
    energy_cost: float
    reversibility: float


class ReversalMechanismsManager:
    """
    Manages commitment reversal mechanisms including competency
    reacquisition, fate reprogramming, dedifferentiation, and transdifferentiation.
    """
    
    def __init__(self):
        """Initialize reversal mechanisms manager"""
        self.reversal_parameters: Dict[ReversalMechanism, ReversalParameters] = {}
        self._setup_reversal_parameters()
    
    def _setup_reversal_parameters(self) -> None:
        """Setup parameters for different reversal mechanisms"""
        self.reversal_parameters = {
            ReversalMechanism.COMPETENCY_REACQUISITION: ReversalParameters(
                success_probability=0.8,
                time_to_completion=2.0,
                energy_cost=0.3,
                reversibility=0.9
            ),
            ReversalMechanism.FATE_REPROGRAMMING: ReversalParameters(
                success_probability=0.6,
                time_to_completion=4.0,
                energy_cost=0.7,
                reversibility=0.6
            ),
            ReversalMechanism.DEDifferentiation: ReversalParameters(
                success_probability=0.4,
                time_to_completion=6.0,
                energy_cost=0.8,
                reversibility=0.4
            ),
            ReversalMechanism.TRANSDifferentiation: ReversalParameters(
                success_probability=0.3,
                time_to_completion=8.0,
                energy_cost=0.9,
                reversibility=0.2
            )
        }
    
    def determine_reversal_mechanism(self, validation_result: ValidationResult) -> ReversalMechanism:
        """Determine appropriate reversal mechanism based on validation result"""
        # Different metrics suggest different reversal mechanisms
        mechanism_mapping = {
            "commitment_accuracy": ReversalMechanism.FATE_REPROGRAMMING,
            "fate_consistency": ReversalMechanism.COMPETENCY_REACQUISITION,
            "temporal_accuracy": ReversalMechanism.DEDifferentiation,
            "spatial_accuracy": ReversalMechanism.TRANSDifferentiation,
            "lineage_fidelity": ReversalMechanism.FATE_REPROGRAMMING
        }
        
        metric_name = validation_result.metric_type.value
        return mechanism_mapping.get(metric_name, ReversalMechanism.FATE_REPROGRAMMING)
    
    def calculate_reversal_probability(self, validation_result: ValidationResult, 
                                     mechanism: ReversalMechanism) -> float:
        """Calculate probability of successful reversal"""
        # Get mechanism parameters
        params = self.reversal_parameters[mechanism]
        
        # Base probability from mechanism
        base_probability = params.success_probability
        
        # Adjust based on validation confidence
        confidence_factor = validation_result.confidence
        
        # Adjust based on accuracy score
        accuracy_factor = validation_result.accuracy_score
        
        # Combined probability
        reversal_probability = base_probability * confidence_factor * (1.0 - accuracy_factor)
        
        return min(1.0, max(0.0, reversal_probability))
    
    def determine_target_fate(self, validation_result: ValidationResult) -> str:
        """Determine target fate for reversal"""
        # Use experimental fate as target
        return validation_result.experimental_fate
    
    def should_reverse_commitment(self, validation_result: ValidationResult) -> bool:
        """Determine if commitment should be reversed based on validation result"""
        # Get validation parameters for this metric
        threshold = 0.5  # Default threshold
        
        # Check if accuracy is below threshold
        return validation_result.accuracy_score < threshold
    
    def create_reversal_event(self, cell_id: str, validation_result: ValidationResult) -> Optional[ReversalEvent]:
        """Create reversal event if reversal is warranted"""
        # Check if reversal is needed
        if not self.should_reverse_commitment(validation_result):
            return None
        
        # Determine reversal mechanism
        mechanism = self.determine_reversal_mechanism(validation_result)
        
        # Calculate reversal probability
        reversal_probability = self.calculate_reversal_probability(validation_result, mechanism)
        
        # Only create event if probability is above threshold
        if reversal_probability > 0.5:
            # Determine target fate
            target_fate = self.determine_target_fate(validation_result)
            
            return ReversalEvent(
                cell_id=cell_id,
                reversal_mechanism=mechanism,
                from_fate=validation_result.predicted_fate,
                to_fate=target_fate,
                reversal_probability=reversal_probability,
                timestamp=0.0,  # Would be set by calling system
                confidence=validation_result.confidence
            )
        
        return None
    
    def apply_reversal(self, cell: NeuroepithelialCell, commitment_state: FateCommitmentState, 
                      reversal_event: ReversalEvent) -> None:
        """Apply reversal to cell and commitment state"""
        # Update cell state based on mechanism
        if reversal_event.reversal_mechanism == ReversalMechanism.COMPETENCY_REACQUISITION:
            cell.cell_type = NeuroepithelialCellType.EARLY_MULTIPOTENT
            commitment_state.current_stage = FateCommitmentStage.PROGENITOR
        elif reversal_event.reversal_mechanism == ReversalMechanism.FATE_REPROGRAMMING:
            cell.cell_type = NeuroepithelialCellType.LATE_MULTIPOTENT
            commitment_state.current_stage = FateCommitmentStage.COMMITTED
        elif reversal_event.reversal_mechanism == ReversalMechanism.DEDifferentiation:
            cell.cell_type = NeuroepithelialCellType.EARLY_MULTIPOTENT
            commitment_state.current_stage = FateCommitmentStage.PROGENITOR
        elif reversal_event.reversal_mechanism == ReversalMechanism.TRANSDifferentiation:
            cell.cell_type = NeuroepithelialCellType.LATE_MULTIPOTENT
            commitment_state.current_stage = FateCommitmentStage.COMMITTED
        
        # Update target fate
        commitment_state.target_fate = reversal_event.to_fate
        
        # Reset commitment strength
        commitment_state.commitment_strength = 0.5  # Intermediate value
        
        # Reset time in stage
        commitment_state.time_in_stage = 0.0
    
    def get_reversal_statistics(self, reversal_events: List[ReversalEvent]) -> Dict[str, float]:
        """Get statistics for reversal events"""
        if not reversal_events:
            return {}
        
        # Count mechanisms
        mechanism_counts = {}
        for event in reversal_events:
            mechanism = event.reversal_mechanism.value
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        
        # Calculate average probabilities
        probabilities = [event.reversal_probability for event in reversal_events]
        confidences = [event.confidence for event in reversal_events]
        
        return {
            "total_reversals": len(reversal_events),
            "average_reversal_probability": np.mean(probabilities),
            "average_confidence": np.mean(confidences),
            "mechanism_distribution": mechanism_counts,
            "success_rate": np.mean(probabilities)  # Simplified success rate
        }
