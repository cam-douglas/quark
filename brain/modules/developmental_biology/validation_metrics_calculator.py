"""
Validation Metrics Calculator

This module calculates validation metrics for lineage commitment validation
including accuracy, consistency, and fidelity calculations.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .neuroepithelial_cells import NeuroepithelialCell
from .lineage_tree_builder import LineageTreeBuilder
from .cell_fate_decision_integrator import FateCommitmentState
from .apoptosis_manager import ApoptosisManager


class ValidationMetric(Enum):
    """Types of validation metrics"""
    COMMITMENT_ACCURACY = "commitment_accuracy"
    FATE_CONSISTENCY = "fate_consistency"
    TEMPORAL_ACCURACY = "temporal_accuracy"
    SPATIAL_ACCURACY = "spatial_accuracy"
    LINEAGE_FIDELITY = "lineage_fidelity"


@dataclass
class ValidationResult:
    """Result of lineage commitment validation"""
    cell_id: str
    metric_type: ValidationMetric
    predicted_fate: str
    experimental_fate: str
    accuracy_score: float
    confidence: float
    validation_timestamp: float


class ValidationMetricsCalculator:
    """
    Calculates validation metrics for lineage commitment validation
    including accuracy, consistency, and fidelity calculations.
    """
    
    def __init__(self):
        """Initialize validation metrics calculator"""
        self.validation_parameters: Dict[str, Dict[str, float]] = {}
        self._setup_validation_parameters()
    
    def _setup_validation_parameters(self) -> None:
        """Setup validation parameters for different metrics"""
        self.validation_parameters = {
            "commitment_accuracy": {
                "threshold": 0.8,
                "weight": 1.0,
                "tolerance": 0.1
            },
            "fate_consistency": {
                "threshold": 0.7,
                "weight": 0.8,
                "tolerance": 0.15
            },
            "temporal_accuracy": {
                "threshold": 0.6,
                "weight": 0.6,
                "tolerance": 0.2
            },
            "spatial_accuracy": {
                "threshold": 0.75,
                "weight": 0.7,
                "tolerance": 0.12
            },
            "lineage_fidelity": {
                "threshold": 0.85,
                "weight": 0.9,
                "tolerance": 0.08
            }
        }
    
    def calculate_commitment_accuracy(self, cell: NeuroepithelialCell, 
                                    commitment_state: FateCommitmentState) -> Optional[ValidationResult]:
        """Calculate commitment accuracy validation"""
        # Get predicted fate
        predicted_fate = commitment_state.target_fate
        
        # Get experimental fate (placeholder - would use real data)
        experimental_fate = self._get_experimental_fate(cell.cell_id)
        
        # Calculate accuracy score
        accuracy_score = self._calculate_fate_accuracy(predicted_fate, experimental_fate)
        
        # Get validation parameters
        params = self.validation_parameters["commitment_accuracy"]
        
        # Calculate confidence
        confidence = self._calculate_validation_confidence(accuracy_score, params)
        
        return ValidationResult(
            cell_id=cell.cell_id,
            metric_type=ValidationMetric.COMMITMENT_ACCURACY,
            predicted_fate=predicted_fate,
            experimental_fate=experimental_fate,
            accuracy_score=accuracy_score,
            confidence=confidence,
            validation_timestamp=0.0
        )
    
    def calculate_fate_consistency(self, cell: NeuroepithelialCell, 
                                 commitment_state: FateCommitmentState,
                                 lineage_tree: Optional[LineageTreeBuilder]) -> Optional[ValidationResult]:
        """Calculate fate consistency validation"""
        if not lineage_tree:
            return None
        
        # Get predicted fate
        predicted_fate = commitment_state.target_fate
        
        # Check consistency with lineage tree
        lineage_fate = self._get_lineage_fate(lineage_tree, cell.cell_id)
        
        # Calculate consistency score
        consistency_score = self._calculate_fate_consistency(predicted_fate, lineage_fate)
        
        # Get validation parameters
        params = self.validation_parameters["fate_consistency"]
        
        # Calculate confidence
        confidence = self._calculate_validation_confidence(consistency_score, params)
        
        return ValidationResult(
            cell_id=cell.cell_id,
            metric_type=ValidationMetric.FATE_CONSISTENCY,
            predicted_fate=predicted_fate,
            experimental_fate=lineage_fate,
            accuracy_score=consistency_score,
            confidence=confidence,
            validation_timestamp=0.0
        )
    
    def calculate_temporal_accuracy(self, cell: NeuroepithelialCell, 
                                  commitment_state: FateCommitmentState) -> ValidationResult:
        """Calculate temporal accuracy validation"""
        # Get predicted timing
        predicted_timing = commitment_state.time_in_stage
        
        # Get experimental timing (placeholder)
        experimental_timing = self._get_experimental_timing(cell.cell_id)
        
        # Calculate temporal accuracy
        temporal_accuracy = self._calculate_temporal_accuracy(predicted_timing, experimental_timing)
        
        # Get validation parameters
        params = self.validation_parameters["temporal_accuracy"]
        
        # Calculate confidence
        confidence = self._calculate_validation_confidence(temporal_accuracy, params)
        
        return ValidationResult(
            cell_id=cell.cell_id,
            metric_type=ValidationMetric.TEMPORAL_ACCURACY,
            predicted_fate=f"timing_{predicted_timing:.2f}",
            experimental_fate=f"timing_{experimental_timing:.2f}",
            accuracy_score=temporal_accuracy,
            confidence=confidence,
            validation_timestamp=0.0
        )
    
    def calculate_spatial_accuracy(self, cell: NeuroepithelialCell, 
                                 commitment_state: FateCommitmentState) -> ValidationResult:
        """Calculate spatial accuracy validation"""
        # Get predicted spatial location
        predicted_location = self._get_cell_spatial_location(cell)
        
        # Get experimental spatial location (placeholder)
        experimental_location = self._get_experimental_spatial_location(cell.cell_id)
        
        # Calculate spatial accuracy
        spatial_accuracy = self._calculate_spatial_accuracy(predicted_location, experimental_location)
        
        # Get validation parameters
        params = self.validation_parameters["spatial_accuracy"]
        
        # Calculate confidence
        confidence = self._calculate_validation_confidence(spatial_accuracy, params)
        
        return ValidationResult(
            cell_id=cell.cell_id,
            metric_type=ValidationMetric.SPATIAL_ACCURACY,
            predicted_fate=f"location_{predicted_location}",
            experimental_fate=f"location_{experimental_location}",
            accuracy_score=spatial_accuracy,
            confidence=confidence,
            validation_timestamp=0.0
        )
    
    def calculate_lineage_fidelity(self, cell: NeuroepithelialCell, 
                                 commitment_state: FateCommitmentState,
                                 lineage_tree: Optional[LineageTreeBuilder]) -> Optional[ValidationResult]:
        """Calculate lineage fidelity validation"""
        if not lineage_tree:
            return None
        
        # Get predicted lineage path
        predicted_lineage = self._get_predicted_lineage_path(commitment_state)
        
        # Get experimental lineage path
        experimental_lineage = self._get_experimental_lineage_path(cell.cell_id)
        
        # Calculate lineage fidelity
        lineage_fidelity = self._calculate_lineage_fidelity(predicted_lineage, experimental_lineage)
        
        # Get validation parameters
        params = self.validation_parameters["lineage_fidelity"]
        
        # Calculate confidence
        confidence = self._calculate_validation_confidence(lineage_fidelity, params)
        
        return ValidationResult(
            cell_id=cell.cell_id,
            metric_type=ValidationMetric.LINEAGE_FIDELITY,
            predicted_fate=f"lineage_{predicted_lineage}",
            experimental_fate=f"lineage_{experimental_lineage}",
            accuracy_score=lineage_fidelity,
            confidence=confidence,
            validation_timestamp=0.0
        )
    
    def calculate_cell_count_variance(self, cell_counts: List[int], apoptosis_manager: Optional[ApoptosisManager] = None) -> float:
        """Compute variance in cell counts across time or regions; adjust for apoptosis if provided."""
        if not cell_counts:
            return 0.0
        adjusted = list(cell_counts)
        if apoptosis_manager is not None and sum(cell_counts) > 0:
            frac = apoptosis_manager.get_apoptosis_fraction(total_cells=max(cell_counts))
            adjusted = [max(0, int(c * (1 - frac))) for c in cell_counts]
        return float(np.var(adjusted) / (np.mean(adjusted) + 1e-9))
    
    def _get_experimental_fate(self, cell_id: str) -> str:
        """Get experimental fate for cell (placeholder)"""
        return "experimental_fate"
    
    def _calculate_fate_accuracy(self, predicted_fate: str, experimental_fate: str) -> float:
        """Calculate fate accuracy score"""
        if predicted_fate == experimental_fate:
            return 1.0
        else:
            return 0.0
    
    def _get_lineage_fate(self, lineage_tree: LineageTreeBuilder, cell_id: str) -> str:
        """Get fate from lineage tree"""
        return "lineage_fate"
    
    def _calculate_fate_consistency(self, predicted_fate: str, lineage_fate: str) -> float:
        """Calculate fate consistency score"""
        if predicted_fate == lineage_fate:
            return 1.0
        else:
            return 0.0
    
    def _get_experimental_timing(self, cell_id: str) -> float:
        """Get experimental timing (placeholder)"""
        return 2.5  # hours
    
    def _calculate_temporal_accuracy(self, predicted_timing: float, experimental_timing: float) -> float:
        """Calculate temporal accuracy score"""
        if experimental_timing == 0:
            return 1.0 if predicted_timing == 0 else 0.0
        
        relative_error = abs(predicted_timing - experimental_timing) / experimental_timing
        return max(0.0, 1.0 - relative_error)
    
    def _get_cell_spatial_location(self, cell: NeuroepithelialCell) -> Tuple[float, float, float]:
        """Get cell spatial location"""
        return (0.0, 0.0, 0.0)
    
    def _get_experimental_spatial_location(self, cell_id: str) -> Tuple[float, float, float]:
        """Get experimental spatial location (placeholder)"""
        return (0.1, 0.1, 0.1)
    
    def _calculate_spatial_accuracy(self, predicted_location: Tuple[float, float, float], 
                                  experimental_location: Tuple[float, float, float]) -> float:
        """Calculate spatial accuracy score"""
        distance = np.sqrt(sum((p - e) ** 2 for p, e in zip(predicted_location, experimental_location)))
        max_distance = 1.0
        accuracy = max(0.0, 1.0 - distance / max_distance)
        return accuracy
    
    def _get_predicted_lineage_path(self, commitment_state: FateCommitmentState) -> str:
        """Get predicted lineage path"""
        return f"path_{commitment_state.target_fate}"
    
    def _get_experimental_lineage_path(self, cell_id: str) -> str:
        """Get experimental lineage path (placeholder)"""
        return "experimental_path"
    
    def _calculate_lineage_fidelity(self, predicted_lineage: str, experimental_lineage: str) -> float:
        """Calculate lineage fidelity score"""
        if predicted_lineage == experimental_lineage:
            return 1.0
        else:
            return 0.0
    
    def _calculate_validation_confidence(self, accuracy_score: float, params: Dict[str, float]) -> float:
        """Calculate validation confidence"""
        threshold = params.get("threshold", 0.5)
        tolerance = params.get("tolerance", 0.1)
        
        if accuracy_score >= threshold:
            confidence = 1.0
        else:
            distance = threshold - accuracy_score
            confidence = max(0.0, 1.0 - distance / tolerance)
        
        return confidence
