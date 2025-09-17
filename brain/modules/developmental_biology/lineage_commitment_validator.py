"""
Lineage Commitment Validator

This module validates lineage commitment accuracy, compares with experimental
fate mapping data, and implements commitment reversal mechanisms.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .neuroepithelial_cells import NeuroepithelialCell
from .lineage_tree_builder import LineageTreeBuilder
from .cell_fate_decision_integrator import FateCommitmentState
from .validation_metrics_calculator import ValidationMetricsCalculator, ValidationResult, ValidationMetric
from .reversal_mechanisms_manager import ReversalMechanismsManager, ReversalEvent


@dataclass
class ValidationStatistics:
    """Statistics for validation results"""
    total_validations: int
    average_accuracy: float
    metric_breakdown: Dict[ValidationMetric, float]
    fate_accuracy: Dict[str, float]
    reversal_events: int


class LineageCommitmentValidator:
    """
    Validates lineage commitment accuracy, compares with experimental
    fate mapping data, and implements commitment reversal mechanisms.
    """
    
    def __init__(self):
        """Initialize lineage commitment validator"""
        self.validation_results: List[ValidationResult] = []
        self.reversal_events: List[ReversalEvent] = []
        self.metrics_calculator = ValidationMetricsCalculator()
        self.reversal_manager = ReversalMechanismsManager()
    
    
    def validate_lineage_commitment(self, cells: Dict[str, NeuroepithelialCell], 
                                  commitment_states: Dict[str, FateCommitmentState],
                                  lineage_trees: Dict[str, LineageTreeBuilder]) -> List[ValidationResult]:
        """Validate lineage commitment accuracy using metrics calculator"""
        validation_results = []
        
        for cell_id, cell in cells.items():
            if cell_id not in commitment_states:
                continue
            
            commitment_state = commitment_states[cell_id]
            lineage_tree = lineage_trees.get(cell_id)
            
            # Validate commitment accuracy
            accuracy_result = self.metrics_calculator.calculate_commitment_accuracy(cell, commitment_state)
            if accuracy_result:
                validation_results.append(accuracy_result)
            
            # Validate fate consistency
            consistency_result = self.metrics_calculator.calculate_fate_consistency(cell, commitment_state, lineage_tree)
            if consistency_result:
                validation_results.append(consistency_result)
            
            # Validate temporal accuracy
            temporal_result = self.metrics_calculator.calculate_temporal_accuracy(cell, commitment_state)
            validation_results.append(temporal_result)
            
            # Validate spatial accuracy
            spatial_result = self.metrics_calculator.calculate_spatial_accuracy(cell, commitment_state)
            validation_results.append(spatial_result)
            
            # Validate lineage fidelity
            fidelity_result = self.metrics_calculator.calculate_lineage_fidelity(cell, commitment_state, lineage_tree)
            if fidelity_result:
                validation_results.append(fidelity_result)
        
        # Store results
        self.validation_results.extend(validation_results)
        
        return validation_results
    
    
    
    def implement_commitment_reversal(self, cells: Dict[str, NeuroepithelialCell], 
                                    commitment_states: Dict[str, FateCommitmentState],
                                    validation_results: List[ValidationResult]) -> List[ReversalEvent]:
        """Implement commitment reversal using reversal manager"""
        reversal_events = []
        
        for result in validation_results:
            # Create reversal event if warranted
            reversal_event = self.reversal_manager.create_reversal_event(result.cell_id, result)
            if reversal_event:
                reversal_events.append(reversal_event)
                
                # Apply reversal to cell and commitment state
                if result.cell_id in cells and result.cell_id in commitment_states:
                    self.reversal_manager.apply_reversal(
                        cells[result.cell_id], 
                        commitment_states[result.cell_id], 
                        reversal_event
                    )
        
        # Store events
        self.reversal_events.extend(reversal_events)
        
        return reversal_events
    
    def get_validation_statistics(self, cell_ids: List[str]) -> ValidationStatistics:
        """Get validation statistics for multiple cells"""
        if not cell_ids:
            return ValidationStatistics(0, 0.0, {}, {}, 0)
        
        # Filter results for specified cells
        cell_results = [r for r in self.validation_results if r.cell_id in cell_ids]
        
        if not cell_results:
            return ValidationStatistics(0, 0.0, {}, {}, 0)
        
        # Calculate statistics
        total_validations = len(cell_results)
        average_accuracy = np.mean([r.accuracy_score for r in cell_results])
        
        # Metric breakdown
        metric_breakdown = {}
        for metric in ValidationMetric:
            metric_results = [r for r in cell_results if r.metric_type == metric]
            if metric_results:
                metric_breakdown[metric] = np.mean([r.accuracy_score for r in metric_results])
        
        # Fate accuracy
        fate_accuracy = {}
        for result in cell_results:
            fate = result.predicted_fate
            if fate not in fate_accuracy:
                fate_accuracy[fate] = []
            fate_accuracy[fate].append(result.accuracy_score)
        
        # Calculate average accuracy per fate
        for fate in fate_accuracy:
            fate_accuracy[fate] = np.mean(fate_accuracy[fate])
        
        # Count reversal events
        reversal_events = len([e for e in self.reversal_events if e.cell_id in cell_ids])
        
        return ValidationStatistics(
            total_validations=total_validations,
            average_accuracy=average_accuracy,
            metric_breakdown=metric_breakdown,
            fate_accuracy=fate_accuracy,
            reversal_events=reversal_events
        )
    
    
    
