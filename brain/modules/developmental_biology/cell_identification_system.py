#!/usr/bin/env python3
"""Cell Identification System.

Main coordinator for marker-based cell identification including cell type
classification, developmental state assessment, and fate prediction based
on molecular marker expression patterns.

Integration: Main identification coordinator for developmental biology
Rationale: Main identification coordinator with focused responsibilities
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .neuroepithelial_cell_types import NeuroepithelialCellType
from .competency_window_types import FateType
from .cell_type_classifier import CellTypeClassifier

logger = logging.getLogger(__name__)

class CellIdentificationSystem:
    """System for marker-based cell identification and classification.
    
    Main coordinator for comprehensive cell identification based on
    molecular marker expression patterns including developmental state
    assessment and fate prediction capabilities.
    """
    
    def __init__(self):
        """Initialize cell identification system."""
        self.cell_classifier = CellTypeClassifier()
        
        logger.info("Initialized CellIdentificationSystem")
    
    def identify_cell_type(self, marker_levels: Dict[str, float]) -> Tuple[NeuroepithelialCellType, float]:
        """Identify cell type from marker expression pattern."""
        return self.cell_classifier.classify_cell_type(marker_levels)
    
    def assess_developmental_state(self, marker_levels: Dict[str, float],
                                 developmental_time: float) -> Dict[str, any]:
        """Assess developmental state from marker expression."""
        # Identify cell type
        predicted_type, confidence = self.identify_cell_type(marker_levels)
        
        # Assess proliferation state
        proliferation_markers = ['PCNA', 'Ki67']
        proliferation_score = np.mean([marker_levels.get(marker, 0.0) 
                                     for marker in proliferation_markers])
        
        # Assess progenitor state
        progenitor_markers = ['Nestin', 'Sox2']
        progenitor_score = np.mean([marker_levels.get(marker, 0.0) 
                                  for marker in progenitor_markers])
        
        # Assess commitment state
        commitment_markers = ['Tbr2']
        commitment_score = np.mean([marker_levels.get(marker, 0.0) 
                                  for marker in commitment_markers])
        
        # Assess temporal appropriateness
        temporal_appropriateness = self._assess_temporal_appropriateness(
            marker_levels, developmental_time)
        
        assessment = {
            'predicted_cell_type': predicted_type,
            'identification_confidence': confidence,
            'proliferation_state': {
                'score': proliferation_score,
                'status': 'high' if proliferation_score > 0.6 else 'moderate' if proliferation_score > 0.3 else 'low'
            },
            'progenitor_state': {
                'score': progenitor_score,
                'status': 'strong' if progenitor_score > 0.7 else 'moderate' if progenitor_score > 0.4 else 'weak'
            },
            'commitment_state': {
                'score': commitment_score,
                'status': 'committed' if commitment_score > 0.4 else 'uncommitted'
            },
            'temporal_appropriateness': temporal_appropriateness,
            'developmental_time': developmental_time
        }
        
        return assessment
    
    def _assess_temporal_appropriateness(self, marker_levels: Dict[str, float],
                                       developmental_time: float) -> Dict[str, any]:
        """Assess if marker expression is appropriate for developmental time."""
        # Expected marker levels at this developmental time
        expected_high = []
        expected_low = []
        
        if developmental_time < 9.0:
            # Early stage
            expected_high = ['Nestin', 'Sox2', 'Ki67']
            expected_low = ['Tbr2']
        elif developmental_time < 11.0:
            # Middle stage
            expected_high = ['Nestin', 'Pax6', 'PCNA']
            expected_low = []
        else:
            # Late stage
            expected_high = ['Tbr2', 'Pax6']
            expected_low = ['Sox2', 'Ki67']
        
        # Check appropriateness
        appropriate_high = sum(1 for marker in expected_high 
                             if marker_levels.get(marker, 0.0) > 0.5)
        appropriate_low = sum(1 for marker in expected_low 
                            if marker_levels.get(marker, 0.0) < 0.3)
        
        total_expectations = len(expected_high) + len(expected_low)
        appropriate_count = appropriate_high + appropriate_low
        
        appropriateness_score = appropriate_count / max(1, total_expectations)
        
        return {
            'score': appropriateness_score,
            'expected_high': expected_high,
            'expected_low': expected_low,
            'appropriate_markers': appropriate_count,
            'total_expectations': total_expectations
        }
    
    def predict_fate_commitment(self, marker_levels: Dict[str, float],
                               competency_levels: Dict[FateType, float]) -> Dict[str, any]:
        """Predict fate commitment based on markers and competency."""
        # Combine marker evidence with competency
        fate_evidence = {}
        
        for fate_type, competency in competency_levels.items():
            # Get marker evidence for this fate
            marker_evidence = self._get_marker_evidence_for_fate(marker_levels, fate_type)
            
            # Combined evidence (competency × marker evidence)
            combined_evidence = competency * marker_evidence
            fate_evidence[fate_type] = combined_evidence
        
        # Find strongest evidence
        if fate_evidence:
            strongest_fate = max(fate_evidence.items(), key=lambda x: x[1])
            commitment_strength = strongest_fate[1]
            
            # Predict commitment if evidence is strong
            commitment_predicted = commitment_strength > 0.6
            
            return {
                'commitment_predicted': commitment_predicted,
                'predicted_fate': strongest_fate[0] if commitment_predicted else None,
                'commitment_strength': commitment_strength,
                'fate_evidence': fate_evidence
            }
        
        return {
            'commitment_predicted': False,
            'predicted_fate': None,
            'commitment_strength': 0.0,
            'fate_evidence': {}
        }
    
    def _get_marker_evidence_for_fate(self, marker_levels: Dict[str, float], 
                                    fate_type: FateType) -> float:
        """Get marker evidence supporting specific fate."""
        # Simplified fate-marker associations
        fate_marker_weights = {
            FateType.MOTOR_NEURON: {'Pax6': 0.5, 'Tbr2': 0.3},
            FateType.INTERNEURON_V0: {'Pax6': 0.6, 'Sox2': 0.3},
            FateType.INTERNEURON_DORSAL: {'Pax6': 0.7, 'Tbr2': 0.4},
            FateType.NEURAL_CREST: {'Pax6': 0.3},
            FateType.OLIGODENDROCYTE: {'Tbr2': 0.5},
            FateType.ASTROCYTE: {'Tbr2': 0.3}
        }
        
        if fate_type not in fate_marker_weights:
            return 0.5
        
        weights = fate_marker_weights[fate_type]
        total_evidence = 0.0
        
        for marker, weight in weights.items():
            if marker in marker_levels:
                total_evidence += weight * marker_levels[marker]
        
        return total_evidence