#!/usr/bin/env python3
"""Cell Type Classifier.

Cell type classification logic for neuroepithelial cells based on
molecular marker expression patterns including confidence scoring
and cell type prediction algorithms.

Integration: Classification component for cell identification system
Rationale: Focused classification logic separated from identification system
"""

from typing import Dict, Tuple
import numpy as np
import logging

from .neuroepithelial_cell_types import NeuroepithelialCellType

logger = logging.getLogger(__name__)

class CellTypeClassifier:
    """Classifier for neuroepithelial cell types based on markers.
    
    Provides cell type classification based on molecular marker
    expression patterns with confidence scoring and validation.
    """
    
    def __init__(self):
        """Initialize cell type classifier."""
        self.identification_thresholds = self._initialize_identification_thresholds()
        logger.info("Initialized CellTypeClassifier")
    
    def _initialize_identification_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize thresholds for cell type identification."""
        return {
            'early_multipotent': {
                'Nestin': 0.7,      # High Nestin
                'Sox2': 0.6,        # High Sox2
                'Pax6': 0.5,        # Moderate Pax6
                'Ki67': 0.5,        # Active proliferation
                'Tbr2': 0.1        # Very low Tbr2
            },
            'late_multipotent': {
                'Nestin': 0.5,      # Moderate Nestin
                'Sox2': 0.4,        # Moderate Sox2
                'Pax6': 0.6,        # High Pax6
                'Ki67': 0.3,        # Reduced proliferation
                'Tbr2': 0.2        # Low Tbr2
            },
            'committed_progenitor': {
                'Nestin': 0.3,      # Reduced Nestin
                'Sox2': 0.3,        # Reduced Sox2
                'Pax6': 0.4,        # Moderate Pax6
                'Ki67': 0.4,        # Moderate proliferation
                'Tbr2': 0.5        # High Tbr2 (key marker)
            },
            'transitioning': {
                'Nestin': 0.4,      # Intermediate
                'Sox2': 0.4,        # Intermediate
                'Pax6': 0.5,        # Maintained
                'Ki67': 0.3,        # Variable
                'Tbr2': 0.3        # Intermediate
            }
        }
    
    def classify_cell_type(self, marker_levels: Dict[str, float]) -> Tuple[NeuroepithelialCellType, float]:
        """Classify cell type from marker expression pattern.
        
        Args:
            marker_levels: Current marker expression levels
            
        Returns:
            Tuple of (predicted_cell_type, confidence_score)
        """
        type_scores = {}
        
        # Calculate scores for each cell type
        for cell_type_str, thresholds in self.identification_thresholds.items():
            score = self._calculate_type_score(marker_levels, thresholds)
            
            # Convert string to enum
            if cell_type_str == 'early_multipotent':
                cell_type = NeuroepithelialCellType.EARLY_MULTIPOTENT
            elif cell_type_str == 'late_multipotent':
                cell_type = NeuroepithelialCellType.LATE_MULTIPOTENT
            elif cell_type_str == 'committed_progenitor':
                cell_type = NeuroepithelialCellType.COMMITTED_PROGENITOR
            else:
                cell_type = NeuroepithelialCellType.TRANSITIONING
            
            type_scores[cell_type] = score
        
        # Find best match
        best_type = max(type_scores.items(), key=lambda x: x[1])
        
        return best_type[0], best_type[1]
    
    def _calculate_type_score(self, marker_levels: Dict[str, float], 
                            thresholds: Dict[str, float]) -> float:
        """Calculate type score based on marker thresholds."""
        total_score = 0.0
        marker_count = 0
        
        for marker, threshold in thresholds.items():
            if marker in marker_levels:
                marker_level = marker_levels[marker]
                
                # Score based on how well marker level matches expectation
                if marker_level >= threshold:
                    # Above threshold - good match
                    score = 1.0
                else:
                    # Below threshold - partial match
                    score = marker_level / threshold
                
                total_score += score
                marker_count += 1
        
        # Average score across all markers
        avg_score = total_score / max(1, marker_count)
        
        return avg_score
    
    def validate_classification(self, marker_levels: Dict[str, float],
                               predicted_type: NeuroepithelialCellType,
                               confidence: float) -> Dict[str, any]:
        """Validate cell type classification.
        
        Args:
            marker_levels: Marker expression levels
            predicted_type: Predicted cell type
            confidence: Classification confidence
            
        Returns:
            Validation results
        """
        validation_results = {
            'classification_valid': confidence > 0.7,
            'confidence_acceptable': confidence > 0.5,
            'marker_consistency': True,
            'temporal_plausibility': True
        }
        
        # Check marker consistency
        type_str = predicted_type.value.replace('_multipotent', '_multipotent').replace('_progenitor', '_progenitor')
        if type_str in self.identification_thresholds:
            thresholds = self.identification_thresholds[type_str]
            
            # Check key markers
            for marker, threshold in thresholds.items():
                if marker in marker_levels:
                    if marker_levels[marker] < threshold * 0.5:  # Below 50% of threshold
                        validation_results['marker_consistency'] = False
                        break
        
        return validation_results
