"""
Tag Validation Manager

This module validates tag fidelity and identifies discrepancies
between expected and actual barcodes.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from .lineage_barcode_types import LineageTag


@dataclass
class TagValidationResult:
    """Result of tag fidelity validation"""
    cell_id: str
    expected_barcode: LineageTag
    actual_barcode: LineageTag
    fidelity_score: float
    validation_passed: bool
    discrepancies: List[str]


class TagValidationManager:
    """
    Validates tag fidelity and identifies discrepancies
    between expected and actual barcodes.
    """
    
    def __init__(self):
        """Initialize tag validation manager"""
        self.validation_threshold = 0.6  # 60% fidelity threshold
    
    def validate_tag_fidelity(self, cell_id: str, expected_barcode: LineageTag, 
                            actual_barcode: LineageTag, fidelity_score: float) -> TagValidationResult:
        """Validate tag fidelity for a single cell"""
        # Identify discrepancies
        discrepancies = self._identify_discrepancies(expected_barcode, actual_barcode)
        
        # Determine if validation passed
        validation_passed = fidelity_score >= self.validation_threshold
        
        return TagValidationResult(
            cell_id=cell_id,
            expected_barcode=expected_barcode,
            actual_barcode=actual_barcode,
            fidelity_score=fidelity_score,
            validation_passed=validation_passed,
            discrepancies=discrepancies
        )
    
    def _identify_discrepancies(self, original: LineageTag, current: LineageTag) -> List[str]:
        """Identify discrepancies between original and current lineage tags"""
        discrepancies = []
        
        if original.cell_id != current.cell_id:
            discrepancies.append("Cell ID mismatch")
        
        if original.generation != current.generation:
            discrepancies.append("Generation mismatch")
        
        if original.lineage_history != current.lineage_history:
            discrepancies.append("Lineage history mismatch")
        
        if original.creation_time != current.creation_time:
            discrepancies.append("Creation time mismatch")
        
        return discrepancies
    
    def calculate_validation_statistics(self, validation_results: List[TagValidationResult]) -> Dict[str, float]:
        """Calculate statistics from validation results"""
        if not validation_results:
            return {}
        
        total_validations = len(validation_results)
        passed_validations = sum(1 for result in validation_results if result.validation_passed)
        fidelity_scores = [result.fidelity_score for result in validation_results]
        
        # Count discrepancy types
        discrepancy_counts = {}
        for result in validation_results:
            for discrepancy in result.discrepancies:
                discrepancy_counts[discrepancy] = discrepancy_counts.get(discrepancy, 0) + 1
        
        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "validation_success_rate": passed_validations / total_validations,
            "average_fidelity": np.mean(fidelity_scores),
            "min_fidelity": np.min(fidelity_scores),
            "max_fidelity": np.max(fidelity_scores),
            "discrepancy_counts": discrepancy_counts
        }
