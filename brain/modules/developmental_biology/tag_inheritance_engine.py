"""
Tag Inheritance Engine

This module handles the inheritance of lineage tags during cell division
with different inheritance modes and fidelity calculations.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from .lineage_barcode_types import LineageTag


class TagInheritanceMode(Enum):
    """Modes of tag inheritance during differentiation"""
    FAITHFUL = "faithful"
    DILUTION = "dilution"
    SELECTIVE = "selective"
    STOCHASTIC = "stochastic"


class TagFidelityLevel(Enum):
    """Levels of tag fidelity"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEGRADED = "degraded"


@dataclass
class TagInheritanceEvent:
    """Event representing tag inheritance during cell division"""
    parent_cell_id: str
    daughter_cell_ids: List[str]
    inheritance_mode: TagInheritanceMode
    fidelity_scores: Dict[str, float]
    timestamp: float


class TagInheritanceEngine:
    """
    Handles the inheritance of lineage tags during cell division
    with different inheritance modes and fidelity calculations.
    """
    
    def __init__(self):
        """Initialize tag inheritance engine"""
        self.preservation_parameters: Dict[str, Dict[str, float]] = {}
        self._setup_preservation_parameters()
    
    def _setup_preservation_parameters(self) -> None:
        """Setup preservation parameters for different inheritance modes"""
        self.preservation_parameters = {
            "faithful": {
                "fidelity_rate": 0.99,
                "degradation_rate": 0.001,
                "mutation_probability": 0.0001,
                "dilution_factor": 0.0
            },
            "dilution": {
                "fidelity_rate": 0.85,
                "degradation_rate": 0.05,
                "mutation_probability": 0.01,
                "dilution_factor": 0.1
            },
            "selective": {
                "fidelity_rate": 0.90,
                "degradation_rate": 0.02,
                "mutation_probability": 0.005,
                "dilution_factor": 0.05
            },
            "stochastic": {
                "fidelity_rate": 0.70,
                "degradation_rate": 0.1,
                "mutation_probability": 0.02,
                "dilution_factor": 0.15
            }
        }
    
    def inherit_barcode(self, parent_barcode: LineageTag, 
                       inheritance_mode: TagInheritanceMode) -> LineageTag:
        """Inherit barcode from parent with mode-specific modifications"""
        params = self.preservation_parameters[inheritance_mode.value]
        
        # Start with perfect copy
        inherited_barcode = self._copy_barcode(parent_barcode)
        
        # Apply inheritance-specific modifications
        if inheritance_mode == TagInheritanceMode.FAITHFUL:
            if np.random.random() < params["mutation_probability"]:
                inherited_barcode = self._apply_random_mutation(inherited_barcode)
        
        elif inheritance_mode == TagInheritanceMode.DILUTION:
            inherited_barcode = self._apply_dilution(inherited_barcode, params["dilution_factor"])
        
        elif inheritance_mode == TagInheritanceMode.SELECTIVE:
            inherited_barcode = self._apply_selective_preservation(inherited_barcode)
        
        elif inheritance_mode == TagInheritanceMode.STOCHASTIC:
            inherited_barcode = self._apply_stochastic_modifications(inherited_barcode, params)
        
        # Update generation
        inherited_barcode.generation = parent_barcode.generation + 1
        
        return inherited_barcode
    
    def _copy_barcode(self, original: LineageTag) -> LineageTag:
        """Create a copy of a lineage tag"""
        return LineageTag(
            cell_id=original.cell_id,
            generation=original.generation,
            lineage_barcodes=original.lineage_barcodes.copy(),
            lineage_history=original.lineage_history.copy(),
            creation_time=original.creation_time,
            last_update_time=original.last_update_time
        )
    
    def _apply_random_mutation(self, tag: LineageTag) -> LineageTag:
        """Apply random mutation to lineage tag"""
        # Modify cell ID slightly to simulate mutation
        tag.cell_id = f"{tag.cell_id}_mut"
        tag.last_update_time = tag.last_update_time + 0.1
        return tag
    
    def _apply_dilution(self, tag: LineageTag, dilution_factor: float) -> LineageTag:
        """Apply dilution to lineage tag"""
        if np.random.random() < dilution_factor:
            tag.cell_id = f"{tag.cell_id}_diluted"
            tag.last_update_time = tag.last_update_time + 0.1
        return tag
    
    def _apply_selective_preservation(self, tag: LineageTag) -> LineageTag:
        """Apply selective preservation to lineage tag"""
        if np.random.random() < 0.3:
            tag.cell_id = f"{tag.cell_id}_selective"
            tag.last_update_time = tag.last_update_time + 0.1
        return tag
    
    def _apply_stochastic_modifications(self, tag: LineageTag, 
                                      params: Dict[str, float]) -> LineageTag:
        """Apply stochastic modifications to lineage tag"""
        if np.random.random() < params["mutation_probability"]:
            tag = self._apply_random_mutation(tag)
        if np.random.random() < params["dilution_factor"]:
            tag = self._apply_dilution(tag, params["dilution_factor"])
        return tag
    
    def calculate_fidelity_score(self, original: LineageTag, current: LineageTag) -> float:
        """Calculate fidelity score between original and current lineage tags"""
        # Simple fidelity based on cell ID similarity and generation difference
        cell_id_similarity = self._calculate_sequence_similarity(original.cell_id, current.cell_id)
        
        # Generation difference factor
        generation_diff = abs(original.generation - current.generation)
        generation_factor = max(0.0, 1.0 - generation_diff * 0.1)
        
        # History similarity
        history_similarity = len(set(original.lineage_history) & set(current.lineage_history)) / max(len(original.lineage_history), len(current.lineage_history)) if original.lineage_history and current.lineage_history else 1.0
        
        # Combined fidelity score
        fidelity_score = (cell_id_similarity * 0.5 + generation_factor * 0.3 + history_similarity * 0.2)
        
        return min(1.0, max(0.0, fidelity_score))
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        total_length = max(len(seq1), len(seq2))
        
        return matches / total_length if total_length > 0 else 0.0
    
    def classify_fidelity_level(self, fidelity_score: float) -> TagFidelityLevel:
        """Classify fidelity level based on score"""
        if fidelity_score >= 0.95:
            return TagFidelityLevel.HIGH
        elif fidelity_score >= 0.80:
            return TagFidelityLevel.MEDIUM
        elif fidelity_score >= 0.60:
            return TagFidelityLevel.LOW
        else:
            return TagFidelityLevel.DEGRADED
    
    def apply_time_degradation(self, tag: LineageTag, inheritance_mode: TagInheritanceMode, 
                             time_delta: float) -> None:
        """Apply time-based degradation to lineage tag"""
        params = self.preservation_parameters[inheritance_mode.value]
        degradation_rate = params["degradation_rate"]
        
        degradation_probability = degradation_rate * time_delta
        
        if np.random.random() < degradation_probability:
            # Apply degradation by modifying cell ID
            tag.cell_id = f"{tag.cell_id}_deg"
            tag.last_update_time = tag.last_update_time + time_delta
