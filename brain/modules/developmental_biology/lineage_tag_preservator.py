"""
Lineage Tag Preservator

This module maintains lineage tags through differentiation, implements tag
inheritance mechanisms, and validates tag fidelity over time.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from .lineage_barcode_types import LineageTag
from .committed_progenitor_types import CommittedProgenitor
from .tag_inheritance_engine import (
    TagInheritanceEngine, TagInheritanceMode, TagFidelityLevel, TagInheritanceEvent
)
from .tag_validation_manager import TagValidationManager, TagValidationResult


@dataclass
class TagPreservationState:
    """State of tag preservation for a cell"""
    cell_id: str
    original_barcode: LineageTag
    current_barcode: LineageTag
    inheritance_mode: TagInheritanceMode
    fidelity_level: TagFidelityLevel
    preservation_time: float
    degradation_factors: Dict[str, float]


class LineageTagPreservator:
    """
    Maintains lineage tags through differentiation, implements tag
    inheritance mechanisms, and validates tag fidelity over time.
    """
    
    def __init__(self):
        """Initialize lineage tag preservator"""
        self.preservation_states: Dict[str, TagPreservationState] = {}
        self.inheritance_events: List[TagInheritanceEvent] = []
        self.validation_results: List[TagValidationResult] = []
        self.inheritance_engine = TagInheritanceEngine()
        self.validation_manager = TagValidationManager()
    
    def initialize_tag_preservation(self, cells: Dict[str, CommittedProgenitor]) -> Dict[str, TagPreservationState]:
        """Initialize tag preservation for committed progenitors"""
        preservation_states = {}
        
        for cell_id, cell in cells.items():
            # Determine inheritance mode based on cell type and position
            inheritance_mode = self._determine_inheritance_mode(cell)
            
            # Initialize preservation state
            state = TagPreservationState(
                cell_id=cell_id,
                original_barcode=cell.lineage_barcode,
                current_barcode=self.inheritance_engine._copy_barcode(cell.lineage_barcode),
                inheritance_mode=inheritance_mode,
                fidelity_level=TagFidelityLevel.HIGH,
                preservation_time=0.0,
                degradation_factors={}
            )
            
            preservation_states[cell_id] = state
            self.preservation_states[cell_id] = state
        
        return preservation_states
    
    def _determine_inheritance_mode(self, cell: CommittedProgenitor) -> TagInheritanceMode:
        """Determine inheritance mode based on cell characteristics"""
        # Motor neurons tend to have faithful inheritance
        if any("motor_neuron" in fate.value for fate in cell.lineage_potency.allowed_fates):
            return TagInheritanceMode.FAITHFUL
        
        # Glial progenitors may have dilution
        if any("oligodendrocyte" in fate.value or "astrocyte" in fate.value 
               for fate in cell.lineage_potency.allowed_fates):
            return TagInheritanceMode.DILUTION
        
        # Interneurons may have selective preservation
        if any("interneuron" in fate.value for fate in cell.lineage_potency.allowed_fates):
            return TagInheritanceMode.SELECTIVE
        
        # Default to stochastic
        return TagInheritanceMode.STOCHASTIC
    
    def preserve_tags_through_division(self, parent_cell_id: str, 
                                     daughter_cell_ids: List[str],
                                     division_time: float) -> TagInheritanceEvent:
        """Preserve lineage tags through cell division"""
        if parent_cell_id not in self.preservation_states:
            raise ValueError(f"Parent cell {parent_cell_id} not found in preservation states")
        
        parent_state = self.preservation_states[parent_cell_id]
        inheritance_mode = parent_state.inheritance_mode
        
        # Create inheritance event
        inheritance_event = TagInheritanceEvent(
            parent_cell_id=parent_cell_id,
            daughter_cell_ids=daughter_cell_ids,
            inheritance_mode=inheritance_mode,
            fidelity_scores={},
            timestamp=division_time
        )
        
        # Process tag inheritance for each daughter cell
        for daughter_id in daughter_cell_ids:
            # Inherit barcode with potential modifications
            inherited_barcode = self.inheritance_engine.inherit_barcode(
                parent_state.current_barcode, inheritance_mode
            )
            
            # Calculate fidelity score
            fidelity_score = self.inheritance_engine.calculate_fidelity_score(
                parent_state.original_barcode, inherited_barcode
            )
            
            # Create preservation state for daughter cell
            daughter_state = TagPreservationState(
                cell_id=daughter_id,
                original_barcode=parent_state.original_barcode,
                current_barcode=inherited_barcode,
                inheritance_mode=inheritance_mode,
                fidelity_level=self.inheritance_engine.classify_fidelity_level(fidelity_score),
                preservation_time=0.0,
                degradation_factors=parent_state.degradation_factors.copy()
            )
            
            self.preservation_states[daughter_id] = daughter_state
            inheritance_event.fidelity_scores[daughter_id] = fidelity_score
        
        self.inheritance_events.append(inheritance_event)
        return inheritance_event
    
    def update_tag_preservation(self, cell_ids: List[str], time_delta: float) -> None:
        """Update tag preservation states over time"""
        for cell_id in cell_ids:
            if cell_id in self.preservation_states:
                state = self.preservation_states[cell_id]
                
                # Update preservation time
                state.preservation_time += time_delta
                
                # Apply time-based degradation
                self.inheritance_engine.apply_time_degradation(
                    state.current_barcode, state.inheritance_mode, time_delta
                )
                
                # Update fidelity level
                fidelity_score = self.inheritance_engine.calculate_fidelity_score(
                    state.original_barcode, state.current_barcode
                )
                state.fidelity_level = self.inheritance_engine.classify_fidelity_level(fidelity_score)
    
    def validate_tag_fidelity(self, cell_ids: List[str]) -> List[TagValidationResult]:
        """Validate tag fidelity for specified cells"""
        validation_results = []
        
        for cell_id in cell_ids:
            if cell_id in self.preservation_states:
                state = self.preservation_states[cell_id]
                
                # Calculate fidelity score
                fidelity_score = self.inheritance_engine.calculate_fidelity_score(
                    state.original_barcode, state.current_barcode
                )
                
                # Validate using validation manager
                result = self.validation_manager.validate_tag_fidelity(
                    cell_id, state.original_barcode, state.current_barcode, fidelity_score
                )
                
                validation_results.append(result)
                self.validation_results.append(result)
        
        return validation_results
    
    def get_preservation_statistics(self, cell_ids: List[str]) -> Dict[str, float]:
        """Get preservation statistics for specified cells"""
        if not cell_ids:
            return {}
        
        states = [self.preservation_states[cid] for cid in cell_ids 
                 if cid in self.preservation_states]
        
        if not states:
            return {}
        
        # Calculate statistics
        fidelity_scores = []
        fidelity_levels = []
        inheritance_modes = []
        
        for state in states:
            fidelity_score = self.inheritance_engine.calculate_fidelity_score(
                state.original_barcode, state.current_barcode
            )
            fidelity_scores.append(fidelity_score)
            fidelity_levels.append(state.fidelity_level.value)
            inheritance_modes.append(state.inheritance_mode.value)
        
        # Count distributions
        fidelity_distribution = {}
        for level in fidelity_levels:
            fidelity_distribution[level] = fidelity_distribution.get(level, 0) + 1
        
        mode_distribution = {}
        for mode in inheritance_modes:
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
        
        return {
            "total_cells": len(states),
            "average_fidelity": np.mean(fidelity_scores),
            "min_fidelity": np.min(fidelity_scores),
            "max_fidelity": np.max(fidelity_scores),
            "fidelity_distribution": fidelity_distribution,
            "inheritance_mode_distribution": mode_distribution,
            "total_inheritance_events": len(self.inheritance_events),
            "total_validations": len(self.validation_results)
        }