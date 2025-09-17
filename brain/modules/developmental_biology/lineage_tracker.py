#!/usr/bin/env python3
"""Lineage Tracking System.

Main coordinator for lineage tracking using molecular barcodes including
lineage tree construction, barcode inheritance management, and lineage
validation for comprehensive cell lineage tracking.

Integration: Main lineage tracking coordinator for developmental biology
Rationale: Centralized lineage tracking with focused responsibilities
"""

from typing import Dict, List, Optional, Set, Tuple
import logging
import numpy as np
from collections import defaultdict

from .lineage_barcode_types import (
    BarcodeType, LineageTag, BarcodeSequence
)
from .molecular_barcode_generator import MolecularBarcodeGenerator

logger = logging.getLogger(__name__)

class LineageTracker:
    """Lineage tracking system for neuroepithelial cells.
    
    Main coordinator for tracking cell lineage progression using
    molecular barcodes including lineage tree construction, inheritance
    management, and validation during embryonic development.
    
    Key Components:
    - Molecular barcode generation and inheritance
    - Lineage tree construction and maintenance
    - Cell fate progression tracking
    - Lineage validation and quality control
    """
    
    def __init__(self, barcode_types: List[BarcodeType] = None):
        """Initialize lineage tracking system.
        
        Args:
            barcode_types: Types of barcodes to use (default: DNA + RNA)
        """
        self.barcode_types = barcode_types or [BarcodeType.DNA_BARCODE, BarcodeType.RNA_BARCODE]
        
        # Initialize barcode generator
        self.barcode_generator = MolecularBarcodeGenerator()
        
        # Lineage tracking state
        self.lineage_tags: Dict[str, LineageTag] = {}  # cell_id -> LineageTag
        self.lineage_tree: Dict[str, List[str]] = defaultdict(list)  # parent -> [children]
        self.generation_map: Dict[int, Set[str]] = defaultdict(set)  # generation -> {cell_ids}
        
        # Tracking statistics
        self.total_cells_tracked = 0
        self.total_divisions_tracked = 0
        self.barcode_integrity_scores: List[float] = []
        
        logger.info("Initialized LineageTracker")
        logger.info(f"Barcode types: {[bt.value for bt in self.barcode_types]}")
    
    def create_initial_lineage_tag(self, cell_id: str, 
                                  developmental_time: float) -> LineageTag:
        """Create initial lineage tag for founding cell.
        
        Args:
            cell_id: Cell identifier
            developmental_time: Current developmental time (weeks)
            
        Returns:
            Initial lineage tag with barcodes
        """
        # Generate barcodes for each type
        lineage_barcodes = {}
        for barcode_type in self.barcode_types:
            barcode = self.barcode_generator.generate_initial_barcode(barcode_type, cell_id)
            lineage_barcodes[barcode_type.value] = barcode
        
        # Create lineage tag
        lineage_tag = LineageTag(
            cell_id=cell_id,
            generation=0,
            lineage_barcodes=lineage_barcodes,
            lineage_history=[],
            creation_time=developmental_time,
            last_update_time=developmental_time
        )
        
        # Store in tracking system
        self.lineage_tags[cell_id] = lineage_tag
        self.generation_map[0].add(cell_id)
        self.total_cells_tracked += 1
        
        logger.info(f"Created initial lineage tag for cell {cell_id}")
        
        return lineage_tag
    
    def track_cell_division(self, parent_cell_id: str, daughter_cell_ids: List[str],
                           developmental_time: float) -> List[LineageTag]:
        """Track cell division and create daughter lineage tags.
        
        Args:
            parent_cell_id: Parent cell identifier
            daughter_cell_ids: List of daughter cell identifiers
            developmental_time: Current developmental time
            
        Returns:
            List of daughter lineage tags
        """
        if parent_cell_id not in self.lineage_tags:
            logger.error(f"Parent cell {parent_cell_id} not found in lineage tracking")
            return []
        
        parent_tag = self.lineage_tags[parent_cell_id]
        daughter_tags = []
        
        # Create lineage tags for each daughter
        for daughter_id in daughter_cell_ids:
            # Inherit barcodes from parent
            daughter_barcodes = {}
            for barcode_type_str, parent_barcode in parent_tag.lineage_barcodes.items():
                barcode_type = BarcodeType(barcode_type_str)
                inherited_barcode = self.barcode_generator.inherit_barcode(
                    parent_barcode, daughter_id)
                daughter_barcodes[barcode_type_str] = inherited_barcode
            
            # Create daughter lineage tag
            daughter_tag = LineageTag(
                cell_id=daughter_id,
                generation=parent_tag.generation + 1,
                lineage_barcodes=daughter_barcodes,
                lineage_history=parent_tag.lineage_history + [parent_cell_id],
                creation_time=developmental_time,
                last_update_time=developmental_time
            )
            
            # Store in tracking system
            self.lineage_tags[daughter_id] = daughter_tag
            self.lineage_tree[parent_cell_id].append(daughter_id)
            self.generation_map[daughter_tag.generation].add(daughter_id)
            
            daughter_tags.append(daughter_tag)
            self.total_cells_tracked += 1
        
        self.total_divisions_tracked += 1
        
        logger.info(f"Tracked division: {parent_cell_id} → {len(daughter_cell_ids)} daughters")
        
        return daughter_tags
    
    def get_lineage_tree(self, root_cell_id: str) -> Dict[str, List[str]]:
        """Get lineage tree starting from root cell.
        
        Args:
            root_cell_id: Root cell identifier
            
        Returns:
            Lineage tree dictionary
        """
        lineage_tree = {}
        
        def build_tree(cell_id: str):
            if cell_id in self.lineage_tree:
                children = self.lineage_tree[cell_id]
                lineage_tree[cell_id] = children
                for child_id in children:
                    build_tree(child_id)
        
        build_tree(root_cell_id)
        
        return lineage_tree
    
    def validate_lineage_integrity(self) -> Dict[str, any]:
        """Validate integrity of lineage tracking system.
        
        Returns:
            Validation results and statistics
        """
        validation_results = {}
        
        # Calculate barcode integrity scores
        integrity_scores = []
        for tag in self.lineage_tags.values():
            for barcode in tag.lineage_barcodes.values():
                integrity_results = self.barcode_generator.validate_barcode_integrity(barcode)
                if integrity_results['overall_valid']:
                    integrity_scores.append(barcode.stability_score)
        
        self.barcode_integrity_scores = integrity_scores
        
        # Calculate statistics
        validation_results.update({
            'total_cells_tracked': self.total_cells_tracked,
            'total_divisions_tracked': self.total_divisions_tracked,
            'average_barcode_integrity': np.mean(integrity_scores) if integrity_scores else 0.0,
            'min_barcode_integrity': np.min(integrity_scores) if integrity_scores else 0.0,
            'barcode_integrity_std': np.std(integrity_scores) if integrity_scores else 0.0,
            'generations_tracked': len(self.generation_map),
            'lineage_tree_completeness': self._calculate_tree_completeness()
        })
        
        # Validation thresholds
        validation_results['validation_passed'] = (
            validation_results['average_barcode_integrity'] > 0.8 and
            validation_results['lineage_tree_completeness'] > 0.95
        )
        
        logger.info(f"Lineage validation: {validation_results['validation_passed']}")
        logger.info(f"Average integrity: {validation_results['average_barcode_integrity']:.3f}")
        
        return validation_results
    
    def _calculate_tree_completeness(self) -> float:
        """Calculate completeness of lineage tree tracking."""
        if self.total_divisions_tracked == 0:
            return 1.0  # No divisions yet, complete by default
        
        # Check for orphaned cells (cells without proper lineage history)
        orphaned_cells = 0
        for cell_id, tag in self.lineage_tags.items():
            if tag.generation > 0:  # Not a founding cell
                # Should have parent in lineage history
                if not tag.lineage_history:
                    orphaned_cells += 1
                else:
                    # Parent should exist in tracking system
                    parent_id = tag.lineage_history[-1]
                    if parent_id not in self.lineage_tags:
                        orphaned_cells += 1
        
        completeness = 1.0 - (orphaned_cells / max(1, self.total_cells_tracked))
        
        return completeness
    
    def export_lineage_analysis(self) -> Dict[str, any]:
        """Export comprehensive lineage tracking analysis.
        
        Returns:
            Complete lineage analysis
        """
        validation_results = self.validate_lineage_integrity()
        
        analysis = {
            'tracking_statistics': {
                'total_cells_tracked': self.total_cells_tracked,
                'total_divisions_tracked': self.total_divisions_tracked,
                'generations_tracked': len(self.generation_map),
                'barcode_types_used': [bt.value for bt in self.barcode_types]
            },
            'lineage_integrity': validation_results,
            'generation_distribution': {
                gen: len(cells) for gen, cells in self.generation_map.items()
            },
            'barcode_performance': {
                'average_stability': validation_results['average_barcode_integrity'],
                'stability_range': (validation_results['min_barcode_integrity'], 
                                  max(self.barcode_integrity_scores) if self.barcode_integrity_scores else 0.0),
                'total_barcodes': len(self.barcode_integrity_scores)
            },
            'system_validation': {
                'lineage_tracking_operational': True,
                'barcode_inheritance_functional': self.total_divisions_tracked > 0,
                'integrity_threshold_met': validation_results['validation_passed']
            }
        }
        
        return analysis
