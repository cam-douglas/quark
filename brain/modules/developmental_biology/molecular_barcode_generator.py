#!/usr/bin/env python3
"""Molecular Barcode Generator.

Main coordinator for generating unique molecular barcodes for lineage
tracking including DNA, RNA, protein, and epigenetic barcodes with
inheritance and validation.

Integration: Barcode generation coordinator for lineage tracking system
Rationale: Main barcode coordinator with focused responsibilities
"""

from typing import Dict, List
import random
import string
import hashlib
import logging

from .lineage_barcode_types import (
    BarcodeType, InheritancePattern, BarcodeSequence, BarcodeInheritanceRule, MutationType
)
from .barcode_mutation_engine import BarcodeMutationEngine

logger = logging.getLogger(__name__)

class MolecularBarcodeGenerator:
    """Generator for molecular lineage barcodes.
    
    Main coordinator for creating unique molecular identifiers for cell
    lineage tracking with different barcode types and inheritance patterns.
    """
    
    def __init__(self, barcode_length: int = 20):
        """Initialize molecular barcode generator.
        
        Args:
            barcode_length: Length of barcode sequences
        """
        self.barcode_length = barcode_length
        self.mutation_engine = BarcodeMutationEngine()
        self.inheritance_rules = self._initialize_inheritance_rules()
        
        # Barcode alphabets
        self.dna_alphabet = ['A', 'T', 'G', 'C']
        self.rna_alphabet = ['A', 'U', 'G', 'C']
        self.protein_alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        self.epigenetic_alphabet = ['M', 'U', 'H3K4', 'H3K9', 'H3K27']
        
        logger.info("Initialized MolecularBarcodeGenerator")
        logger.info(f"Barcode length: {barcode_length}")
    
    def _initialize_inheritance_rules(self) -> Dict[BarcodeType, BarcodeInheritanceRule]:
        """Initialize inheritance rules for different barcode types."""
        return {
            BarcodeType.DNA_BARCODE: BarcodeInheritanceRule(
                inheritance_pattern=InheritancePattern.STABLE_INHERITANCE,
                mutation_rate=0.001,
                dilution_rate=0.0,
                amplification_factor=1.0,
                mutation_types=[MutationType.POINT_MUTATION]
            ),
            BarcodeType.RNA_BARCODE: BarcodeInheritanceRule(
                inheritance_pattern=InheritancePattern.MUTATION_PRONE,
                mutation_rate=0.01,
                dilution_rate=0.05,
                amplification_factor=1.0,
                mutation_types=[MutationType.POINT_MUTATION, MutationType.DELETION]
            ),
            BarcodeType.PROTEIN_BARCODE: BarcodeInheritanceRule(
                inheritance_pattern=InheritancePattern.DILUTION_PRONE,
                mutation_rate=0.005,
                dilution_rate=0.1,
                amplification_factor=0.9,
                mutation_types=[MutationType.POINT_MUTATION, MutationType.DELETION]
            ),
            BarcodeType.EPIGENETIC_BARCODE: BarcodeInheritanceRule(
                inheritance_pattern=InheritancePattern.MUTATION_PRONE,
                mutation_rate=0.02,
                dilution_rate=0.03,
                amplification_factor=1.1,
                mutation_types=[MutationType.POINT_MUTATION, MutationType.INSERTION, MutationType.DELETION]
            )
        }
    
    def generate_initial_barcode(self, barcode_type: BarcodeType, 
                                cell_id: str) -> BarcodeSequence:
        """Generate initial barcode for a cell."""
        # Select appropriate alphabet
        alphabet = self._get_alphabet_for_type(barcode_type)
        
        # Generate unique sequence using cell_id as seed
        random.seed(hash(cell_id + barcode_type.value))
        barcode_elements = [random.choice(alphabet) for _ in range(self.barcode_length)]
        
        # Create sequence ID
        sequence_hash = hashlib.md5(''.join(barcode_elements).encode()).hexdigest()[:8]
        sequence_id = f"{barcode_type.value}_{sequence_hash}"
        
        # Calculate stability and detectability
        inheritance_rule = self.inheritance_rules[barcode_type]
        stability_score = 1.0 - inheritance_rule.mutation_rate - inheritance_rule.dilution_rate
        
        barcode_sequence = BarcodeSequence(
            sequence_id=sequence_id,
            barcode_elements=barcode_elements,
            barcode_type=barcode_type,
            inheritance_pattern=inheritance_rule.inheritance_pattern,
            stability_score=max(0.1, stability_score),
            detectability_score=self._calculate_detectability(barcode_type)
        )
        
        logger.debug(f"Generated {barcode_type.value} barcode: {sequence_id}")
        
        return barcode_sequence
    
    def _get_alphabet_for_type(self, barcode_type: BarcodeType) -> List[str]:
        """Get alphabet for specific barcode type."""
        if barcode_type == BarcodeType.DNA_BARCODE:
            return self.dna_alphabet
        elif barcode_type == BarcodeType.RNA_BARCODE:
            return self.rna_alphabet
        elif barcode_type == BarcodeType.PROTEIN_BARCODE:
            return self.protein_alphabet
        else:  # EPIGENETIC_BARCODE
            return self.epigenetic_alphabet
    
    def _calculate_detectability(self, barcode_type: BarcodeType) -> float:
        """Calculate detectability score for barcode type."""
        detectability_scores = {
            BarcodeType.DNA_BARCODE: 0.95,
            BarcodeType.RNA_BARCODE: 0.8,
            BarcodeType.PROTEIN_BARCODE: 0.7,
            BarcodeType.EPIGENETIC_BARCODE: 0.6
        }
        
        return detectability_scores.get(barcode_type, 0.5)
    
    def inherit_barcode(self, parent_barcode: BarcodeSequence, 
                       daughter_cell_id: str) -> BarcodeSequence:
        """Inherit barcode from parent with potential modifications."""
        inheritance_rule = self.inheritance_rules[parent_barcode.barcode_type]
        
        # Apply inheritance effects
        inherited_elements = self.mutation_engine.apply_inheritance_effects(
            parent_barcode.barcode_elements, inheritance_rule)
        
        # Create new sequence ID
        sequence_hash = hashlib.md5(''.join(inherited_elements).encode()).hexdigest()[:8]
        new_sequence_id = f"{parent_barcode.barcode_type.value}_{sequence_hash}"
        
        # Calculate new stability
        new_stability = parent_barcode.stability_score * 0.98  # Slight degradation
        
        inherited_barcode = BarcodeSequence(
            sequence_id=new_sequence_id,
            barcode_elements=inherited_elements,
            barcode_type=parent_barcode.barcode_type,
            inheritance_pattern=parent_barcode.inheritance_pattern,
            stability_score=max(0.1, new_stability),
            detectability_score=parent_barcode.detectability_score
        )
        
        logger.debug(f"Inherited barcode: {parent_barcode.sequence_id} → {new_sequence_id}")
        
        return inherited_barcode
    
    def calculate_barcode_similarity(self, barcode1: BarcodeSequence, 
                                   barcode2: BarcodeSequence) -> float:
        """Calculate similarity between two barcodes."""
        if barcode1.barcode_type != barcode2.barcode_type:
            return 0.0
        
        elements1 = barcode1.barcode_elements
        elements2 = barcode2.barcode_elements
        
        max_len = max(len(elements1), len(elements2))
        if max_len == 0:
            return 1.0
        
        # Count matching elements at same positions
        min_len = min(len(elements1), len(elements2))
        matches = sum(1 for i in range(min_len) if elements1[i] == elements2[i])
        
        # Similarity accounting for length differences
        similarity = matches / max_len
        
        return similarity
    
    def validate_barcode_integrity(self, barcode: BarcodeSequence) -> Dict[str, bool]:
        """Validate barcode integrity and detectability."""
        validation_results = {}
        
        # Check length constraints
        validation_results['minimum_length'] = len(barcode.barcode_elements) >= 5
        validation_results['maximum_length'] = len(barcode.barcode_elements) <= 30
        
        # Check alphabet consistency
        alphabet = self._get_alphabet_for_type(barcode.barcode_type)
        valid_elements = all(elem in alphabet for elem in barcode.barcode_elements)
        validation_results['alphabet_consistency'] = valid_elements
        
        # Check quality thresholds
        validation_results['detectability_threshold'] = barcode.detectability_score > 0.3
        validation_results['stability_threshold'] = barcode.stability_score > 0.1
        
        # Overall validation
        validation_results['overall_valid'] = all(validation_results.values())
        
        return validation_results