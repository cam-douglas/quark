#!/usr/bin/env python3
"""Barcode Mutation Engine.

Handles barcode mutations, inheritance patterns, and stability management
for molecular lineage barcodes including point mutations, deletions,
insertions, and dilution effects.

Integration: Mutation handling component for barcode generation system
Rationale: Focused mutation logic separated from main generator
"""

from typing import List
import random
import logging

from .lineage_barcode_types import MutationType, BarcodeInheritanceRule

logger = logging.getLogger(__name__)

class BarcodeMutationEngine:
    """Engine for handling barcode mutations and inheritance effects.
    
    Manages mutation processes, dilution effects, and amplification
    for molecular barcodes during cell division and lineage progression.
    """
    
    def __init__(self):
        """Initialize barcode mutation engine."""
        # Barcode alphabets for different types
        self.dna_alphabet = ['A', 'T', 'G', 'C']
        self.rna_alphabet = ['A', 'U', 'G', 'C']
        self.protein_alphabet = list('ACDEFGHIKLMNPQRSTVWY')  # 20 amino acids
        self.epigenetic_alphabet = ['M', 'U', 'H3K4', 'H3K9', 'H3K27']  # Methylation/histone marks
        
        logger.info("Initialized BarcodeMutationEngine")
    
    def apply_inheritance_effects(self, barcode_elements: List[str],
                                 inheritance_rule: BarcodeInheritanceRule) -> List[str]:
        """Apply all inheritance effects to barcode elements.
        
        Args:
            barcode_elements: Original barcode elements
            inheritance_rule: Rules for inheritance
            
        Returns:
            Modified barcode elements
        """
        modified_elements = barcode_elements.copy()
        
        # Apply mutations
        if random.random() < inheritance_rule.mutation_rate:
            modified_elements = self.apply_mutations(modified_elements, inheritance_rule.mutation_types)
        
        # Apply dilution
        if random.random() < inheritance_rule.dilution_rate:
            modified_elements = self.apply_dilution(modified_elements)
        
        # Apply amplification
        if inheritance_rule.amplification_factor != 1.0:
            modified_elements = self.apply_amplification(modified_elements, inheritance_rule.amplification_factor)
        
        return modified_elements
    
    def apply_mutations(self, barcode_elements: List[str], 
                       allowed_mutations: List[MutationType]) -> List[str]:
        """Apply mutations to barcode elements."""
        mutated_elements = barcode_elements.copy()
        
        for mutation_type in allowed_mutations:
            if random.random() < 0.3:  # 30% chance for each mutation type
                if mutation_type == MutationType.POINT_MUTATION and mutated_elements:
                    mutated_elements = self._apply_point_mutation(mutated_elements)
                    
                elif mutation_type == MutationType.DELETION and len(mutated_elements) > 5:
                    mutated_elements = self._apply_deletion(mutated_elements)
                    
                elif mutation_type == MutationType.INSERTION and len(mutated_elements) < 30:
                    mutated_elements = self._apply_insertion(mutated_elements)
        
        return mutated_elements
    
    def _apply_point_mutation(self, elements: List[str]) -> List[str]:
        """Apply point mutation to barcode."""
        mutated = elements.copy()
        pos = random.randint(0, len(mutated) - 1)
        
        # Determine alphabet based on current element
        alphabet = self._get_alphabet_for_element(mutated[pos])
        
        # Replace with different element from same alphabet
        new_element = random.choice([e for e in alphabet if e != mutated[pos]])
        mutated[pos] = new_element
        
        return mutated
    
    def _apply_deletion(self, elements: List[str]) -> List[str]:
        """Apply deletion mutation to barcode."""
        if len(elements) <= 5:
            return elements  # Don't delete if too short
        
        mutated = elements.copy()
        pos = random.randint(0, len(mutated) - 1)
        del mutated[pos]
        
        return mutated
    
    def _apply_insertion(self, elements: List[str]) -> List[str]:
        """Apply insertion mutation to barcode."""
        if len(elements) >= 30:
            return elements  # Don't insert if too long
        
        mutated = elements.copy()
        pos = random.randint(0, len(mutated))
        
        # Determine alphabet from neighboring elements
        if pos > 0:
            neighbor = mutated[pos - 1]
        elif pos < len(mutated):
            neighbor = mutated[pos]
        else:
            neighbor = 'A'  # Default
        
        alphabet = self._get_alphabet_for_element(neighbor)
        new_element = random.choice(alphabet)
        mutated.insert(pos, new_element)
        
        return mutated
    
    def _get_alphabet_for_element(self, element: str) -> List[str]:
        """Get appropriate alphabet for barcode element."""
        if element in self.dna_alphabet:
            return self.dna_alphabet
        elif element in self.rna_alphabet:
            return self.rna_alphabet
        elif element in self.protein_alphabet:
            return self.protein_alphabet
        elif element in self.epigenetic_alphabet:
            return self.epigenetic_alphabet
        else:
            return self.dna_alphabet  # Default
    
    def apply_dilution(self, barcode_elements: List[str]) -> List[str]:
        """Apply dilution effects to barcode."""
        if len(barcode_elements) <= 10:
            return barcode_elements  # Keep minimum length
        
        # Randomly remove some elements
        diluted = barcode_elements.copy()
        num_to_remove = random.randint(1, min(3, len(diluted) - 10))
        
        for _ in range(num_to_remove):
            if len(diluted) > 10:
                pos = random.randint(0, len(diluted) - 1)
                del diluted[pos]
        
        return diluted
    
    def apply_amplification(self, barcode_elements: List[str], 
                          amplification_factor: float) -> List[str]:
        """Apply amplification effects to barcode."""
        if amplification_factor <= 1.0 or len(barcode_elements) >= 25:
            return barcode_elements
        
        # Duplicate some elements
        amplified = barcode_elements.copy()
        num_to_add = int((amplification_factor - 1.0) * len(amplified))
        num_to_add = min(num_to_add, 5)  # Limit amplification
        
        for _ in range(num_to_add):
            if len(amplified) < 25:
                element_to_duplicate = random.choice(amplified)
                pos = random.randint(0, len(amplified))
                amplified.insert(pos, element_to_duplicate)
        
        return amplified
    
    def calculate_mutation_distance(self, original: List[str], mutated: List[str]) -> float:
        """Calculate mutation distance between original and mutated barcodes.
        
        Args:
            original: Original barcode elements
            mutated: Mutated barcode elements
            
        Returns:
            Mutation distance (0-1, where 0 = identical)
        """
        if not original and not mutated:
            return 0.0
        
        max_len = max(len(original), len(mutated))
        if max_len == 0:
            return 0.0
        
        # Calculate edit distance (simplified)
        min_len = min(len(original), len(mutated))
        matches = sum(1 for i in range(min_len) if original[i] == mutated[i])
        
        # Account for length differences
        length_penalty = abs(len(original) - len(mutated))
        
        distance = (min_len - matches + length_penalty) / max_len
        
        return min(1.0, distance)
