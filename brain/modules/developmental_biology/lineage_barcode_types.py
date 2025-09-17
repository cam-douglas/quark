#!/usr/bin/env python3
"""Lineage Barcode Type Definitions.

Type definitions and data structures for molecular lineage barcoding system
including barcode formats, inheritance patterns, and mutation handling.

Integration: Type definitions for lineage tracking system
Rationale: Centralized barcode type definitions with biological accuracy
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

class BarcodeType(Enum):
    """Types of molecular barcodes for lineage tracking."""
    DNA_BARCODE = "dna_barcode"           # Genomic DNA integration
    RNA_BARCODE = "rna_barcode"           # Transcriptional barcode
    PROTEIN_BARCODE = "protein_barcode"   # Protein-based marking
    EPIGENETIC_BARCODE = "epigenetic_barcode"  # Chromatin modifications

class InheritancePattern(Enum):
    """Inheritance patterns for barcode transmission."""
    STABLE_INHERITANCE = "stable"         # No mutations, stable transmission
    MUTATION_PRONE = "mutation_prone"     # Some mutations during division
    DILUTION_PRONE = "dilution_prone"     # Gradual loss over divisions
    AMPLIFYING = "amplifying"             # Increases over time

class MutationType(Enum):
    """Types of barcode mutations."""
    POINT_MUTATION = "point_mutation"     # Single nucleotide/unit change
    DELETION = "deletion"                 # Loss of barcode elements
    INSERTION = "insertion"               # Addition of barcode elements
    RECOMBINATION = "recombination"       # Rearrangement of elements

@dataclass
class BarcodeSequence:
    """Molecular barcode sequence definition."""
    sequence_id: str                      # Unique sequence identifier
    barcode_elements: List[str]           # Individual barcode elements
    barcode_type: BarcodeType             # Type of molecular barcode
    inheritance_pattern: InheritancePattern  # How barcode is inherited
    stability_score: float               # Stability during divisions (0-1)
    detectability_score: float           # How easily detected (0-1)
    
@dataclass
class BarcodeInheritanceRule:
    """Rules for barcode inheritance during cell division."""
    inheritance_pattern: InheritancePattern
    mutation_rate: float                  # Probability of mutation per division
    dilution_rate: float                 # Rate of barcode dilution
    amplification_factor: float          # Amplification during inheritance
    mutation_types: List[MutationType]    # Allowed mutation types

@dataclass
class LineageTag:
    """Complete lineage tag including multiple barcodes."""
    cell_id: str                          # Cell identifier
    generation: int                       # Generation number
    lineage_barcodes: Dict[str, BarcodeSequence]  # Multiple barcode types
    lineage_history: List[str]            # Parent cell IDs
    creation_time: float                  # When tag was created (dev weeks)
    last_update_time: float              # Last modification time
