#!/usr/bin/env python3
"""DNA Sequence Configuration Module - Configuration and parameters for DNA analysis.

Defines configuration structures and parameters for biological DNA sequence analysis.

Integration: Configuration types used by DNAController and AlphaGenome workflows.
Rationale: Centralized configuration definitions separate from control logic.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BiologicalSequenceConfig:
    """Configuration for biological DNA sequence analysis"""
    api_key: Optional[str] = None
    sequence_length_limit: int = 1000000  # 1M base pairs (AlphaGenome limit)
    resolution: str = "single_bp"  # single base pair resolution
    prediction_outputs: List[str] = None
    ontology_terms: List[str] = None

    def __post_init__(self):
        if self.prediction_outputs is None:
            self.prediction_outputs = [
                "RNA_SEQ",           # Gene expression
                "ATAC",              # Chromatin accessibility (ATAC-seq)
                "CHIP_HISTONE",      # Histone modifications
                "DNASE",             # DNase hypersensitivity
                "CONTACT_MAPS",      # 3D chromatin interactions
                "CAGE"               # Cap analysis gene expression
            ]

        if self.ontology_terms is None:
            # Neurobiological ontology terms for brain development (verified supported)
            self.ontology_terms = [
                "UBERON:0001157",  # brain (verified working)
            ]

def get_default_neural_config() -> BiologicalSequenceConfig:
    """Get default configuration optimized for neural development analysis."""
    return BiologicalSequenceConfig(
        sequence_length_limit=500000,  # Optimized for neural genes
        resolution="single_bp",
        prediction_outputs=[
            "RNA_SEQ",           # Critical for neural gene expression
            "CHIP_HISTONE",      # Important for neural development
            "ATAC",              # Chromatin accessibility in neural tissue
            "DNASE"              # Neural-specific regulatory regions
        ],
        ontology_terms=[
            "UBERON:0001157",    # brain
            "UBERON:0001016",    # nervous system
            "UBERON:0000955",    # brain gray matter
            "UBERON:0002316"     # white matter
        ]
    )

def get_developmental_config() -> BiologicalSequenceConfig:
    """Get configuration optimized for developmental biology analysis."""
    return BiologicalSequenceConfig(
        sequence_length_limit=750000,  # Larger for developmental genes
        resolution="single_bp",
        prediction_outputs=[
            "RNA_SEQ",           # Developmental gene expression
            "CHIP_HISTONE",      # Developmental chromatin marks
            "CONTACT_MAPS",      # 3D chromatin during development
            "CAGE"               # Developmental promoter activity
        ],
        ontology_terms=[
            "UBERON:0001157",    # brain
            "UBERON:0000922",    # embryo
            "UBERON:0007023",    # neural plate
            "UBERON:0001049"     # neural tube
        ]
    )
