"""
AlphaGenome Integration Module for Brain Simulation

This module integrates Google DeepMind's AlphaGenome API to enable:
- DNA sequence analysis and prediction
- Biological cell construction simulation
- Gene regulatory network modeling
- Chromatin structure prediction
- Variant effect prediction

Author: Brain Simulation Team
Date: 2025
License: Apache 2.0 (compatible with AlphaGenome)
"""

from ...........................................................dna_controller import DNAController
from ...........................................................cell_constructor import CellConstructor
from ...........................................................genome_analyzer import GenomeAnalyzer
from ...........................................................biological_simulator import BiologicalSimulator

__version__ = "1.0.0"
__all__ = [
    "DNAController",
    "CellConstructor", 
    "GenomeAnalyzer",
    "BiologicalSimulator"
]
