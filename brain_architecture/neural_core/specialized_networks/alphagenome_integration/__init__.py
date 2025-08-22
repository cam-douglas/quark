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

from . import dna_controller
from . import cell_constructor
from . import genome_analyzer
from . import biological_simulator

__version__ = "1.0.0"
__all__ = [
    "dna_controller",
    "cell_constructor",
    "genome_analyzer",
    "biological_simulator",
]
