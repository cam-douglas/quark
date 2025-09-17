#!/usr/bin/env python3
"""Genome Analysis Module - Main interface for genome analysis system.

Provides unified interface to genomic analysis components with preserved integrations.

Integration: Main genome analysis interface for AlphaGenome biological workflows.
Rationale: Clean API abstraction maintaining all existing functionality.
"""

from .genomic_types import GenomicRegion, RegulatoryElement, GeneRegulatoryNetwork

# Import the main GenomeAnalyzer class from the original file for now
# This maintains backward compatibility while we modularize
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from the new modular core
from .analyzer_core import GenomeAnalyzer

def create_genome_analyzer(dna_controller=None, cell_constructor=None):
    """Factory function to create a GenomeAnalyzer with specified components."""
    return GenomeAnalyzer(
        dna_controller=dna_controller,
        cell_constructor=cell_constructor
    )

# Export main interface for backward compatibility
__all__ = [
    'GenomeAnalyzer',
    'GenomicRegion',
    'RegulatoryElement',
    'GeneRegulatoryNetwork',
    'create_genome_analyzer'
]
