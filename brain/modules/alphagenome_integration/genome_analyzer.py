#!/usr/bin/env python3
"""Genome Analyzer - Clean interface to modular genome analysis system.

Integration: This module participates in genomic workflows via GenomeAnalyzer.
Rationale: Streamlined interface to modular genome analysis components.
"""

# Import from modular genome analysis system
from .genome_analysis import (
    GenomeAnalyzer,
    GenomicRegion,
    RegulatoryElement,
    GeneRegulatoryNetwork,
    create_genome_analyzer
)

# Re-export for backward compatibility
__all__ = [
    'GenomeAnalyzer', 'GenomicRegion', 'RegulatoryElement',
    'GeneRegulatoryNetwork', 'create_genome_analyzer'
]
