#!/usr/bin/env python3
"""Genomic Types Module - Core data structures for genome analysis.

Defines data classes and type definitions for genomic analysis and regulatory networks.

Integration: Core types used by GenomeAnalyzer and regulatory network analysis.
Rationale: Centralized genomic type definitions separate from analysis logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

@dataclass
class GenomicRegion:
    """Represents a genomic region with annotations"""
    chromosome: str
    start: int
    end: int
    region_type: str  # promoter, enhancer, exon, intron, intergenic
    gene_associations: List[str]
    conservation_score: float
    regulatory_potential: float
    developmental_relevance: str

@dataclass
class RegulatoryElement:
    """Represents a regulatory element (enhancer, promoter, silencer)"""
    element_id: str
    genomic_region: GenomicRegion
    element_type: str  # promoter, enhancer, silencer, insulator
    target_genes: List[str]
    tissue_specificity: Dict[str, float]
    developmental_timing: List[str]
    transcription_factors: List[str]
    chromatin_state: Dict[str, float]
    activity_score: float

@dataclass
class GeneRegulatoryNetwork:
    """Represents a gene regulatory network"""
    network_id: str
    biological_process: str
    core_genes: List[str]
    transcription_factors: List[str]
    regulatory_interactions: Dict[str, List[str]]  # TF -> target genes
    feedback_loops: List[Tuple[str, str]]
    network_topology: Dict[str, Any]
    expression_dynamics: Dict[str, np.ndarray]
