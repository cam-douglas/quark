#!/usr/bin/env python3
"""Pruning Types Module - Unified data structures for all pruning operations.

Consolidates all pruning-related data types from multiple pruning systems.

Integration: Core types used by unified pruning system and brain optimization.
Rationale: Centralized pruning type definitions eliminating redundancy.
"""

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
from enum import Enum

class PruningStrategy(Enum):
    """Strategies for pruning operations"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BIOLOGICAL = "biological"
    OPTIMIZATION = "optimization"

class PruningDecisionType(Enum):
    """Types of pruning decisions"""
    KEEP = "keep"
    REMOVE = "remove"
    REVIEW = "review"
    MERGE = "merge"

@dataclass
class PruningCandidate:
    """Unified pruning candidate structure"""
    file_path: Path
    file_size: int
    duplicate_score: float
    redundancy_score: float
    biological_relevance: float
    optimization_potential: float
    last_modified: float
    usage_frequency: int

    # Analysis metadata
    fingerprint: str
    content_hash: str
    similarity_matches: List[str] = None

    def __post_init__(self):
        if self.similarity_matches is None:
            self.similarity_matches = []

@dataclass
class PruningPlan:
    """Unified pruning plan structure"""
    plan_id: str
    strategy: PruningStrategy
    total_candidates: int
    estimated_space_saved: int
    safety_score: float

    # Categorized actions
    files_to_remove: List[PruningCandidate] = None
    files_to_merge: List[Tuple[List[PruningCandidate], str]] = None
    files_to_review: List[PruningCandidate] = None
    files_to_keep: List[PruningCandidate] = None

    # Execution metadata
    requires_confirmation: bool = True
    estimated_execution_time: float = 0.0
    risk_assessment: str = "medium"

    def __post_init__(self):
        if self.files_to_remove is None:
            self.files_to_remove = []
        if self.files_to_merge is None:
            self.files_to_merge = []
        if self.files_to_review is None:
            self.files_to_review = []
        if self.files_to_keep is None:
            self.files_to_keep = []

@dataclass
class ProgressMetrics:
    """Unified progress tracking for pruning operations"""
    files_analyzed: int = 0
    duplicates_found: int = 0
    space_analyzed: int = 0
    space_recoverable: int = 0

    # Performance metrics
    analysis_time: float = 0.0
    throughput_files_per_sec: float = 0.0

    # Quality metrics
    false_positive_rate: float = 0.0
    confidence_score: float = 0.0

    # Safety metrics
    protected_files: int = 0
    critical_files_flagged: int = 0

@dataclass
class FileAnalysis:
    """Comprehensive file analysis result"""
    file_path: Path
    file_size: int
    content_type: str

    # Content analysis
    content_hash: str
    structural_fingerprint: str
    semantic_similarity: float

    # Usage analysis
    import_count: int
    last_access: float
    modification_frequency: float

    # Biological relevance (for biological pruning)
    biological_score: float
    neural_relevance: float
    simulation_critical: bool

    # Optimization potential
    optimization_score: float
    refactoring_potential: float
    consolidation_candidate: bool
