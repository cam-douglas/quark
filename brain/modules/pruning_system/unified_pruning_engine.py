#!/usr/bin/env python3
"""Unified Pruning Engine - Consolidated pruning system combining all strategies.

Combines biological, optimization, and integrated pruning approaches into one system.

Integration: Main pruning engine for brain optimization and cleanup workflows.
Rationale: Eliminates redundancy by consolidating multiple pruning systems.
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from .pruning_types import (
    PruningCandidate, PruningPlan, ProgressMetrics, FileAnalysis,
    PruningStrategy, PruningDecisionType
)

logger = logging.getLogger(__name__)

class UnifiedPruningEngine:
    """
    Consolidated pruning system combining all pruning strategies.
    Eliminates redundancy from multiple separate pruning systems.
    """

    def __init__(self, strategy: PruningStrategy = PruningStrategy.CONSERVATIVE):
        self.strategy = strategy
        self.analysis_cache = {}
        self.progress_metrics = ProgressMetrics()

        # Safety configurations
        self.protected_patterns = [
            "*/credentials/*", "*/secrets/*", "*/__init__.py",
            "*/brain_main.py", "*/quark_state_system_launcher.py"
        ]

        self.critical_directories = [
            "brain/core", "state/quark_state_system",
            "data/credentials", "management/rules"
        ]

        logger.info(f"UnifiedPruningEngine initialized with {strategy.value} strategy")

    def analyze_directory(self, directory: Path, recursive: bool = True) -> Dict[str, Any]:
        """Analyze directory for pruning opportunities."""

        start_time = time.time()
        candidates = []

        # Get all files to analyze
        if recursive:
            files = list(directory.rglob("*"))
        else:
            files = list(directory.iterdir())

        files = [f for f in files if f.is_file()]

        for file_path in files:
            try:
                # Skip protected files
                if self._is_protected_file(file_path):
                    continue

                analysis = self._analyze_file(file_path)
                candidate = self._create_pruning_candidate(analysis)

                if candidate:
                    candidates.append(candidate)

                self.progress_metrics.files_analyzed += 1

            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")

        # Update metrics
        analysis_time = time.time() - start_time
        self.progress_metrics.analysis_time = analysis_time
        self.progress_metrics.throughput_files_per_sec = len(files) / analysis_time if analysis_time > 0 else 0

        return {
            "directory": str(directory),
            "total_files": len(files),
            "candidates": candidates,
            "analysis_time": analysis_time,
            "strategy": self.strategy.value
        }

    def create_pruning_plan(self, candidates: List[PruningCandidate]) -> PruningPlan:
        """Create comprehensive pruning plan from candidates."""

        plan_id = f"pruning_plan_{int(time.time())}"

        # Categorize candidates based on strategy
        files_to_remove = []
        files_to_merge = []
        files_to_review = []
        files_to_keep = []

        for candidate in candidates:
            decision = self._make_pruning_decision(candidate)

            if decision == PruningDecisionType.REMOVE:
                files_to_remove.append(candidate)
            elif decision == PruningDecisionType.MERGE:
                # Group similar files for merging
                self._add_to_merge_groups(candidate, files_to_merge)
            elif decision == PruningDecisionType.REVIEW:
                files_to_review.append(candidate)
            else:  # KEEP
                files_to_keep.append(candidate)

        # Calculate metrics
        total_space_saved = sum(c.file_size for c in files_to_remove)
        safety_score = self._calculate_safety_score(files_to_remove, files_to_merge)

        return PruningPlan(
            plan_id=plan_id,
            strategy=self.strategy,
            total_candidates=len(candidates),
            estimated_space_saved=total_space_saved,
            safety_score=safety_score,
            files_to_remove=files_to_remove,
            files_to_merge=files_to_merge,
            files_to_review=files_to_review,
            files_to_keep=files_to_keep,
            risk_assessment="low" if safety_score > 0.8 else "medium" if safety_score > 0.5 else "high"
        )

    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Comprehensive file analysis combining all strategies."""

        # Basic file info
        stat = file_path.stat()
        file_size = stat.st_size

        # Content analysis
        content_hash = self._calculate_content_hash(file_path)
        fingerprint = self._calculate_structural_fingerprint(file_path)

        # Usage analysis
        import_count = self._count_imports(file_path)

        # Biological relevance (for biological pruning)
        bio_score = self._calculate_biological_relevance(file_path)

        # Optimization potential
        opt_score = self._calculate_optimization_potential(file_path)

        return FileAnalysis(
            file_path=file_path,
            file_size=file_size,
            content_type=self._detect_content_type(file_path),
            content_hash=content_hash,
            structural_fingerprint=fingerprint,
            semantic_similarity=0.0,  # Would be calculated against other files
            import_count=import_count,
            last_access=stat.st_atime,
            modification_frequency=1.0,  # Simplified
            biological_score=bio_score,
            neural_relevance=bio_score * 0.8,  # Derived from biological score
            simulation_critical=bio_score > 0.7,
            optimization_score=opt_score,
            refactoring_potential=opt_score * 0.6,
            consolidation_candidate=opt_score > 0.5
        )

    def _make_pruning_decision(self, candidate: PruningCandidate) -> PruningDecisionType:
        """Make pruning decision based on strategy and candidate analysis."""

        # Safety checks first
        if candidate.biological_relevance > 0.8:
            return PruningDecisionType.KEEP

        if candidate.usage_frequency > 10:
            return PruningDecisionType.KEEP

        # Strategy-specific decisions
        if self.strategy == PruningStrategy.CONSERVATIVE:
            if candidate.duplicate_score > 0.9 and candidate.redundancy_score > 0.8:
                return PruningDecisionType.REMOVE
            elif candidate.optimization_potential > 0.7:
                return PruningDecisionType.REVIEW
            else:
                return PruningDecisionType.KEEP

        elif self.strategy == PruningStrategy.AGGRESSIVE:
            if candidate.duplicate_score > 0.7 or candidate.redundancy_score > 0.6:
                return PruningDecisionType.REMOVE
            elif candidate.optimization_potential > 0.5:
                return PruningDecisionType.MERGE
            else:
                return PruningDecisionType.REVIEW

        elif self.strategy == PruningStrategy.BIOLOGICAL:
            if candidate.biological_relevance < 0.3:
                return PruningDecisionType.REMOVE
            elif candidate.biological_relevance < 0.6:
                return PruningDecisionType.REVIEW
            else:
                return PruningDecisionType.KEEP

        else:  # OPTIMIZATION
            if candidate.optimization_potential > 0.8:
                return PruningDecisionType.MERGE
            elif candidate.optimization_potential > 0.5:
                return PruningDecisionType.REVIEW
            else:
                return PruningDecisionType.KEEP

    def _create_pruning_candidate(self, analysis: FileAnalysis) -> Optional[PruningCandidate]:
        """Create pruning candidate from file analysis."""

        return PruningCandidate(
            file_path=analysis.file_path,
            file_size=analysis.file_size,
            duplicate_score=analysis.semantic_similarity,
            redundancy_score=analysis.refactoring_potential,
            biological_relevance=analysis.biological_score,
            optimization_potential=analysis.optimization_score,
            last_modified=analysis.last_access,
            usage_frequency=analysis.import_count,
            fingerprint=analysis.structural_fingerprint,
            content_hash=analysis.content_hash
        )

    def _is_protected_file(self, file_path: Path) -> bool:
        """Check if file is protected from pruning."""

        path_str = str(file_path)

        # Check protected patterns
        for pattern in self.protected_patterns:
            if pattern.replace("*", "") in path_str:
                return True

        # Check critical directories
        for crit_dir in self.critical_directories:
            if crit_dir in path_str:
                return True

        return False

    def _calculate_content_hash(self, file_path: Path) -> str:
        """Calculate content hash for duplicate detection."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def _calculate_structural_fingerprint(self, file_path: Path) -> str:
        """Calculate structural fingerprint for similarity detection."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Create fingerprint from structure
            structure_elements = []
            for line in lines:
                line = line.strip()
                if line.startswith('class ') or line.startswith('def '):
                    structure_elements.append(line)

            structure_str = '|'.join(structure_elements)
            return hashlib.md5(structure_str.encode()).hexdigest()[:16]

        except Exception:
            return "unknown"

    def _count_imports(self, file_path: Path) -> int:
        """Count how many times this file is imported."""
        # Simplified import counting
        return 1  # Would be implemented with comprehensive search

    def _detect_content_type(self, file_path: Path) -> str:
        """Detect file content type."""
        suffix = file_path.suffix.lower()

        type_map = {
            '.py': 'python',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.txt': 'text'
        }

        return type_map.get(suffix, 'unknown')

    def _calculate_biological_relevance(self, file_path: Path) -> float:
        """Calculate biological relevance score."""
        path_str = str(file_path).lower()

        biological_keywords = [
            'biological', 'bio', 'dna', 'gene', 'cell', 'neural', 'brain',
            'development', 'morphogen', 'simulation', 'alphagenome'
        ]

        score = 0.0
        for keyword in biological_keywords:
            if keyword in path_str:
                score += 0.2

        return min(1.0, score)

    def _calculate_optimization_potential(self, file_path: Path) -> float:
        """Calculate optimization potential score."""
        try:
            # Check file size and complexity
            stat = file_path.stat()
            file_size = stat.st_size

            # Larger files have higher optimization potential
            size_score = min(1.0, file_size / 100000)  # Normalize to 100KB

            # Check for optimization keywords
            path_str = str(file_path).lower()
            optimization_keywords = ['optimize', 'prune', 'clean', 'refactor', 'integrate']

            keyword_score = sum(0.2 for keyword in optimization_keywords if keyword in path_str)

            return min(1.0, (size_score + keyword_score) / 2)

        except Exception:
            return 0.0

    def _add_to_merge_groups(self, candidate: PruningCandidate, merge_list: List[Tuple[List[PruningCandidate], str]]):
        """Add candidate to appropriate merge group."""
        # Simple grouping by similar fingerprints
        for group, group_name in merge_list:
            if group and group[0].fingerprint == candidate.fingerprint:
                group.append(candidate)
                return

        # Create new group
        group_name = f"merge_group_{len(merge_list)}"
        merge_list.append(([candidate], group_name))

    def _calculate_safety_score(self, files_to_remove: List[PruningCandidate],
                               files_to_merge: List[Tuple[List[PruningCandidate], str]]) -> float:
        """Calculate safety score for pruning plan."""

        total_files = len(files_to_remove) + sum(len(group) for group, _ in files_to_merge)

        if total_files == 0:
            return 1.0

        # Calculate average biological relevance of files being modified
        total_relevance = 0.0
        for candidate in files_to_remove:
            total_relevance += candidate.biological_relevance

        for group, _ in files_to_merge:
            for candidate in group:
                total_relevance += candidate.biological_relevance

        avg_relevance = total_relevance / total_files if total_files > 0 else 0

        # Higher relevance = lower safety (more dangerous to remove)
        safety_score = 1.0 - avg_relevance

        return max(0.0, min(1.0, safety_score))
