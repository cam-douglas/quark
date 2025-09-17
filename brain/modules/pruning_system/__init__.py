#!/usr/bin/env python3
"""Unified Pruning System - Main interface for consolidated pruning operations.

Replaces multiple redundant pruning systems with one unified, efficient system.

Integration: Main pruning interface for brain optimization and cleanup workflows.
Rationale: Eliminates redundancy and provides single pruning system interface.
"""

from typing import Optional
from .pruning_types import (
    PruningCandidate, PruningPlan, ProgressMetrics, FileAnalysis,
    PruningStrategy, PruningDecisionType
)
from .unified_pruning_engine import UnifiedPruningEngine

# Factory functions for backward compatibility
def create_integrated_pruning_system():
    """Create unified pruning system (replaces IntegratedPruningSystem)."""
    return UnifiedPruningEngine(strategy=PruningStrategy.CONSERVATIVE)

def create_optimized_pruning_system():
    """Create optimization-focused pruning system (replaces OptimizedPruningSystem)."""
    return UnifiedPruningEngine(strategy=PruningStrategy.OPTIMIZATION)

def create_biological_pruning_system():
    """Create biology-focused pruning system (replaces BiologicalPruningSystem)."""
    return UnifiedPruningEngine(strategy=PruningStrategy.BIOLOGICAL)

# Main unified interface
class PruningSystem:
    """Unified interface to all pruning operations."""

    def __init__(self, strategy: PruningStrategy = PruningStrategy.CONSERVATIVE):
        self.engine = UnifiedPruningEngine(strategy)

    def analyze_and_plan(self, directory: str, strategy: Optional[PruningStrategy] = None):
        """Analyze directory and create pruning plan."""
        from pathlib import Path

        if strategy:
            self.engine.strategy = strategy

        # Analyze directory
        analysis = self.engine.analyze_directory(Path(directory))

        # Create plan
        plan = self.engine.create_pruning_plan(analysis["candidates"])

        return {
            "analysis": analysis,
            "plan": plan,
            "summary": {
                "total_files": analysis["total_files"],
                "candidates": len(analysis["candidates"]),
                "files_to_remove": len(plan.files_to_remove),
                "estimated_space_saved": plan.estimated_space_saved,
                "safety_score": plan.safety_score
            }
        }

# Export unified interface
__all__ = [
    'PruningSystem', 'UnifiedPruningEngine',
    'PruningCandidate', 'PruningPlan', 'ProgressMetrics', 'FileAnalysis',
    'PruningStrategy', 'PruningDecisionType',
    'create_integrated_pruning_system', 'create_optimized_pruning_system',
    'create_biological_pruning_system'
]
