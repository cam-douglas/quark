"""
Validation Pipeline Types

Types and data structures for end-to-end validation pipeline.

Author: Quark AI
Date: 2025-01-27
"""

from typing import List
from dataclasses import dataclass
from enum import Enum


class ValidationPipelineStatus(Enum):
    """Overall validation pipeline status"""
    EXCELLENT = "excellent"      # >95% accuracy
    GOOD = "good"               # 85-95% accuracy
    ACCEPTABLE = "acceptable"    # 70-85% accuracy
    NEEDS_IMPROVEMENT = "needs_improvement"  # 50-70% accuracy
    FAILED = "failed"           # <50% accuracy


@dataclass
class PipelineValidationReport:
    """Comprehensive validation report"""
    overall_status: ValidationPipelineStatus
    experimental_accuracy: float
    integration_score: float
    foundation_layer_status: str
    validation_systems_status: str
    literature_sources_count: int
    recommendations: List[str]
