"""
Proliferation Validation Types

Types and data structures for proliferation rate validation.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class ValidationMetric(Enum):
    """Types of validation metrics"""
    CELL_CYCLE_LENGTH = "cell_cycle_length"
    DIVISION_RATE = "division_rate"
    PHASE_DURATION = "phase_duration"
    PROLIFERATION_INDEX = "proliferation_index"


class ValidationStatus(Enum):
    """Status of validation"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ExperimentalData:
    """Experimental data for validation"""
    metric_name: str
    expected_value: float
    standard_deviation: float
    sample_size: int
    reference_source: str
    developmental_stage: str


@dataclass
class ValidationResult:
    """Result of a validation comparison"""
    metric_type: ValidationMetric
    experimental_value: float
    simulated_value: float
    validation_status: ValidationStatus
    confidence_interval: tuple
    reference_source: str
    z_score: float
    p_value: Optional[float] = None
