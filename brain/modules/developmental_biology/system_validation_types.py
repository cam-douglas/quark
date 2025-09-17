"""
System Validation Types

Types and data structures for end-to-end system validation.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SystemValidationStatus(Enum):
    """Status of system validation"""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class SystemValidationResult:
    """Result of end-to-end system validation"""
    validation_name: str
    status: SystemValidationStatus
    accuracy_score: float
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None
