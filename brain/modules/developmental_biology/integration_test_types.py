"""
Integration Test Types and Data Structures

Common types and data structures for foundation integration testing.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class IntegrationTestStatus(Enum):
    """Status of integration tests"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class IntegrationTestResult:
    """Result of a foundation integration test"""
    test_name: str
    foundation_component: str
    validation_component: str
    test_status: IntegrationTestStatus
    experimental_accuracy: float
    integration_score: float
    error_message: Optional[str] = None
