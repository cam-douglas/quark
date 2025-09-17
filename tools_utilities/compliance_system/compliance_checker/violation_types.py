"""
Violation Types and Data Structures

Defines data structures for rule violations and compliance reports.

Author: Quark AI
Date: 2025-01-27
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class Violation:
    """Represents a rule violation"""
    rule_type: str
    file_path: str
    line_number: Optional[int]
    message: str
    severity: str  # "error", "warning", "info"
    fix_suggestion: Optional[str] = None


@dataclass
class ComplianceReport:
    """Compliance check results"""
    timestamp: datetime
    total_files_checked: int
    violations: List[Violation]
    compliant: bool
    summary: Dict[str, int]
