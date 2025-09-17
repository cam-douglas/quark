"""
Compliance Checker Package

Modular rule compliance checking system for Quark project.
Split from original 426-line file to maintain <300 line limit.

Author: Quark AI
Date: 2025-01-27
"""

from .core_checker import RuleComplianceChecker
from .violation_types import Violation, ComplianceReport
from .cli import main

__all__ = ['RuleComplianceChecker', 'Violation', 'ComplianceReport', 'main']
