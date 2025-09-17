"""
Compliance System Launcher Package

Unified launcher for all compliance checking systems.
Refactored from 362-line file to maintain <300 line limit.

Author: Quark AI
Date: 2025-01-27
"""

from .core_launcher import ComplianceSystemLauncher
from .cursor_integration import setup_cursor_integration

__all__ = ['ComplianceSystemLauncher', 'setup_cursor_integration']
