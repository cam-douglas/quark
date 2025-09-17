"""
Quark Compliance System

Unified compliance checking system that provides:
- Rule compliance checking
- Three-phase operation validation (before/during/after)
- Pre-push hook integration
- Cursor AI operation integration

All functionality organized into modular components.

Author: Quark AI
Date: 2025-01-27
"""

from .core_system import QuarkComplianceSystem
from .cli_interface import main

__version__ = "1.0.0"
__all__ = ["QuarkComplianceSystem", "main"]
