"""
Three-Phase Compliance System Package

Comprehensive compliance checking that runs:
1. BEFORE operations (pre-validation)
2. DURING operations (real-time monitoring)  
3. AFTER operations (post-validation)

Refactored from 436-line file to maintain <300 line limit.

Author: Quark AI
Date: 2025-01-27
"""

from .core_system import ThreePhaseComplianceSystem
from .file_monitor import ComplianceFileMonitor
from .operation_context import operation_context

__all__ = ['ThreePhaseComplianceSystem', 'ComplianceFileMonitor', 'operation_context']
