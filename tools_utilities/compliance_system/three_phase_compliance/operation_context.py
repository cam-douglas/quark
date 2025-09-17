"""
Operation Context Manager

Context manager for three-phase compliance checking operations.

Author: Quark AI
Date: 2025-01-27
"""

from contextlib import contextmanager
from typing import List, Optional
from .core_system import ThreePhaseComplianceSystem


@contextmanager
def operation_context(compliance_system: ThreePhaseComplianceSystem, 
                     operation_name: str, 
                     target_files: Optional[List[str]] = None):
    """
    Context manager for three-phase compliance checking
    
    Usage:
    with operation_context(compliance_system, "file_edit", ["file.py"]):
        # Your operation here
        pass
    """
    # Phase 1: Before
    if not compliance_system.phase_before_operation(operation_name, target_files):
        raise RuntimeError(f"Operation '{operation_name}' blocked due to pre-existing violations")
    
    # Phase 2: During
    compliance_system.phase_during_operation(operation_name, target_files)
    
    try:
        yield compliance_system
    finally:
        # Phase 3: After
        compliance_system.phase_after_operation(operation_name, target_files)
