"""
Unified Agent System Package

Consolidated system for both general agent delegation and specialized
engineering role orchestration.

Author: Quark AI
Date: 2025-01-27
"""

from .cli_interface import main
from .delegation_handler import run_delegation_system
from .orchestrator_handler import run_orchestrator_system
from .demo_system import run_unified_demo, show_system_status

__all__ = [
    'main',
    'run_delegation_system',
    'run_orchestrator_system', 
    'run_unified_demo',
    'show_system_status'
]
