"""
Brain Core Package
==================
High-level orchestration layer for the brain system.
"""

from .brain_orchestrator import BrainOrchestrator
from .component_registry import ComponentRegistry

__all__ = ['BrainOrchestrator', 'ComponentRegistry']
