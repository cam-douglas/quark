#!/usr/bin/env python3
"""LLM Manipulation Planner - Interface to modular manipulation planning.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular manipulation planning components.
"""

# Import from modular manipulation planning system
from .manipulation_planning.planner_core import *

# Re-export for backward compatibility
__all__ = ['LLMManipulationPlanner']
