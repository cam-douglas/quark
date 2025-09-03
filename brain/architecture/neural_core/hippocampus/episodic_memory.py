#!/usr/bin/env python3
"""Episodic Memory - Interface to modular memory systems.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular episodic memory components.
"""

# Import from modular memory systems
from .memory_systems.episodic_core import *

# Re-export for backward compatibility
__all__ = ['EpisodicMemory']
