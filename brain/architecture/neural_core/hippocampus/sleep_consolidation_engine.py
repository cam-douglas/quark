#!/usr/bin/env python3
"""Sleep Consolidation Engine - Interface to modular sleep systems.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular sleep consolidation components.
"""

# Import from modular sleep systems
from .sleep_systems.consolidation_core import *

# Re-export for backward compatibility
__all__ = ['SleepConsolidationEngine']
