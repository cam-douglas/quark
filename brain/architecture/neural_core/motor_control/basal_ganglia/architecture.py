#!/usr/bin/env python3
"""Basal Ganglia Architecture - Clean interface to modular basal ganglia system.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular basal ganglia components.
"""

# Import from modular basal ganglia system
from .bg_components import BasalGangliaArchitecture, NucleusConfig

# Re-export for backward compatibility
__all__ = ['BasalGangliaArchitecture', 'NucleusConfig']
