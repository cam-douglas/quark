"""
Safety Officer Module - AGI Safety Control Agent (SENTINEL)

This module implements the embedded, immutable guardian system that enforces
constraints, ensures human control, and intercepts unsafe behaviors.

Biological Protocols: Implements thalamic security rules with GFAP + NeuN markers
SENTINEL Features: Immutable code vault, human override, behavioral watchdog

Author: Safety & Ethics Officer
Version: 1.0.0
Priority: 0 (Supreme Authority)
Biological Markers: GFAP (structural integrity), NeuN (neuronal identity)
"""

from . import safety_officer
from . import sentinel_agent
from . import biological_protocols
from . import safety_constraints
from . import audit_system

__version__ = "1.0.0"
__author__ = "Safety & Ethics Officer"
__priority__ = 0
__biological_markers__ = ["GFAP", "NeuN"]

__all__ = [
    "safety_officer",
    "sentinel_agent",
    "biological_protocols",
    "safety_constraints",
    "audit_system",
]
