#!/usr/bin/env python3
"""Quantum Braket Integration - Interface to modular quantum integration.

Integration: Quantum computing integration for AlphaGenome biological workflows.
Rationale: Streamlined interface to modular quantum integration components.
"""

# Import from modular quantum integration system
from .quantum_integration.braket_core import *

# Re-export for backward compatibility
__all__ = ['QuantumBraketIntegration']
