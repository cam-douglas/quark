#!/usr/bin/env python3
"""Language Cortex - Clean interface to modular language processing system.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular language processing with secure credentials.
"""

# Import from modular language processing system
from .language_processing import LanguageCortex

# Re-export for backward compatibility
__all__ = ['LanguageCortex']
