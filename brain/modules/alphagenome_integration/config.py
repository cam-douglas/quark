#!/usr/bin/env python3
"""AlphaGenome Configuration - Interface to modular configuration system.

Integration: Configuration management for AlphaGenome biological workflows.
Rationale: Streamlined interface to modular configuration components.
"""

# Import from modular configuration system
from .configuration.config_core import *

# Re-export for backward compatibility
__all__ = ['AlphaGenomeConfig']
