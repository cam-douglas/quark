#!/usr/bin/env python3
"""Resource Manager - Clean interface to modular resource management.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular resource management components.
"""

# Import from modular resource management system
from .resource_management import ResourceManager

# Re-export for backward compatibility
__all__ = ['ResourceManager']
