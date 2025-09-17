#!/usr/bin/env python3
"""Dataset Integration - Clean interface to modular dataset management.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular dataset management components.
"""

# Import from modular dataset management system
from .dataset_management import DatasetIntegration

# Re-export for backward compatibility
__all__ = ['DatasetIntegration']
