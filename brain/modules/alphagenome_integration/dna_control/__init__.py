#!/usr/bin/env python3
"""DNA Control Module - Main interface for DNA control and analysis system.

Provides unified interface to DNA control components with preserved integrations.

Integration: Main DNA control interface for AlphaGenome biological workflows.
Rationale: Clean API abstraction maintaining all existing functionality.
"""

from typing import Optional
from .sequence_config import BiologicalSequenceConfig, get_default_neural_config, get_developmental_config
from .controller_core import DNAController

def create_dna_controller(config: Optional[BiologicalSequenceConfig] = None):
    """Factory function to create a DNAController with specified configuration."""
    return DNAController(config=config)

# Export main interface for backward compatibility
__all__ = [
    'DNAController',
    'BiologicalSequenceConfig',
    'get_default_neural_config',
    'get_developmental_config',
    'create_dna_controller'
]
