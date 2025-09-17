#!/usr/bin/env python3
"""DNA Controller - Clean interface to modular DNA control system.

Integration: This module participates in DNA workflows via DNAController.
Rationale: Streamlined interface to modular DNA control components.
"""

# Import from modular DNA control system
from .dna_control import (
    DNAController,
    BiologicalSequenceConfig,
    get_default_neural_config,
    get_developmental_config,
    create_dna_controller
)

# Re-export for backward compatibility
__all__ = [
    'DNAController', 'BiologicalSequenceConfig',
    'get_default_neural_config', 'get_developmental_config',
    'create_dna_controller'
]
