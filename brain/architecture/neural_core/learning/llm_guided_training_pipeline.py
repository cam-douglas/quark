#!/usr/bin/env python3
"""LLM Guided Training Pipeline - Clean interface to modular training system.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Streamlined interface to modular LLM-guided training components.
"""

# Import from modular training pipeline system
from .training_pipeline import LLMGuidedTrainingPipeline

# Re-export for backward compatibility
__all__ = ['LLMGuidedTrainingPipeline']
