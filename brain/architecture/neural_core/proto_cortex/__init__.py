"""Proto-Cortex Module

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from .layer_sheet import LayerSheet  # noqa: F401

__all__ = ["LayerSheet"]
