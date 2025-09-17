"""
AlphaGenome models module - main entry point for DNA analysis models.
"""

from typing import Optional

from .types import OutputType, GenomicInterval, Variant, PredictionOutputs
from .model_core import AlphaGenomeModel

# Re-export key types for convenience
__all__ = [
    'OutputType',
    'GenomicInterval',
    'Variant',
    'PredictionOutputs',
    'AlphaGenomeModel',
    'create',
    'get_default_model',
    'dna_client'
]


def create(api_key: Optional[str] = None, **kwargs) -> AlphaGenomeModel:
    """Factory function to create AlphaGenome model."""
    return AlphaGenomeModel(api_key=api_key, **kwargs)


# Module-level instance for convenience
dna_client = None

def get_default_model(api_key: Optional[str] = None) -> AlphaGenomeModel:
    """Get or create default model instance."""
    global dna_client
    if dna_client is None:
        dna_client = create(api_key)
    return dna_client
