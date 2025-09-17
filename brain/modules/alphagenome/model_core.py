"""
AlphaGenome core model - main prediction model class.
"""

import os
import json
import hashlib
import logging
from typing import List, Optional

import numpy as np

from .types import GenomicInterval, Variant, PredictionOutputs, OutputType
from .mock_data import generate_mock_predictions

logger = logging.getLogger(__name__)


class AlphaGenomeModel:
    """AlphaGenome DNA sequence analysis model."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        self.api_key = api_key
        self.cache_dir = cache_dir or "/tmp/alphagenome_cache"
        self.model_loaded = False

        # Initialize model parameters (mock for now)
        self._initialize_model()

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("AlphaGenome model initialized")

    def _initialize_model(self):
        """Initialize model parameters."""
        # In a real implementation, this would load model weights
        self.model_params = {
            'sequence_embedding_dim': 256,
            'max_sequence_length': 1000000,
            'output_resolution': 128,  # predictions every 128bp
            'model_version': '1.0'
        }
        self.model_loaded = True

    def _get_cache_key(self, interval: GenomicInterval, variant: Optional[Variant] = None) -> str:
        """Generate cache key for predictions."""
        key_parts = [str(interval)]
        if variant:
            key_parts.append(str(variant))
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[PredictionOutputs]:
        """Check if predictions exist in cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                # Reconstruct PredictionOutputs
                outputs = PredictionOutputs()
                for key, value in data.items():
                    if key in ['rna_seq', 'atac', 'histone_h3k27ac', 'histone_h3k27me3',
                              'histone_h3k9me3', 'dnase', 'cage', 'chi_c', 'conservation',
                              'regulatory_score'] and value is not None:
                        setattr(outputs, key, np.array(value))
                    elif key == 'interval' and value:
                        outputs.interval = GenomicInterval(**value)
                    else:
                        setattr(outputs, key, value)
                logger.info(f"Loaded predictions from cache: {cache_key}")
                return outputs
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, outputs: PredictionOutputs):
        """Save predictions to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(outputs.to_dict(), f)
            logger.info(f"Saved predictions to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def predict_interval(self, interval: GenomicInterval,
                        ontology_terms: Optional[List[str]] = None,
                        requested_outputs: Optional[List[OutputType]] = None) -> PredictionOutputs:
        """Predict molecular phenotypes for a genomic interval."""

        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        # Default to all outputs if none specified
        if requested_outputs is None:
            requested_outputs = list(OutputType)

        # Check cache first
        cache_key = self._get_cache_key(interval)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Generate predictions (mock for now)
        outputs = generate_mock_predictions(interval, requested_outputs, self.model_params)

        # Save to cache
        self._save_to_cache(cache_key, outputs)

        return outputs

    def predict_variant(self, interval: GenomicInterval,
                       variant: Variant,
                       ontology_terms: Optional[List[str]] = None,
                       requested_outputs: Optional[List[OutputType]] = None) -> PredictionOutputs:
        """Predict molecular phenotypes for a variant."""

        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        # Default to all outputs if none specified
        if requested_outputs is None:
            requested_outputs = list(OutputType)

        # Check cache first
        cache_key = self._get_cache_key(interval, variant)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # For variants, generate predictions with perturbations
        base_outputs = generate_mock_predictions(interval, requested_outputs, self.model_params)

        # Add variant effects (mock perturbations)
        variant_pos_bin = (variant.position - interval.start) // self.model_params['output_resolution']

        for output_type in requested_outputs:
            track = getattr(base_outputs, output_type.value)
            if track is not None and 0 <= variant_pos_bin < len(track):
                # Apply variant effect in surrounding region
                effect_size = np.random.uniform(-0.5, 0.5)
                effect_range = 10  # bins affected
                start = max(0, variant_pos_bin - effect_range // 2)
                end = min(len(track), variant_pos_bin + effect_range // 2)

                # Apply gaussian-weighted effect
                positions = np.arange(start, end)
                weights = np.exp(-0.5 * ((positions - variant_pos_bin) / 3) ** 2)
                track[start:end] += effect_size * weights
                track[track < 0] = 0  # Ensure non-negative

        base_outputs.variant = variant

        # Save to cache
        self._save_to_cache(cache_key, base_outputs)

        return base_outputs
