"""
AlphaGenome mock data generation - creates realistic test data.
"""

import numpy as np
from typing import List, Dict, Any

from .types import GenomicInterval, PredictionOutputs, OutputType


def generate_mock_predictions(interval: GenomicInterval,
                            requested_outputs: List[OutputType],
                            model_params: Dict[str, Any]) -> PredictionOutputs:
    """Generate mock predictions for testing.
    
    Args:
        interval: Genomic interval to predict for
        requested_outputs: List of output types to generate
        model_params: Model parameters including resolution
        
    Returns:
        PredictionOutputs with mock data
    """
    seq_length = interval.length
    resolution = model_params['output_resolution']
    num_bins = seq_length // resolution + 1

    outputs = PredictionOutputs(
        interval=interval,
        model_version=model_params['model_version']
    )

    # Generate realistic-looking mock data for each requested output
    for output_type in requested_outputs:
        if output_type == OutputType.RNA_SEQ:
            data = _generate_expression_data(num_bins)
            outputs.rna_seq = data
            outputs.confidence_scores['rna_seq'] = 0.85

        elif output_type == OutputType.ATAC:
            data = _generate_accessibility_data(num_bins)
            outputs.atac = data
            outputs.confidence_scores['atac'] = 0.82

        elif output_type == OutputType.CONSERVATION:
            data = _generate_conservation_data(num_bins)
            outputs.conservation = data
            outputs.confidence_scores['conservation'] = 0.90

        elif output_type == OutputType.REGULATORY_SCORE:
            data = _generate_regulatory_data(num_bins)
            outputs.regulatory_score = data
            outputs.confidence_scores['regulatory_score'] = 0.78

        else:
            # Generic histone/other marks
            data = np.random.gamma(2, 0.5, num_bins)
            data = np.clip(data, 0, 5)
            setattr(outputs, output_type.value, data)
            outputs.confidence_scores[output_type.value] = 0.75

    return outputs


def _generate_expression_data(num_bins: int) -> np.ndarray:
    """Generate realistic gene expression data."""
    # Gene expression: sparse with some peaks
    data = np.random.exponential(0.1, num_bins)
    data[data > 1] = 0  # Most regions have low expression

    # Add some expression peaks
    for _ in range(np.random.randint(1, 5)):
        peak_pos = np.random.randint(0, num_bins)
        peak_width = np.random.randint(5, 20)
        start = max(0, peak_pos - peak_width // 2)
        end = min(num_bins, peak_pos + peak_width // 2)
        data[start:end] += np.random.uniform(2, 10)

    return data


def _generate_accessibility_data(num_bins: int) -> np.ndarray:
    """Generate realistic chromatin accessibility data."""
    # Chromatin accessibility: broader peaks
    data = np.random.beta(0.5, 5, num_bins) * 2

    # Add accessibility peaks
    for _ in range(np.random.randint(3, 8)):
        peak_pos = np.random.randint(0, num_bins)
        peak_width = np.random.randint(10, 50)
        start = max(0, peak_pos - peak_width // 2)
        end = min(num_bins, peak_pos + peak_width // 2)
        data[start:end] += np.random.uniform(1, 3)

    return data


def _generate_conservation_data(num_bins: int) -> np.ndarray:
    """Generate realistic conservation scores."""
    # Conservation: smoother signal
    data = np.random.beta(2, 2, num_bins)

    # Apply smoothing
    from scipy.ndimage import gaussian_filter1d
    data = gaussian_filter1d(data, sigma=5)

    return data


def _generate_regulatory_data(num_bins: int) -> np.ndarray:
    """Generate realistic regulatory potential scores."""
    # Regulatory potential: combination of other signals
    data = np.random.beta(1, 3, num_bins)

    # Add regulatory elements
    for _ in range(np.random.randint(2, 6)):
        elem_pos = np.random.randint(0, num_bins)
        elem_width = np.random.randint(5, 15)
        start = max(0, elem_pos - elem_width // 2)
        end = min(num_bins, elem_pos + elem_width // 2)
        data[start:end] = np.maximum(data[start:end], np.random.uniform(0.7, 1.0))

    return data
