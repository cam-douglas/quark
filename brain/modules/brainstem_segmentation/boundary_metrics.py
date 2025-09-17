"""
Boundary error measurement utilities for brainstem segmentation validation.

Computes distance-transform based boundary errors between predicted and reference
segmentations to validate the ±200 µm accuracy requirement.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple


def measure_boundary_error(
    prediction: np.ndarray, 
    reference: np.ndarray, 
    voxel_size_um: float = 50.0
) -> np.ndarray:
    """Compute per-voxel boundary errors between prediction and reference.

    Args:
        prediction: Integer label volume (H, W, D).
        reference: Integer label volume (H, W, D), same shape as prediction.
        voxel_size_um: Voxel size in micrometers for unit conversion.

    Returns:
        Array of boundary errors in micrometers for each mismatched voxel.
    """
    # Find mismatched voxels
    mismatch_mask = (prediction != reference)
    
    if not np.any(mismatch_mask):
        # Perfect match - return zero error
        return np.array([0.0])
    
    # For each mismatched voxel, find distance to nearest correct boundary
    errors_um = []
    
    # Get unique labels in reference
    unique_labels = np.unique(reference)
    
    for label in unique_labels:
        # Find voxels that should be this label but aren't
        should_be_label = (reference == label)
        incorrectly_labeled = mismatch_mask & should_be_label
        
        if not np.any(incorrectly_labeled):
            continue
            
        # Find correct boundaries for this label
        correct_boundary = (prediction == label)
        
        if not np.any(correct_boundary):
            # No correct voxels for this label - use large error
            num_incorrect = np.sum(incorrectly_labeled)
            errors_um.extend([1000.0] * num_incorrect)  # 1mm error
            continue
            
        # Compute distance transform from correct boundaries
        distance_map = distance_transform_edt(~correct_boundary)
        
        # Extract errors for incorrectly labeled voxels
        incorrect_coords = np.where(incorrectly_labeled)
        voxel_errors = distance_map[incorrect_coords] * voxel_size_um
        errors_um.extend(voxel_errors.tolist())
    
    return np.array(errors_um)


def compute_boundary_statistics(
    prediction: np.ndarray,
    reference: np.ndarray, 
    voxel_size_um: float = 50.0
) -> dict[str, float]:
    """Compute comprehensive boundary error statistics.

    Args:
        prediction: Integer label volume.
        reference: Integer label volume.
        voxel_size_um: Voxel size in micrometers.

    Returns:
        Dictionary with error statistics in micrometers.
    """
    errors_um = measure_boundary_error(prediction, reference, voxel_size_um)
    
    if len(errors_um) == 0:
        return {
            "mean_error_um": 0.0,
            "median_error_um": 0.0,
            "p95_error_um": 0.0,
            "max_error_um": 0.0,
            "num_mismatched_voxels": 0
        }
    
    return {
        "mean_error_um": float(np.mean(errors_um)),
        "median_error_um": float(np.median(errors_um)),
        "p95_error_um": float(np.percentile(errors_um, 95)),
        "max_error_um": float(np.max(errors_um)),
        "num_mismatched_voxels": len(errors_um)
    }
