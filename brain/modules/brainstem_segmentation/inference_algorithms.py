"""
Algorithms and utilities for brainstem segmentation inference.

This module contains standalone, testable functions used by
`BrainstemInferenceEngine` to keep the engine file concise and compliant
with Quark's file-size limits.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize a 3-D volume.

    Args:
        volume: Raw input volume of shape (H, W, D).

    Returns:
        Normalized volume with zero mean and unit variance.
    """
    vol = volume.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-8)


def patch_based_inference(
    model: torch.nn.Module,
    device: str,
    volume: np.ndarray,
    morphogen_data: Optional[np.ndarray],
    patch_size: Tuple[int, int, int],
    stride: int,
    num_classes: int,
    max_memory_gb: float,
) -> np.ndarray:
    """Run memory-efficient patch-based inference.

    Args:
        model: Segmentation model in evaluation mode.
        device: Torch device string.
        volume: Normalized volume (H, W, D).
        morphogen_data: Optional morphogen priors (C, H, W, D).
        patch_size: Sliding-window patch size.
        stride: Patch stride.
        num_classes: Number of output classes.
        max_memory_gb: Soft VRAM limit for safety logging.

    Returns:
        Argmax label volume of shape (H, W, D) with dtype int32.
    """
    h, w, d = volume.shape

    output_volume = np.zeros((num_classes, h, w, d), dtype=np.float32)
    patch_counts = np.zeros((h, w, d), dtype=np.int32)

    patches_processed = 0

    with torch.no_grad():
        for z in range(0, d - patch_size[2] + 1, stride):
            for y in range(0, w - patch_size[1] + 1, stride):
                for x in range(0, h - patch_size[0] + 1, stride):
                    vol_patch = volume[x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]]

                    morph_tensor = None
                    if morphogen_data is not None:
                        morph_patch = morphogen_data[
                            :, x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]
                        ]
                        morph_tensor = torch.from_numpy(morph_patch).unsqueeze(0).float().to(device)

                    vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0).unsqueeze(0).float().to(device)

                    if morph_tensor is not None:
                        output = model(vol_tensor, morph_tensor)
                    else:
                        output = model(vol_tensor)

                    output_patch = output.squeeze().cpu().numpy()

                    output_volume[:, x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]] += output_patch
                    patch_counts[x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]] += 1

                    patches_processed += 1
                    if patches_processed % 100 == 0 and torch.cuda.is_available():
                        memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                        if memory_gb > max_memory_gb:
                            logger.warning(
                                "VRAM usage high: %.1f GiB exceeds configured soft limit %.1f GiB",
                                memory_gb,
                                max_memory_gb,
                            )

    output_volume = output_volume / np.maximum(patch_counts, 1)
    segmentation = np.argmax(output_volume, axis=0).astype(np.int32)
    return segmentation


def fallback_segmentation(volume: np.ndarray) -> np.ndarray:
    """Simple rule-based fallback segmentation for testing.

    Args:
        volume: Input 3-D volume (H, W, D).

    Returns:
        Integer label volume (H, W, D).
    """
    h, w, d = volume.shape
    seg = np.zeros((h, w, d), dtype=np.int32)

    volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    x0, x1 = int(0.3 * h), int(0.7 * h)
    y0, y1 = int(0.6 * w), w
    z0, z1 = int(0.4 * d), int(0.6 * d)

    seg[x0:x1, y0:y1, z0:z1] = 1  # brainstem coarse region

    mid_end = x0 + int(0.3 * (x1 - x0))
    pons_end = x0 + int(0.7 * (x1 - x0))
    seg[x0:mid_end, y0:y1, z0:z1] = 2
    seg[mid_end:pons_end, y0:y1, z0:z1] = 3
    seg[pons_end:x1, y0:y1, z0:z1] = 4

    high_mask = volume_norm > np.percentile(volume_norm, 75)
    mid_mask = seg == 2
    pons_mask = seg == 3
    seg[mid_mask & high_mask] = 5
    seg[pons_mask & (volume_norm > np.percentile(volume_norm[pons_mask], 80))] = 6

    return seg


def get_brainstem_regions(segmentation: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute binary masks for key anatomical regions from labels."""
    regions: Dict[str, np.ndarray] = {
        "midbrain": segmentation == 1,
        "red_nucleus": segmentation == 2,
        "substantia_nigra": segmentation == 3,
        "pons": segmentation == 4,
        "locus_coeruleus": segmentation == 5,
        "pontine_nuclei": segmentation == 6,
        "medulla": segmentation == 7,
        "raphe_nuclei": segmentation == 8,
        "nucleus_ambiguus": segmentation == 9,
    }
    return regions


def get_segmentation_stats(segmentation: np.ndarray) -> Dict[str, float | int | Dict[str, int]]:
    """Compute coverage and counts for a segmentation mask."""
    unique_labels, counts = np.unique(segmentation, return_counts=True)
    stats: Dict[str, float | int | Dict[str, int]] = {
        "total_voxels": int(segmentation.size),
        "segmented_voxels": int(np.sum(segmentation > 0)),
        "background_voxels": int(np.sum(segmentation == 0)),
        "segmentation_coverage": float(np.sum(segmentation > 0) / segmentation.size),
        "num_regions": int(len(unique_labels) - (1 if 0 in unique_labels else 0)),
        "region_counts": {f"class_{int(label)}": int(count) for label, count in zip(unique_labels, counts)},
    }
    return stats


def compute_overall_dice(
    prediction: np.ndarray, target: np.ndarray, num_classes: int
) -> float:
    """Compute mean Dice across classes 1..num_classes-1 (exclude background).

    Args:
        prediction: Integer label volume.
        target: Integer label volume (manual spot-checks).
        num_classes: Total number of classes, including background 0.

    Returns:
        Overall Dice in [0, 1]. Returns 0.0 if no foreground classes present.
    """
    dices: list[float] = []
    for cls in range(1, num_classes):
        pred_mask = prediction == cls
        tgt_mask = target == cls
        intersection = float(np.logical_and(pred_mask, tgt_mask).sum())
        union = float(pred_mask.sum() + tgt_mask.sum())
        if union == 0:
            # If both are empty for this class, treat as perfect agreement
            dices.append(1.0)
        else:
            dices.append((2.0 * intersection + 1e-6) / (union + 1e-6))
    return float(np.mean(dices)) if dices else 0.0


