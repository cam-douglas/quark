#!/usr/bin/env python3
"""
Brainstem Segmentation Inference Engine - Phase 4 Step 2.O2

Implements automatic brainstem segmentation for embryo simulation startup.
Integrates trained model inference into voxel-map builder workflow.

Author: Quark AI
Date: 2025-09-16
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for brainstem segmentation inference."""

    # Model parameters
    model_path: str = "/data/models/brainstem_segmentation/best_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Inference parameters
    patch_size: Tuple[int, int, int] = (64, 64, 64)
    stride: int = 32  # 50% overlap
    batch_size: int = 4

    # Memory management
    max_memory_gb: float = 8.0

    # Output parameters
    save_intermediates: bool = False
    output_dir: Optional[str] = None


class BrainstemInferenceEngine:
    """
    Inference engine for automatic brainstem segmentation.

    Loads trained model and performs segmentation on embryo volumes
    during simulation startup.
    """

    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.morphogen_processor = None
        self._load_model()

    def _load_model(self):
        """Load the trained brainstem segmentation model."""
        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}, segmentation will use fallback mode")
                self._create_fallback_segmentation()
                return

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=False)

            # Import model architecture (this will be available after training)
            try:
                from .model_architecture_designer import ViTGNNHybrid
                model = ViTGNNHybrid(
                    input_channels=1,
                    patch_size=(16, 16, 16),
                    embed_dim=768,
                    vit_layers=8,
                    gnn_layers=3,
                    num_heads=8,
                    num_classes=16  # Full 16-class schema
                )
            except ImportError:
                logger.error("Model architecture not available. Run training first.")
                return

            # Load trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.config.device)
            model.eval()

            self.model = model
            logger.info(f"Model loaded: {checkpoint['epoch']} epochs, "
                       f"Dice: {checkpoint.get('metrics', {}).get('dice', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self._create_fallback_segmentation()

    def _create_fallback_segmentation(self):
        """Create a fallback segmentation method for testing when model is not available."""
        logger.info("Creating fallback segmentation method for testing")
        self.fallback_mode = True
        self.model = None  # Explicitly set to None to trigger fallback in segment_volume

    def _load_morphogen_processor(self):
        """Load morphogen processing utilities."""
        try:
            from .morphogen_integration import MorphogenDataProcessor
            self.morphogen_processor = MorphogenDataProcessor()
        except ImportError:
            logger.warning("Morphogen processor not available, using dummy priors")
            self.morphogen_processor = None

    def segment_volume(self, volume: np.ndarray,
                      morphogen_data: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Segment brainstem in input volume using trained model or fallback method.

        Args:
            volume: Input 3D volume (H, W, D)
            morphogen_data: Optional morphogen concentration data

        Returns:
            Segmentation mask or None if inference fails
        """
        if self.model is None:
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                logger.info("Using fallback segmentation method")
                return self._fallback_segmentation(volume, morphogen_data)
            else:
                logger.warning("Model not loaded and no fallback available, skipping segmentation")
                return None

        try:
            start_time = time.time()

            # Normalize volume
            volume_normalized = self._normalize_volume(volume)

            # Generate morphogen priors if not provided
            if morphogen_data is None and self.morphogen_processor is not None:
                morphogen_data = self.morphogen_processor.create_morphogen_priors_for_volume(volume.shape)

            # Perform patch-based inference
            segmentation = self._patch_based_inference(volume_normalized, morphogen_data)

            inference_time = time.time() - start_time
            logger.info(".2f")

            return segmentation

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume using z-score normalization."""
        volume = volume.astype(np.float32)
        return (volume - volume.mean()) / (volume.std() + 1e-8)

    def _patch_based_inference(self, volume: np.ndarray,
                              morphogen_data: Optional[np.ndarray]) -> np.ndarray:
        """
        Perform patch-based inference for memory efficiency.

        Args:
            volume: Normalized volume (H, W, D)
            morphogen_data: Morphogen priors (3, H, W, D) or None

        Returns:
            Segmentation mask (H, W, D)
        """
        h, w, d = volume.shape
        patch_size = self.config.patch_size
        stride = self.config.stride

        # Initialize output volume
        num_classes = 16  # Full brainstem schema
        output_volume = np.zeros((num_classes, h, w, d), dtype=np.float32)

        # Track patch counts for averaging
        patch_counts = np.zeros((h, w, d), dtype=np.int32)

        patches_processed = 0

        with torch.no_grad():
            # Extract and process patches
            for z in range(0, d - patch_size[2] + 1, stride):
                for y in range(0, w - patch_size[1] + 1, stride):
                    for x in range(0, h - patch_size[0] + 1, stride):

                        # Extract volume patch
                        vol_patch = volume[x:x+patch_size[0],
                                         y:y+patch_size[1],
                                         z:z+patch_size[2]]

                        # Extract morphogen patch if available
                        morph_patch = None
                        if morphogen_data is not None:
                            morph_patch = morphogen_data[:, x:x+patch_size[0],
                                                       y:y+patch_size[1],
                                                       z:z+patch_size[2]]

                        # Convert to tensors
                        vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0).unsqueeze(0).float()
                        vol_tensor = vol_tensor.to(self.config.device)

                        if morph_patch is not None:
                            morph_tensor = torch.from_numpy(morph_patch).unsqueeze(0).float()
                            morph_tensor = morph_tensor.to(self.config.device)
                        else:
                            morph_tensor = None

                        # Inference
                        if morph_tensor is not None:
                            # Model with morphogen input
                            output = self.model(vol_tensor, morph_tensor)
                        else:
                            # Model without morphogen input
                            output = self.model(vol_tensor)

                        output_patch = output.squeeze().cpu().numpy()

                        # Accumulate predictions
                        output_volume[:, x:x+patch_size[0],
                                    y:y+patch_size[1],
                                    z:z+patch_size[2]] += output_patch

                        # Track patch counts
                        patch_counts[x:x+patch_size[0],
                                   y:y+patch_size[1],
                                   z:z+patch_size[2]] += 1

                        patches_processed += 1

                        # Memory management: break if approaching memory limit
                        if patches_processed % 100 == 0:
                            if torch.cuda.is_available():
                                memory_gb = torch.cuda.memory_allocated() / (1024**3)
                                if memory_gb > self.config.max_memory_gb:
                                    logger.warning(".1f")
                                    break

        # Average overlapping predictions
        output_volume = output_volume / np.maximum(patch_counts, 1)

        # Get final segmentation
        segmentation = np.argmax(output_volume, axis=0)

        return segmentation.astype(np.int32)

    def _fallback_segmentation(self, volume: np.ndarray,
                              morphogen_data: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Fallback segmentation method for testing when trained model is not available.

        Creates a simple rule-based segmentation based on anatomical heuristics.

        Args:
            volume: Input 3D volume (H, W, D)
            morphogen_data: Optional morphogen concentration data

        Returns:
            Segmentation mask with basic anatomical regions
        """
        try:
            logger.info("Running fallback brainstem segmentation (rule-based)")

            h, w, d = volume.shape
            segmentation = np.zeros((h, w, d), dtype=np.int32)

            # Normalize volume for processing
            volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

            # Define anatomical regions based on relative positions
            # Brainstem is typically in the posterior-ventral region
            brainstem_start_x = int(0.3 * h)  # Start at 30% from anterior
            brainstem_end_x = int(0.7 * h)    # End at 70% from anterior
            brainstem_start_y = int(0.6 * w)  # Start at 60% from dorsal
            brainstem_end_y = w               # Go to ventral surface
            brainstem_start_z = int(0.4 * d)  # Central left-right
            brainstem_end_z = int(0.6 * d)

            # Create brainstem region (label 1)
            segmentation[brainstem_start_x:brainstem_end_x,
                        brainstem_start_y:brainstem_end_y,
                        brainstem_start_z:brainstem_end_z] = 1

            # Subdivide into midbrain, pons, medulla based on anterior-posterior position
            midbrain_end = brainstem_start_x + int(0.3 * (brainstem_end_x - brainstem_start_x))
            pons_end = brainstem_start_x + int(0.7 * (brainstem_end_x - brainstem_start_x))

            # Midbrain (label 2) - most anterior part
            segmentation[brainstem_start_x:midbrain_end,
                        brainstem_start_y:brainstem_end_y,
                        brainstem_start_z:brainstem_end_z] = 2

            # Pons (label 3) - middle part
            segmentation[midbrain_end:pons_end,
                        brainstem_start_y:brainstem_end_y,
                        brainstem_start_z:brainstem_end_z] = 3

            # Medulla (label 4) - most posterior part
            segmentation[pons_end:brainstem_end_x,
                        brainstem_start_y:brainstem_end_y,
                        brainstem_start_z:brainstem_end_z] = 4

            # Add some anatomical nuclei based on intensity patterns
            high_intensity_mask = volume_norm > np.percentile(volume_norm, 75)

            # Red nucleus (label 5) - in midbrain, high intensity
            midbrain_mask = (segmentation == 2)
            red_nucleus_mask = midbrain_mask & high_intensity_mask
            segmentation[red_nucleus_mask] = 5

            # Locus coeruleus (label 6) - in pons, distinct pattern
            pons_mask = (segmentation == 3)
            locus_mask = pons_mask & (volume_norm > np.percentile(volume_norm[pons_mask], 80))
            segmentation[locus_mask] = 6

            logger.info(f"Fallback segmentation complete: {np.sum(segmentation > 0)} voxels segmented")
            logger.info(f"Regions identified: {len(np.unique(segmentation))} unique labels")

            return segmentation

        except Exception as e:
            logger.error(f"Fallback segmentation failed: {e}")
            return None

    def segment_from_nifti(self, nifti_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None) -> Optional[np.ndarray]:
        """
        Segment brainstem from NIFTI file.

        Args:
            nifti_path: Path to input NIFTI file
            output_path: Optional path to save segmentation

        Returns:
            Segmentation mask or None if failed
        """
        try:
            # Load volume
            img = nib.load(str(nifti_path))
            volume = img.get_fdata()

            # Perform segmentation
            segmentation = self.segment_volume(volume)

            if segmentation is not None and output_path is not None:
                # Save segmentation as NIFTI
                seg_img = nib.Nifti1Image(segmentation, img.affine, img.header)
                nib.save(seg_img, str(output_path))
                logger.info(f"Segmentation saved to {output_path}")

            return segmentation

        except Exception as e:
            logger.error(f"Failed to segment NIFTI file {nifti_path}: {e}")
            return None

    def get_brainstem_regions(self, segmentation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract anatomical regions from segmentation mask.

        Args:
            segmentation: Segmentation mask

        Returns:
            Dictionary mapping region names to binary masks
        """
        regions = {}

        # Midbrain regions
        regions['midbrain'] = (segmentation == 1)
        regions['red_nucleus'] = (segmentation == 2)
        regions['substantia_nigra'] = (segmentation == 3)

        # Pons regions
        regions['pons'] = (segmentation == 4)
        regions['locus_coeruleus'] = (segmentation == 5)
        regions['pontine_nuclei'] = (segmentation == 6)

        # Medulla regions
        regions['medulla'] = (segmentation == 7)
        regions['raphe_nuclei'] = (segmentation == 8)
        regions['nucleus_ambiguus'] = (segmentation == 9)

        return regions

    def get_segmentation_stats(self, segmentation: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics for segmentation results.

        Args:
            segmentation: Segmentation mask

        Returns:
            Dictionary with segmentation statistics
        """
        unique_labels, counts = np.unique(segmentation, return_counts=True)

        stats = {
            'total_voxels': segmentation.size,
            'segmented_voxels': np.sum(segmentation > 0),
            'background_voxels': np.sum(segmentation == 0),
            'segmentation_coverage': np.sum(segmentation > 0) / segmentation.size,
            'num_regions': len(unique_labels) - (1 if 0 in unique_labels else 0),
            'region_counts': {f'class_{label}': count for label, count in zip(unique_labels, counts)}
        }

        return stats


def auto_segment_brainstem(volume: np.ndarray,
                          morphogen_data: Optional[np.ndarray] = None,
                          config: InferenceConfig = None) -> Optional[np.ndarray]:
    """
    Convenience function for automatic brainstem segmentation.

    Args:
        volume: Input 3D volume
        morphogen_data: Optional morphogen concentration data
        config: Inference configuration

    Returns:
        Segmentation mask or None if failed
    """
    engine = BrainstemInferenceEngine(config)
    return engine.segment_volume(volume, morphogen_data)
