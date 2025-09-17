"""
Atlas registration pipeline for brainstem segmentation.

Implements affine + non-linear registration of MRI and histology data
to a common coordinate system using ANTs (Advanced Normalization Tools).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


class AtlasRegistrationPipeline:
    """Pipeline for registering embryonic brain data to common atlas space."""

    def __init__(self, data_dir: Path, template_name: str = "DevCCF_E13.5"):
        """Initialize registration pipeline.

        Args:
            data_dir: Base directory containing datasets
            template_name: Name of template atlas for registration
        """
        self.data_dir = Path(data_dir)
        self.template_name = template_name
        self.templates_dir = self.data_dir / "templates" / "devccf"
        self.registered_dir = self.data_dir / "registered"
        self.registered_dir.mkdir(parents=True, exist_ok=True)

    def register_mri_to_atlas(
        self, 
        moving_path: Path, 
        output_prefix: str,
        use_existing_transforms: bool = False
    ) -> Dict[str, Path]:
        """Register MRI volume to atlas template using ANTs.

        Args:
            moving_path: Path to moving image (to be registered)
            output_prefix: Output prefix for registered files
            use_existing_transforms: Whether to reuse existing transform files

        Returns:
            Dictionary with paths to registered outputs
        """
        template_path = self.templates_dir / f"{self.template_name}_Reference.nii.gz"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        output_dir = self.registered_dir / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        registered_image = output_dir / f"{output_prefix}_registered.nii.gz"
        transform_prefix = output_dir / f"{output_prefix}_transform"
        
        if use_existing_transforms and registered_image.exists():
            logger.info(f"Using existing registration: {registered_image}")
            return self._get_registration_outputs(output_dir, output_prefix)
        
        # ANTs registration command
        cmd = [
            "antsRegistration",
            "--dimensionality", "3",
            "--float", "0",
            "--output", f"[{transform_prefix},{registered_image}]",
            "--interpolation", "Linear",
            "--winsorize-image-intensities", "[0.005,0.995]",
            "--use-histogram-matching", "0",
            
            # Rigid registration
            "--transform", "Rigid[0.1]",
            "--metric", f"MI[{template_path},{moving_path},1,32,Regular,0.25]",
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            
            # Affine registration
            "--transform", "Affine[0.1]",
            "--metric", f"MI[{template_path},{moving_path},1,32,Regular,0.25]",
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            
            # Non-linear registration (SyN)
            "--transform", "SyN[0.1,3,0]",
            "--metric", f"CC[{template_path},{moving_path},1,4]",
            "--convergence", "[100x70x50x20,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox"
        ]
        
        logger.info(f"Running ANTs registration: {moving_path} -> {template_path}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                logger.error(f"ANTs registration failed: {result.stderr}")
                raise RuntimeError(f"Registration failed: {result.stderr}")
            
            logger.info(f"Registration completed: {registered_image}")
            return self._get_registration_outputs(output_dir, output_prefix)
            
        except subprocess.TimeoutExpired:
            logger.error("Registration timed out after 1 hour")
            raise RuntimeError("Registration timeout")
        except FileNotFoundError:
            logger.error("ANTs not found - install with: pip install antspyx")
            raise RuntimeError("ANTs not available")

    def _get_registration_outputs(self, output_dir: Path, prefix: str) -> Dict[str, Path]:
        """Get paths to registration output files."""
        return {
            "registered_image": output_dir / f"{prefix}_registered.nii.gz",
            "transform_composite": output_dir / f"{prefix}_transform1Composite.h5",
            "transform_inverse": output_dir / f"{prefix}_transform1InverseComposite.h5",
            "transform_affine": output_dir / f"{prefix}_transform0GenericAffine.mat",
            "output_directory": output_dir
        }

    def apply_transform_to_labels(
        self, 
        labels_path: Path, 
        transform_prefix: Path,
        output_path: Path
    ) -> bool:
        """Apply registration transform to label volume.

        Args:
            labels_path: Path to label volume
            transform_prefix: Prefix of transform files
            output_path: Output path for transformed labels

        Returns:
            True if successful
        """
        template_path = self.templates_dir / f"{self.template_name}_Reference.nii.gz"
        composite_transform = f"{transform_prefix}1Composite.h5"
        
        cmd = [
            "antsApplyTransforms",
            "--dimensionality", "3",
            "--input", str(labels_path),
            "--reference-image", str(template_path),
            "--output", str(output_path),
            "--interpolation", "NearestNeighbor",  # For labels
            "--transform", composite_transform
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"Transform application failed: {result.stderr}")
                return False
            
            logger.info(f"Labels transformed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Transform application error: {e}")
            return False

    def compute_landmark_error(
        self, 
        registered_labels: Path, 
        template_labels: Path,
        voxel_size_um: float = 50.0
    ) -> Dict[str, float]:
        """Compute landmark registration errors.

        Args:
            registered_labels: Transformed label volume
            template_labels: Template label volume
            voxel_size_um: Voxel size in micrometers

        Returns:
            Dictionary with landmark error statistics
        """
        try:
            reg_img = nib.load(str(registered_labels))
            temp_img = nib.load(str(template_labels))
            
            reg_data = reg_img.get_fdata().astype(int)
            temp_data = temp_img.get_fdata().astype(int)
            
            # Compute centroids for each label
            reg_centroids = self._compute_label_centroids(reg_data)
            temp_centroids = self._compute_label_centroids(temp_data)
            
            # Calculate distances between corresponding centroids
            errors = []
            common_labels = set(reg_centroids.keys()) & set(temp_centroids.keys())
            
            for label in common_labels:
                if label == 0:  # Skip background
                    continue
                reg_centroid = reg_centroids[label]
                temp_centroid = temp_centroids[label]
                
                # Euclidean distance in voxels, convert to micrometers
                distance_voxels = np.linalg.norm(np.array(reg_centroid) - np.array(temp_centroid))
                distance_um = distance_voxels * voxel_size_um
                errors.append(distance_um)
            
            if not errors:
                return {"error": "No common labels found"}
            
            return {
                "mean_error_um": float(np.mean(errors)),
                "median_error_um": float(np.median(errors)),
                "max_error_um": float(np.max(errors)),
                "p95_error_um": float(np.percentile(errors, 95)),
                "num_landmarks": len(errors),
                "common_labels": len(common_labels) - 1  # Exclude background
            }
            
        except Exception as e:
            logger.error(f"Landmark error computation failed: {e}")
            return {"error": str(e)}

    def _compute_label_centroids(self, labels: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
        """Compute centroid coordinates for each label."""
        centroids = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            mask = labels == label
            coords = np.where(mask)
            if len(coords[0]) > 0:
                centroid = (
                    float(np.mean(coords[0])),
                    float(np.mean(coords[1])),
                    float(np.mean(coords[2]))
                )
                centroids[int(label)] = centroid
        
        return centroids


def run_atlas_registration_pipeline(data_dir: Path) -> bool:
    """Execute complete atlas registration pipeline.

    Args:
        data_dir: Base directory containing datasets

    Returns:
        True if successful
    """
    try:
        pipeline = AtlasRegistrationPipeline(data_dir)
        
        # Check for input data
        nextbrain_volume = data_dir / "nextbrain" / "T2w.nii.gz"
        nextbrain_labels = data_dir / "nextbrain" / "manual_segmentation.nii.gz"
        
        if not nextbrain_volume.exists():
            logger.warning(f"NextBrain volume not found: {nextbrain_volume}")
            return False
        
        # Register MRI volume
        registration_outputs = pipeline.register_mri_to_atlas(
            nextbrain_volume, 
            "nextbrain_to_devccf"
        )
        
        # Apply transform to labels if available
        if nextbrain_labels.exists():
            registered_labels = registration_outputs["output_directory"] / "nextbrain_labels_registered.nii.gz"
            pipeline.apply_transform_to_labels(
                nextbrain_labels,
                registration_outputs["output_directory"] / "nextbrain_to_devccf_transform",
                registered_labels
            )
            
            # Compute landmark errors
            template_labels = pipeline.templates_dir / f"{pipeline.template_name}_Annotations.nii.gz"
            if template_labels.exists():
                errors = pipeline.compute_landmark_error(registered_labels, template_labels)
                
                # Save error report
                error_report = registration_outputs["output_directory"] / "landmark_error_report.json"
                with open(error_report, 'w') as f:
                    json.dump(errors, f, indent=2)
                
                logger.info(f"Landmark error report saved: {error_report}")
        
        return True
        
    except Exception as e:
        logger.error(f"Atlas registration pipeline failed: {e}")
        return False
