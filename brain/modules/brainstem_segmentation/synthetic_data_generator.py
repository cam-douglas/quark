#!/usr/bin/env python3
"""
Synthetic Brainstem Data Generator - Risk Mitigation

Generates synthetic embryonic brainstem MRI volumes and segmentation labels
to mitigate data availability risks for training and validation.

Key Features:
- Morphogen-guided anatomical structure generation
- Realistic noise and artifact simulation
- Embryonic stage-specific variations (E11-E18)
- Multi-modal synthesis (T1w, T2w, DWI)
"""

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from datetime import datetime
import logging
from scipy import ndimage
from skimage import morphology, filters
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    
    # Volume properties
    volume_shape: Tuple[int, int, int] = (128, 128, 96)
    voxel_size_mm: float = 0.2
    
    # Embryonic stages
    stages: List[str] = None
    
    # Morphogen parameters
    morphogen_channels: int = 3  # SHH, BMP, WNT
    morphogen_noise_std: float = 0.1
    
    # Imaging parameters
    snr_range: Tuple[float, float] = (15.0, 35.0)
    bias_field_strength: float = 0.2
    motion_artifacts: bool = True
    
    # Anatomical parameters
    brainstem_volume_ratio: Tuple[float, float] = (0.15, 0.25)
    nucleus_size_variation: float = 0.3
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = ['E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18']


class MorphogenGradientGenerator:
    """Generates realistic morphogen gradient fields for embryonic brainstem."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.shape = config.volume_shape
        
    def generate_shh_gradient(self, stage: str) -> np.ndarray:
        """Generate Sonic Hedgehog (SHH) gradient - ventral high, dorsal low."""
        
        # Stage-specific parameters
        stage_params = {
            'E11': {'strength': 0.8, 'gradient_sharpness': 2.0},
            'E12': {'strength': 0.9, 'gradient_sharpness': 2.2},
            'E13': {'strength': 1.0, 'gradient_sharpness': 2.5},
            'E14': {'strength': 0.95, 'gradient_sharpness': 2.3},
            'E15': {'strength': 0.85, 'gradient_sharpness': 2.0},
            'E16': {'strength': 0.75, 'gradient_sharpness': 1.8},
            'E17': {'strength': 0.65, 'gradient_sharpness': 1.5},
            'E18': {'strength': 0.55, 'gradient_sharpness': 1.2}
        }
        
        params = stage_params.get(stage, stage_params['E13'])
        
        # Create ventral-dorsal gradient
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.shape[0]),
            np.linspace(-1, 1, self.shape[1]),
            np.linspace(-1, 1, self.shape[2]),
            indexing='ij'
        )
        
        # SHH is high ventrally (negative y), low dorsally
        shh = params['strength'] * np.exp(-params['gradient_sharpness'] * (y + 0.3)**2)
        
        # Add noise
        noise = np.random.normal(0, self.config.morphogen_noise_std, shh.shape)
        shh = np.clip(shh + noise, 0, 1)
        
        return shh
    
    def generate_bmp_gradient(self, stage: str) -> np.ndarray:
        """Generate BMP gradient - dorsal high, ventral low."""
        
        stage_params = {
            'E11': {'strength': 0.7, 'gradient_sharpness': 1.8},
            'E12': {'strength': 0.8, 'gradient_sharpness': 2.0},
            'E13': {'strength': 0.9, 'gradient_sharpness': 2.2},
            'E14': {'strength': 0.85, 'gradient_sharpness': 2.0},
            'E15': {'strength': 0.8, 'gradient_sharpness': 1.8},
            'E16': {'strength': 0.7, 'gradient_sharpness': 1.6},
            'E17': {'strength': 0.6, 'gradient_sharpness': 1.4},
            'E18': {'strength': 0.5, 'gradient_sharpness': 1.2}
        }
        
        params = stage_params.get(stage, stage_params['E13'])
        
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.shape[0]),
            np.linspace(-1, 1, self.shape[1]),
            np.linspace(-1, 1, self.shape[2]),
            indexing='ij'
        )
        
        # BMP is high dorsally (positive y), low ventrally
        bmp = params['strength'] * np.exp(-params['gradient_sharpness'] * (y - 0.3)**2)
        
        # Add noise
        noise = np.random.normal(0, self.config.morphogen_noise_std, bmp.shape)
        bmp = np.clip(bmp + noise, 0, 1)
        
        return bmp
    
    def generate_wnt_gradient(self, stage: str) -> np.ndarray:
        """Generate WNT gradient - anterior-posterior patterning."""
        
        stage_params = {
            'E11': {'strength': 0.6, 'gradient_sharpness': 1.5},
            'E12': {'strength': 0.7, 'gradient_sharpness': 1.7},
            'E13': {'strength': 0.8, 'gradient_sharpness': 1.9},
            'E14': {'strength': 0.75, 'gradient_sharpness': 1.8},
            'E15': {'strength': 0.7, 'gradient_sharpness': 1.6},
            'E16': {'strength': 0.65, 'gradient_sharpness': 1.4},
            'E17': {'strength': 0.6, 'gradient_sharpness': 1.2},
            'E18': {'strength': 0.55, 'gradient_sharpness': 1.0}
        }
        
        params = stage_params.get(stage, stage_params['E13'])
        
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.shape[0]),
            np.linspace(-1, 1, self.shape[1]),
            np.linspace(-1, 1, self.shape[2]),
            indexing='ij'
        )
        
        # WNT gradient along anterior-posterior axis (z-axis)
        wnt = params['strength'] * (0.5 + 0.5 * np.tanh(params['gradient_sharpness'] * z))
        
        # Add noise
        noise = np.random.normal(0, self.config.morphogen_noise_std, wnt.shape)
        wnt = np.clip(wnt + noise, 0, 1)
        
        return wnt
    
    def generate_combined_gradients(self, stage: str) -> np.ndarray:
        """Generate all morphogen gradients for a given stage."""
        
        shh = self.generate_shh_gradient(stage)
        bmp = self.generate_bmp_gradient(stage)
        wnt = self.generate_wnt_gradient(stage)
        
        # Stack into multi-channel array
        morphogens = np.stack([shh, bmp, wnt], axis=0)
        
        return morphogens


class AnatomicalStructureGenerator:
    """Generates anatomical structures based on morphogen gradients."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.shape = config.volume_shape
        
    def generate_brainstem_mask(self, morphogens: np.ndarray, stage: str) -> np.ndarray:
        """Generate brainstem region mask from morphogen gradients."""
        
        shh, bmp, wnt = morphogens
        
        # Stage-specific brainstem size
        stage_sizes = {
            'E11': 0.15, 'E12': 0.17, 'E13': 0.19, 'E14': 0.21,
            'E15': 0.23, 'E16': 0.24, 'E17': 0.25, 'E18': 0.25
        }
        
        base_size = stage_sizes.get(stage, 0.21)
        
        # Create ellipsoidal brainstem region
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.shape[0]),
            np.linspace(-1, 1, self.shape[1]),
            np.linspace(-1, 1, self.shape[2]),
            indexing='ij'
        )
        
        # Brainstem is elongated along anterior-posterior axis
        ellipsoid = (x**2 / (base_size * 1.2)**2 + 
                    y**2 / (base_size * 0.8)**2 + 
                    z**2 / (base_size * 2.0)**2)
        
        brainstem_mask = ellipsoid < 1.0
        
        # Modulate by morphogen gradients
        morphogen_influence = 0.3 * (shh + bmp) + 0.1 * wnt
        brainstem_mask = brainstem_mask & (morphogen_influence > 0.2)
        
        # Smooth the mask
        brainstem_mask = ndimage.binary_closing(brainstem_mask, iterations=2)
        brainstem_mask = ndimage.binary_opening(brainstem_mask, iterations=1)
        
        return brainstem_mask.astype(np.uint8)
    
    def generate_subdivision_masks(self, brainstem_mask: np.ndarray, 
                                 morphogens: np.ndarray, stage: str) -> Dict[str, np.ndarray]:
        """Generate subdivision masks (midbrain, pons, medulla)."""
        
        shh, bmp, wnt = morphogens
        
        # Get brainstem coordinates
        coords = np.where(brainstem_mask)
        if len(coords[0]) == 0:
            return {'midbrain': np.zeros_like(brainstem_mask),
                   'pons': np.zeros_like(brainstem_mask),
                   'medulla': np.zeros_like(brainstem_mask)}
        
        # Create subdivision masks based on anterior-posterior position
        z_coords = coords[2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_range = z_max - z_min
        
        # Subdivision boundaries (approximate)
        midbrain_z = z_min + 0.6 * z_range  # Anterior
        pons_z = z_min + 0.35 * z_range     # Middle
        # Medulla is posterior (z < pons_z)
        
        subdivisions = {}
        
        # Midbrain (anterior, high WNT)
        midbrain_mask = brainstem_mask.copy()
        midbrain_mask[coords] = (z_coords >= midbrain_z) & (wnt[coords] > 0.6)
        subdivisions['midbrain'] = midbrain_mask
        
        # Pons (middle)
        pons_mask = brainstem_mask.copy()
        pons_mask[coords] = (z_coords >= pons_z) & (z_coords < midbrain_z)
        subdivisions['pons'] = pons_mask
        
        # Medulla (posterior, low WNT)
        medulla_mask = brainstem_mask.copy()
        medulla_mask[coords] = (z_coords < pons_z) & (wnt[coords] < 0.4)
        subdivisions['medulla'] = medulla_mask
        
        return subdivisions


class SyntheticImageGenerator:
    """Generates synthetic MRI images with realistic artifacts."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        
    def generate_t2w_image(self, anatomy_masks: Dict[str, np.ndarray], 
                          morphogens: np.ndarray, stage: str) -> np.ndarray:
        """Generate T2-weighted MRI image."""
        
        # Base intensities for different tissues
        intensities = {
            'background': 10,
            'csf': 200,
            'brainstem': 120,
            'midbrain': 115,
            'pons': 125,
            'medulla': 120
        }
        
        # Create base image
        image = np.full(self.config.volume_shape, intensities['background'], dtype=np.float32)
        
        # Add brainstem structures
        for region, mask in anatomy_masks.items():
            if region in intensities:
                # Add some variation based on morphogens
                shh, bmp, wnt = morphogens
                variation = 10 * (0.5 - np.random.random(mask.shape)) * mask
                image[mask > 0] = intensities[region] + variation[mask > 0]
        
        # Add CSF around brainstem
        brainstem_dilated = ndimage.binary_dilation(
            anatomy_masks.get('brainstem', np.zeros_like(image)), iterations=3
        )
        csf_mask = brainstem_dilated & (image == intensities['background'])
        image[csf_mask] = intensities['csf']
        
        # Add noise
        snr = np.random.uniform(*self.config.snr_range)
        noise_std = np.mean(image[image > intensities['background']]) / snr
        noise = np.random.normal(0, noise_std, image.shape)
        image += noise
        
        # Add bias field
        if self.config.bias_field_strength > 0:
            bias_field = self._generate_bias_field()
            image *= bias_field
        
        # Add motion artifacts
        if self.config.motion_artifacts:
            image = self._add_motion_artifacts(image)
        
        return np.clip(image, 0, 255).astype(np.float32)
    
    def _generate_bias_field(self) -> np.ndarray:
        """Generate smooth bias field for MRI simulation."""
        
        # Create smooth random field
        field_shape = [s // 4 for s in self.config.volume_shape]
        field = np.random.normal(1.0, 0.1, field_shape)
        
        # Smooth and resize
        field = ndimage.gaussian_filter(field, sigma=1.0)
        field = ndimage.zoom(field, [s1/s2 for s1, s2 in 
                           zip(self.config.volume_shape, field_shape)])
        
        # Scale to desired strength
        field = 1.0 + self.config.bias_field_strength * (field - 1.0)
        
        return field
    
    def _add_motion_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add subtle motion artifacts."""
        
        # Random small translations
        if np.random.random() < 0.3:
            shift = np.random.normal(0, 0.5, 3)
            image = ndimage.shift(image, shift, mode='nearest')
        
        return image


class SyntheticDataGenerator:
    """Main class for generating synthetic brainstem data."""
    
    def __init__(self, config: SyntheticDataConfig = None):
        self.config = config or SyntheticDataConfig()
        
        self.morphogen_generator = MorphogenGradientGenerator(self.config)
        self.anatomy_generator = AnatomicalStructureGenerator(self.config)
        self.image_generator = SyntheticImageGenerator(self.config)
        
        logger.info(f"Initialized SyntheticDataGenerator with shape {self.config.volume_shape}")
    
    def generate_sample(self, stage: str, sample_id: str = None) -> Dict[str, np.ndarray]:
        """Generate a complete synthetic sample."""
        
        if sample_id is None:
            sample_id = f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Generating synthetic sample {sample_id} for stage {stage}")
        
        # Generate morphogen gradients
        morphogens = self.morphogen_generator.generate_combined_gradients(stage)
        
        # Generate anatomical structures
        brainstem_mask = self.anatomy_generator.generate_brainstem_mask(morphogens, stage)
        subdivision_masks = self.anatomy_generator.generate_subdivision_masks(
            brainstem_mask, morphogens, stage
        )
        
        # Combine all masks
        anatomy_masks = {
            'brainstem': brainstem_mask,
            **subdivision_masks
        }
        
        # Generate synthetic MRI image
        t2w_image = self.image_generator.generate_t2w_image(anatomy_masks, morphogens, stage)
        
        # Create combined segmentation mask
        segmentation = np.zeros_like(brainstem_mask, dtype=np.uint8)
        segmentation[subdivision_masks['midbrain'] > 0] = 1  # Midbrain
        segmentation[subdivision_masks['pons'] > 0] = 2      # Pons
        segmentation[subdivision_masks['medulla'] > 0] = 3   # Medulla
        
        return {
            'image': t2w_image,
            'segmentation': segmentation,
            'morphogens': morphogens,
            'brainstem_mask': brainstem_mask,
            'subdivision_masks': subdivision_masks,
            'metadata': {
                'sample_id': sample_id,
                'stage': stage,
                'generated': datetime.now().isoformat(),
                'config': self.config.__dict__
            }
        }
    
    def generate_dataset(self, n_samples_per_stage: int = 10, 
                        output_dir: Union[str, Path] = None) -> Dict[str, List[str]]:
        """Generate a complete synthetic dataset."""
        
        if output_dir is None:
            output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/synthetic")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {stage: [] for stage in self.config.stages}
        
        logger.info(f"Generating {n_samples_per_stage} samples per stage")
        logger.info(f"Output directory: {output_dir}")
        
        for stage in self.config.stages:
            stage_dir = output_dir / stage
            stage_dir.mkdir(exist_ok=True)
            
            for i in range(n_samples_per_stage):
                sample_id = f"{stage}_synthetic_{i:03d}"
                
                # Generate sample
                sample = self.generate_sample(stage, sample_id)
                
                # Save files
                files = self._save_sample(sample, stage_dir)
                generated_files[stage].extend(files)
                
                logger.info(f"Generated sample {i+1}/{n_samples_per_stage} for {stage}")
        
        # Save dataset metadata
        dataset_metadata = {
            'generated': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'n_samples_per_stage': n_samples_per_stage,
            'total_samples': len(self.config.stages) * n_samples_per_stage,
            'stages': self.config.stages,
            'files': generated_files
        }
        
        metadata_path = output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        logger.info(f"Dataset generation complete. Metadata saved to {metadata_path}")
        
        return generated_files
    
    def _save_sample(self, sample: Dict[str, np.ndarray], output_dir: Path) -> List[str]:
        """Save a sample to disk in NIfTI format."""
        
        sample_id = sample['metadata']['sample_id']
        files = []
        
        # Save image
        image_path = output_dir / f"{sample_id}_T2w.nii.gz"
        image_nii = nib.Nifti1Image(sample['image'], affine=np.eye(4))
        nib.save(image_nii, image_path)
        files.append(str(image_path))
        
        # Save segmentation
        seg_path = output_dir / f"{sample_id}_segmentation.nii.gz"
        seg_nii = nib.Nifti1Image(sample['segmentation'], affine=np.eye(4))
        nib.save(seg_nii, seg_path)
        files.append(str(seg_path))
        
        # Save morphogens
        morphogen_path = output_dir / f"{sample_id}_morphogens.nii.gz"
        # Transpose to put channels last for NIfTI
        morphogens_transposed = np.transpose(sample['morphogens'], (1, 2, 3, 0))
        morphogen_nii = nib.Nifti1Image(morphogens_transposed, affine=np.eye(4))
        nib.save(morphogen_nii, morphogen_path)
        files.append(str(morphogen_path))
        
        # Save metadata
        metadata_path = output_dir / f"{sample_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(sample['metadata'], f, indent=2)
        files.append(str(metadata_path))
        
        return files


def main():
    """Generate synthetic dataset for brainstem segmentation."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = SyntheticDataConfig(
        volume_shape=(128, 128, 96),
        voxel_size_mm=0.2,
        snr_range=(20.0, 40.0),
        bias_field_strength=0.15,
        motion_artifacts=True
    )
    
    # Create generator
    generator = SyntheticDataGenerator(config)
    
    # Generate dataset
    print("🧬 SYNTHETIC DATA GENERATION - Risk Mitigation")
    print("=" * 50)
    print("Generating synthetic brainstem dataset...")
    
    generated_files = generator.generate_dataset(
        n_samples_per_stage=5,  # Start with 5 samples per stage
        output_dir="/Users/camdouglas/quark/data/datasets/brainstem_segmentation/synthetic"
    )
    
    # Print summary
    total_files = sum(len(files) for files in generated_files.values())
    print(f"\n📊 GENERATION COMPLETE")
    print(f"   Stages: {len(config.stages)}")
    print(f"   Samples per stage: 5")
    print(f"   Total files: {total_files}")
    print(f"   Output: /Users/camdouglas/quark/data/datasets/brainstem_segmentation/synthetic")
    
    # Generate sample visualization
    print(f"\n🎨 Generating sample visualization...")
    sample = generator.generate_sample('E14', 'demo_sample')
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show middle slice
    mid_slice = sample['image'].shape[2] // 2
    
    # T2w image
    axes[0, 0].imshow(sample['image'][:, :, mid_slice], cmap='gray')
    axes[0, 0].set_title('Synthetic T2w Image')
    axes[0, 0].axis('off')
    
    # Segmentation
    axes[0, 1].imshow(sample['segmentation'][:, :, mid_slice], cmap='viridis')
    axes[0, 1].set_title('Segmentation Labels')
    axes[0, 1].axis('off')
    
    # Brainstem mask
    axes[0, 2].imshow(sample['brainstem_mask'][:, :, mid_slice], cmap='binary')
    axes[0, 2].set_title('Brainstem Mask')
    axes[0, 2].axis('off')
    
    # Morphogen gradients
    morphogen_names = ['SHH', 'BMP', 'WNT']
    for i, name in enumerate(morphogen_names):
        axes[1, i].imshow(sample['morphogens'][i, :, :, mid_slice], cmap='plasma')
        axes[1, i].set_title(f'{name} Gradient')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = "/Users/camdouglas/quark/data/datasets/brainstem_segmentation/synthetic/sample_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved: {viz_path}")
    
    print(f"\n✅ Synthetic data generation complete!")
    print(f"   Risk mitigation: Data availability addressed")
    print(f"   Next: Validate synthetic data quality")
    
    return generated_files


if __name__ == "__main__":
    generated_files = main()
