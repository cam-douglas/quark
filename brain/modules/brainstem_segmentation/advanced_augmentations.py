#!/usr/bin/env python3
"""
Advanced Data Augmentation Pipeline for Brainstem Segmentation
Implements rotations, elastic deformations, and intensity augmentations
"""

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from typing import Tuple, Optional, Dict, Any
import random


class AdvancedAugmentation3D:
    """Advanced 3D augmentation pipeline for medical imaging."""
    
    def __init__(self,
                 rotation_range: Tuple[float, float] = (-15, 15),
                 elastic_alpha: float = 100.0,
                 elastic_sigma: float = 10.0,
                 elastic_prob: float = 0.5,
                 intensity_shift_range: float = 0.1,
                 intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
                 gamma_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.01,
                 blur_sigma_range: Tuple[float, float] = (0.0, 1.0),
                 flip_prob: float = 0.5,
                 cutmix_prob: float = 0.3):
        """
        Args:
            rotation_range: Range of rotation angles in degrees
            elastic_alpha: Elastic deformation alpha parameter
            elastic_sigma: Elastic deformation sigma parameter
            elastic_prob: Probability of applying elastic deformation
            intensity_shift_range: Range for intensity shifting
            intensity_scale_range: Range for intensity scaling
            gamma_range: Range for gamma correction
            noise_std: Standard deviation for Gaussian noise
            blur_sigma_range: Range for Gaussian blur sigma
            flip_prob: Probability of random flips
            cutmix_prob: Probability of CutMix augmentation
        """
        self.rotation_range = rotation_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.blur_sigma_range = blur_sigma_range
        self.flip_prob = flip_prob
        self.cutmix_prob = cutmix_prob
    
    def random_rotation_3d(self, 
                          image: np.ndarray, 
                          label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random 3D rotation to image and label."""
        
        # Random rotation angles for each axis
        angles = [random.uniform(*self.rotation_range) for _ in range(3)]
        
        # Choose random rotation axes
        axes = [(0, 1), (0, 2), (1, 2)]
        axis = random.choice(axes)
        angle = random.uniform(*self.rotation_range)
        
        # Apply rotation
        rotated_image = rotate(image, angle, axes=axis, reshape=False, 
                               order=1, mode='constant', cval=0)
        rotated_label = rotate(label, angle, axes=axis, reshape=False, 
                               order=0, mode='constant', cval=0)
        
        return rotated_image, rotated_label
    
    def elastic_deformation_3d(self, 
                              image: np.ndarray, 
                              label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation to 3D volume."""
        
        if random.random() > self.elastic_prob:
            return image, label
        
        # Handle both 3D and 4D (channels first) tensors
        if len(image.shape) == 4:
            # Process each channel separately
            deformed_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                deformed_image[c], _ = self._elastic_deform_3d(image[c], label)
            _, deformed_label = self._elastic_deform_3d(image[0], label)
            return deformed_image, deformed_label
        else:
            return self._elastic_deform_3d(image, label)
    
    def _elastic_deform_3d(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Internal elastic deformation for 3D volumes."""
        shape = image.shape
        
        # Generate random displacement fields
        dx = gaussian_filter(np.random.randn(*shape) * self.elastic_alpha, 
                            self.elastic_sigma, mode='constant', cval=0)
        dy = gaussian_filter(np.random.randn(*shape) * self.elastic_alpha, 
                            self.elastic_sigma, mode='constant', cval=0)
        dz = gaussian_filter(np.random.randn(*shape) * self.elastic_alpha, 
                            self.elastic_sigma, mode='constant', cval=0)
        
        # Create coordinate arrays
        x, y, z = np.meshgrid(np.arange(shape[0]), 
                             np.arange(shape[1]), 
                             np.arange(shape[2]), indexing='ij')
        
        # Apply displacement
        indices = (x + dx, y + dy, z + dz)
        
        # Map coordinates with different interpolation orders
        deformed_image = map_coordinates(image, indices, order=1, 
                                        mode='constant', cval=0)
        deformed_label = map_coordinates(label, indices, order=0, 
                                        mode='constant', cval=0)
        
        return deformed_image, deformed_label
    
    def intensity_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply various intensity augmentations."""
        
        # Intensity shift
        if random.random() < 0.5:
            shift = random.uniform(-self.intensity_shift_range, 
                                 self.intensity_shift_range)
            image = image + shift
        
        # Intensity scale
        if random.random() < 0.5:
            scale = random.uniform(*self.intensity_scale_range)
            image = image * scale
        
        # Gamma correction
        if random.random() < 0.5:
            gamma = random.uniform(*self.gamma_range)
            # Normalize to [0, 1] for gamma correction
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
                image = np.power(image, gamma)
                image = image * (img_max - img_min) + img_min
        
        # Gaussian noise
        if random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
        
        # Gaussian blur
        if random.random() < 0.3:
            sigma = random.uniform(*self.blur_sigma_range)
            if sigma > 0:
                image = gaussian_filter(image, sigma=sigma)
        
        return image
    
    def random_flip_3d(self, 
                      image: np.ndarray, 
                      label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flips along different axes."""
        
        # Random flip along each axis
        for axis in range(3):
            if random.random() < self.flip_prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        
        return image, label
    
    def cutmix_3d(self, 
                 image1: np.ndarray, label1: np.ndarray,
                 image2: np.ndarray, label2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply CutMix augmentation to 3D volumes."""
        
        if random.random() > self.cutmix_prob:
            return image1, label1
        
        # Get random box coordinates
        shape = image1.shape
        lam = random.uniform(0.3, 0.7)  # Mix ratio
        
        # Calculate box size based on lambda
        box_size = [int(s * np.cbrt(lam)) for s in shape]
        
        # Random box position
        x1 = random.randint(0, shape[0] - box_size[0])
        y1 = random.randint(0, shape[1] - box_size[1])
        z1 = random.randint(0, shape[2] - box_size[2])
        
        x2 = x1 + box_size[0]
        y2 = y1 + box_size[1]
        z2 = z1 + box_size[2]
        
        # Mix images and labels
        mixed_image = image1.copy()
        mixed_label = label1.copy()
        
        mixed_image[x1:x2, y1:y2, z1:z2] = image2[x1:x2, y1:y2, z1:z2]
        mixed_label[x1:x2, y1:y2, z1:z2] = label2[x1:x2, y1:y2, z1:z2]
        
        return mixed_image, mixed_label
    
    def __call__(self, 
                image: np.ndarray, 
                label: np.ndarray,
                mix_sample: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full augmentation pipeline.
        
        Args:
            image: Input image volume
            label: Corresponding label volume
            mix_sample: Optional second sample for CutMix
            
        Returns:
            Augmented image and label
        """
        # Apply geometric augmentations
        if random.random() < 0.7:
            image, label = self.random_rotation_3d(image, label)
        
        if random.random() < self.elastic_prob:
            image, label = self.elastic_deformation_3d(image, label)
        
        if random.random() < 0.5:
            image, label = self.random_flip_3d(image, label)
        
        # Apply CutMix if mix_sample provided
        if mix_sample is not None:
            image, label = self.cutmix_3d(image, label, mix_sample[0], mix_sample[1])
        
        # Apply intensity augmentations (only to image)
        image = self.intensity_augmentation(image)
        
        return image, label


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper with advanced augmentations."""
    
    def __init__(self, 
                 base_dataset,
                 augmentation: Optional[AdvancedAugmentation3D] = None,
                 training: bool = True):
        """
        Args:
            base_dataset: Base dataset to wrap
            augmentation: Augmentation pipeline
            training: Whether in training mode
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation if training else None
        self.training = training
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample
        image, label = self.base_dataset[idx]
        
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        # Apply augmentations if in training mode
        if self.augmentation is not None:
            # Optionally get another sample for CutMix
            mix_sample = None
            if random.random() < 0.3:  # 30% chance of CutMix
                mix_idx = random.randint(0, len(self.base_dataset) - 1)
                mix_image, mix_label = self.base_dataset[mix_idx]
                if isinstance(mix_image, torch.Tensor):
                    mix_image = mix_image.numpy()
                if isinstance(mix_label, torch.Tensor):
                    mix_label = mix_label.numpy()
                mix_sample = (mix_image, mix_label)
            
            # Apply augmentations
            image, label = self.augmentation(image, label, mix_sample)
        
        # Convert back to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        return image, label


def test_augmentations():
    """Test the advanced augmentation pipeline."""
    
    print("🎨 TESTING ADVANCED AUGMENTATIONS")
    print("=" * 50)
    
    # Create augmentation pipeline
    augmentation = AdvancedAugmentation3D(
        rotation_range=(-20, 20),
        elastic_alpha=150.0,
        elastic_sigma=12.0,
        elastic_prob=0.5,
        intensity_shift_range=0.15,
        intensity_scale_range=(0.85, 1.15),
        gamma_range=(0.7, 1.3),
        noise_std=0.02,
        blur_sigma_range=(0.0, 1.5),
        flip_prob=0.5,
        cutmix_prob=0.3
    )
    
    # Create synthetic test data
    image = np.random.randn(64, 64, 64).astype(np.float32)
    label = np.random.randint(0, 6, (64, 64, 64)).astype(np.int32)
    
    print("✅ Augmentation pipeline created")
    print(f"   Original image shape: {image.shape}")
    print(f"   Original label shape: {label.shape}")
    
    # Test individual augmentations
    print("\n📊 Testing individual augmentations:")
    
    # Rotation
    rot_image, rot_label = augmentation.random_rotation_3d(image, label)
    print(f"   ✅ Rotation: {rot_image.shape}, unique labels: {np.unique(rot_label).tolist()}")
    
    # Elastic deformation
    elastic_image, elastic_label = augmentation.elastic_deformation_3d(image, label)
    print(f"   ✅ Elastic: {elastic_image.shape}, unique labels: {np.unique(elastic_label).tolist()}")
    
    # Intensity
    intensity_image = augmentation.intensity_augmentation(image)
    print(f"   ✅ Intensity: {intensity_image.shape}, value range: [{intensity_image.min():.2f}, {intensity_image.max():.2f}]")
    
    # Flip
    flip_image, flip_label = augmentation.random_flip_3d(image, label)
    print(f"   ✅ Flip: {flip_image.shape}, unique labels: {np.unique(flip_label).tolist()}")
    
    # Full pipeline
    print("\n🔄 Testing full augmentation pipeline:")
    aug_image, aug_label = augmentation(image, label)
    print(f"   ✅ Full pipeline: {aug_image.shape}")
    print(f"   Value range: [{aug_image.min():.2f}, {aug_image.max():.2f}]")
    print(f"   Unique labels: {np.unique(aug_label).tolist()}")
    
    # Test with CutMix
    mix_image = np.random.randn(64, 64, 64).astype(np.float32)
    mix_label = np.random.randint(0, 6, (64, 64, 64)).astype(np.int32)
    cutmix_image, cutmix_label = augmentation(image, label, (mix_image, mix_label))
    print(f"   ✅ With CutMix: {cutmix_image.shape}")
    
    print("\n✅ Advanced augmentation pipeline ready!")
    return augmentation


if __name__ == "__main__":
    test_augmentations()
