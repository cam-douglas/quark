#!/usr/bin/env python3
"""Atlas Coordinate System Mapper.

Implements coordinate system mapping and registration between morphogen
solver grid coordinates and Allen Brain Atlas coordinate systems with
alignment tools and transformation validation.

Integration: Coordinate mapping component for atlas validation system
Rationale: Focused coordinate transformation logic separated from main validation
"""

from typing import Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import ndimage
import logging

from .spatial_grid import GridDimensions
from .atlas_validation_types import AtlasReference, CoordinateTransform, CoordinateSystem

logger = logging.getLogger(__name__)

class AtlasCoordinateMapper:
    """Coordinate mapper for atlas-morphogen grid alignment.
    
    Handles coordinate system transformations, registration algorithms,
    and alignment validation between morphogen solver grid coordinates
    and Allen Brain Atlas coordinate systems.
    """
    
    def __init__(self, morphogen_grid_dims: GridDimensions):
        """Initialize atlas coordinate mapper.
        
        Args:
            morphogen_grid_dims: Morphogen solver grid dimensions
        """
        self.grid_dims = morphogen_grid_dims
        
        logger.info("Initialized AtlasCoordinateMapper")
        logger.info(f"Morphogen grid: {morphogen_grid_dims.x_size}x{morphogen_grid_dims.y_size}x{morphogen_grid_dims.z_size}")
        logger.info(f"Grid resolution: {morphogen_grid_dims.resolution} µm")
    
    def create_coordinate_transform(self, atlas_ref: AtlasReference) -> CoordinateTransform:
        """Create coordinate transformation from morphogen grid to atlas coordinates.
        
        Args:
            atlas_ref: Allen atlas reference data
            
        Returns:
            Coordinate transformation object
        """
        logger.info("Creating coordinate transformation")
        
        # Calculate scaling factors
        atlas_dims = atlas_ref.dimensions
        morphogen_dims = (self.grid_dims.x_size, self.grid_dims.y_size, self.grid_dims.z_size)
        
        scaling_factors = (
            atlas_dims[0] / morphogen_dims[0],
            atlas_dims[1] / morphogen_dims[1],
            atlas_dims[2] / morphogen_dims[2]
        )
        
        # Calculate resolution scaling
        resolution_scaling = atlas_ref.resolution_um / self.grid_dims.resolution
        
        # Adjust scaling for resolution differences
        final_scaling = tuple(s * resolution_scaling for s in scaling_factors)
        
        # Create transformation matrix (4x4 homogeneous coordinates)
        transform_matrix = np.eye(4)
        transform_matrix[0, 0] = final_scaling[0]
        transform_matrix[1, 1] = final_scaling[1]
        transform_matrix[2, 2] = final_scaling[2]
        
        # Translation to center alignment (simplified)
        translation_offset = (
            (atlas_dims[0] - morphogen_dims[0] * final_scaling[0]) / 2,
            (atlas_dims[1] - morphogen_dims[1] * final_scaling[1]) / 2,
            (atlas_dims[2] - morphogen_dims[2] * final_scaling[2]) / 2
        )
        
        transform_matrix[0, 3] = translation_offset[0]
        transform_matrix[1, 3] = translation_offset[1]
        transform_matrix[2, 3] = translation_offset[2]
        
        # No rotation for initial alignment
        rotation_angles = (0.0, 0.0, 0.0)
        
        coordinate_transform = CoordinateTransform(
            source_system=CoordinateSystem.MORPHOGEN_GRID,
            target_system=atlas_ref.coordinate_system,
            transformation_matrix=transform_matrix,
            scaling_factors=final_scaling,
            translation_offset=translation_offset,
            rotation_angles=rotation_angles
        )
        
        logger.info(f"Created coordinate transform: scaling={final_scaling}, translation={translation_offset}")
        
        return coordinate_transform
    
    def transform_morphogen_to_atlas(self, morphogen_data: np.ndarray,
                                   coordinate_transform: CoordinateTransform) -> np.ndarray:
        """Transform morphogen data to atlas coordinate system.
        
        Args:
            morphogen_data: Morphogen concentration data
            coordinate_transform: Coordinate transformation
            
        Returns:
            Transformed data in atlas coordinates
        """
        # Apply scaling transformation
        scaling = coordinate_transform.scaling_factors
        
        # Use scipy ndimage for transformation
        transformed_data = ndimage.zoom(morphogen_data, scaling, order=1)
        
        # Apply translation (padding/cropping)
        translation = coordinate_transform.translation_offset
        
        # Pad or crop to match atlas dimensions
        transformed_data = self._apply_translation_padding(transformed_data, translation)
        
        return transformed_data
    
    def _apply_translation_padding(self, data: np.ndarray, 
                                  translation: Tuple[float, float, float]) -> np.ndarray:
        """Apply translation through padding or cropping."""
        # Convert translation to integer offsets
        offset_x = int(translation[0])
        offset_y = int(translation[1])
        offset_z = int(translation[2])
        
        # Apply padding/cropping for each dimension
        if offset_x > 0:
            # Pad at beginning
            data = np.pad(data, ((offset_x, 0), (0, 0), (0, 0)), mode='constant')
        elif offset_x < 0:
            # Crop at beginning
            data = data[-offset_x:, :, :]
        
        if offset_y > 0:
            data = np.pad(data, ((0, 0), (offset_y, 0), (0, 0)), mode='constant')
        elif offset_y < 0:
            data = data[:, -offset_y:, :]
        
        if offset_z > 0:
            data = np.pad(data, ((0, 0), (0, 0), (offset_z, 0)), mode='constant')
        elif offset_z < 0:
            data = data[:, :, -offset_z:]
        
        return data
    
    def register_morphogen_to_atlas(self, morphogen_segmentation: np.ndarray,
                                   atlas_ref: AtlasReference,
                                   registration_method: str = "rigid") -> CoordinateTransform:
        """Register morphogen segmentation to atlas using image registration.
        
        Args:
            morphogen_segmentation: Morphogen-based segmentation
            atlas_ref: Atlas reference data
            registration_method: Registration method ("rigid", "affine", "deformable")
            
        Returns:
            Optimized coordinate transformation
        """
        logger.info(f"Registering morphogen data to atlas using {registration_method} registration")
        
        # Initial transformation
        initial_transform = self.create_coordinate_transform(atlas_ref)
        
        # Transform morphogen data to atlas space
        transformed_morphogen = self.transform_morphogen_to_atlas(
            morphogen_segmentation, initial_transform)
        
        # Ensure same dimensions as atlas
        atlas_labels = atlas_ref.region_labels
        if transformed_morphogen.shape != atlas_labels.shape:
            # Resize to match atlas
            transformed_morphogen = self._resize_to_match_atlas(
                transformed_morphogen, atlas_labels.shape)
        
        # Perform registration optimization (simplified)
        optimized_transform = self._optimize_registration(
            transformed_morphogen, atlas_labels, initial_transform, registration_method)
        
        return optimized_transform
    
    def _resize_to_match_atlas(self, data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize data to match atlas dimensions."""
        # Calculate zoom factors
        zoom_factors = tuple(target_dim / current_dim 
                           for target_dim, current_dim in zip(target_shape, data.shape))
        
        # Apply zoom
        resized_data = ndimage.zoom(data, zoom_factors, order=1)
        
        return resized_data
    
    def _optimize_registration(self, morphogen_data: np.ndarray, atlas_labels: np.ndarray,
                              initial_transform: CoordinateTransform,
                              method: str) -> CoordinateTransform:
        """Optimize registration between morphogen data and atlas."""
        # Simplified registration optimization
        # In practice, would use sophisticated registration algorithms
        
        optimized_transform = initial_transform
        
        if method == "rigid":
            # Try small rotations and translations to improve alignment
            best_score = self._calculate_alignment_score(morphogen_data, atlas_labels)
            
            # Test small rotations (±5 degrees)
            for angle_x in [-0.087, 0, 0.087]:  # ±5 degrees in radians
                for angle_y in [-0.087, 0, 0.087]:
                    for angle_z in [-0.087, 0, 0.087]:
                        # Apply rotation
                        rotated_data = self._apply_rotation(morphogen_data, (angle_x, angle_y, angle_z))
                        
                        # Calculate alignment score
                        score = self._calculate_alignment_score(rotated_data, atlas_labels)
                        
                        if score > best_score:
                            best_score = score
                            # Update transform with best rotation
                            optimized_transform.rotation_angles = (angle_x, angle_y, angle_z)
        
        logger.info(f"Registration optimization completed, alignment score: {best_score:.3f}")
        
        return optimized_transform
    
    def _apply_rotation(self, data: np.ndarray, angles: Tuple[float, float, float]) -> np.ndarray:
        """Apply rotation to 3D data."""
        # Create rotation matrix
        rotation = Rotation.from_euler('xyz', angles)
        rotation_matrix = rotation.as_matrix()
        
        # Apply rotation using scipy
        # Note: This is a simplified implementation
        rotated_data = ndimage.rotate(data, np.degrees(angles[0]), axes=(1, 2), reshape=False, order=1)
        rotated_data = ndimage.rotate(rotated_data, np.degrees(angles[1]), axes=(0, 2), reshape=False, order=1)
        rotated_data = ndimage.rotate(rotated_data, np.degrees(angles[2]), axes=(0, 1), reshape=False, order=1)
        
        return rotated_data
    
    def _calculate_alignment_score(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate alignment score between two 3D datasets."""
        # Normalize both datasets
        data1_norm = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1)) if np.max(data1) > np.min(data1) else data1
        data2_norm = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2)) if np.max(data2) > np.min(data2) else data2
        
        # Calculate normalized cross-correlation
        correlation = np.corrcoef(data1_norm.flatten(), data2_norm.flatten())[0, 1]
        
        # Return correlation (higher is better alignment)
        return correlation if not np.isnan(correlation) else 0.0
