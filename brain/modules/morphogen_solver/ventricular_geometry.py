#!/usr/bin/env python3
"""Ventricular Cavity Geometry Generator.

Generates 3D ventricular cavity geometries including ellipsoidal, tubular,
and irregular cavity shapes for embryonic brain development modeling.

Integration: Geometry generation component for ventricular topology
Rationale: Focused geometry algorithms separated from main topology logic
"""

from typing import Tuple
import numpy as np
import logging

from .spatial_grid import GridDimensions
from .ventricular_types import VentricularRegion

logger = logging.getLogger(__name__)

class VentricularGeometryGenerator:
    """Generator for ventricular cavity geometries.
    
    Creates 3D voxel masks for different ventricular cavity shapes
    including ellipsoidal (lateral/third ventricles), tubular (aqueduct),
    and irregular (fourth ventricle) geometries.
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize geometry generator.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
        """
        self.dims = grid_dimensions
        logger.info("Initialized VentricularGeometryGenerator")
    
    def generate_cavity_voxels(self, region: VentricularRegion) -> np.ndarray:
        """Generate voxels for a specific ventricular cavity.
        
        Args:
            region: Ventricular region definition
            
        Returns:
            3D boolean array for this cavity
        """
        # Convert position and dimensions to voxel coordinates
        center_vox = (
            int(region.center_position[0] / self.dims.resolution),
            int(region.center_position[1] / self.dims.resolution),
            int(region.center_position[2] / self.dims.resolution)
        )
        
        size_vox = (
            int(region.dimensions[0] / self.dims.resolution),
            int(region.dimensions[1] / self.dims.resolution),
            int(region.dimensions[2] / self.dims.resolution)
        )
        
        # Generate cavity based on shape type
        if region.shape_type == "ellipsoid":
            return self._generate_ellipsoid_cavity(center_vox, size_vox)
        elif region.shape_type == "tube":
            return self._generate_tube_cavity(center_vox, size_vox)
        elif region.shape_type == "irregular":
            return self._generate_irregular_cavity(center_vox, size_vox)
        else:
            logger.warning(f"Unknown shape type: {region.shape_type}")
            return np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size), dtype=bool)
    
    def _generate_ellipsoid_cavity(self, center: Tuple[int, int, int],
                                  size: Tuple[int, int, int]) -> np.ndarray:
        """Generate ellipsoidal cavity (lateral and third ventricles)."""
        cavity_mask = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size), dtype=bool)
        
        cx, cy, cz = center
        rx, ry, rz = size[0]//2, size[1]//2, size[2]//2
        
        # Create meshgrid for ellipsoid equation
        x_range = slice(max(0, cx-rx), min(self.dims.x_size, cx+rx))
        y_range = slice(max(0, cy-ry), min(self.dims.y_size, cy+ry))
        z_range = slice(max(0, cz-rz), min(self.dims.z_size, cz+rz))
        
        x, y, z = np.meshgrid(
            np.arange(x_range.start, x_range.stop),
            np.arange(y_range.start, y_range.stop),
            np.arange(z_range.start, z_range.stop),
            indexing='ij'
        )
        
        # Ellipsoid equation: (x-cx)²/rx² + (y-cy)²/ry² + (z-cz)²/rz² ≤ 1
        ellipsoid = ((x - cx)**2 / rx**2 + 
                    (y - cy)**2 / ry**2 + 
                    (z - cz)**2 / rz**2) <= 1
        
        cavity_mask[x_range, y_range, z_range] = ellipsoid
        
        return cavity_mask
    
    def _generate_tube_cavity(self, center: Tuple[int, int, int],
                             size: Tuple[int, int, int]) -> np.ndarray:
        """Generate tubular cavity (cerebral aqueduct)."""
        cavity_mask = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size), dtype=bool)
        
        cx, cy, cz = center
        radius = min(size[0], size[1]) // 2
        length = size[2]
        
        # Create cylindrical tube along z-axis
        z_range = slice(max(0, cz-length//2), min(self.dims.z_size, cz+length//2))
        
        for z in range(z_range.start, z_range.stop):
            # Circular cross-section
            y, x = np.ogrid[0:self.dims.y_size, 0:self.dims.x_size]
            circle = (x - cx)**2 + (y - cy)**2 <= radius**2
            cavity_mask[:, :, z] |= circle.T
        
        return cavity_mask
    
    def _generate_irregular_cavity(self, center: Tuple[int, int, int],
                                  size: Tuple[int, int, int]) -> np.ndarray:
        """Generate irregular cavity (fourth ventricle)."""
        # Start with ellipsoid and modify for diamond shape
        cavity_mask = self._generate_ellipsoid_cavity(center, size)
        
        # Add diamond-like modifications for fourth ventricle
        cx, cy, cz = center
        rx, ry, rz = size[0]//2, size[1]//2, size[2]//2
        
        # Create diamond cross-section in x-y plane
        x_range = slice(max(0, cx-rx), min(self.dims.x_size, cx+rx))
        y_range = slice(max(0, cy-ry), min(self.dims.y_size, cy+ry))
        z_range = slice(max(0, cz-rz), min(self.dims.z_size, cz+rz))
        
        for z in range(z_range.start, z_range.stop):
            y, x = np.ogrid[y_range.start:y_range.stop, x_range.start:x_range.stop]
            # Diamond shape: |x-cx|/rx + |y-cy|/ry ≤ 1
            diamond = (np.abs(x - cx) / rx + np.abs(y - cy) / ry) <= 1
            cavity_mask[x_range, y_range, z] = diamond.T
        
        return cavity_mask
