#!/usr/bin/env python3
"""Arachnoid Trabecular Structure System.

Generates and manages trabecular structure for arachnoid membrane including
fiber generation, connectivity analysis, and structural validation.

Integration: Trabecular structure component for arachnoid system
Rationale: Focused trabecular modeling separated from main arachnoid system
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import GridDimensions

logger = logging.getLogger(__name__)

class ArachnoidTrabecularGenerator:
    """Generator for arachnoid trabecular structure.
    
    Creates realistic trabecular fiber networks between dura and brain
    surfaces with proper connectivity and structural validation.
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize trabecular generator.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
        """
        self.dims = grid_dimensions
        logger.info("Initialized ArachnoidTrabecularGenerator")
    
    def generate_trabecular_structure(self, dura_surface: np.ndarray,
                                     brain_surface: np.ndarray,
                                     n_trabeculae: int = 50) -> np.ndarray:
        """Generate trabecular structure between dura and brain surfaces.
        
        Args:
            dura_surface: Dura mater surface mesh
            brain_surface: Brain surface mask
            n_trabeculae: Number of trabecular fibers to generate
            
        Returns:
            3D trabecular mesh array
        """
        logger.info(f"Generating {n_trabeculae} trabecular fibers")
        
        trabecular_mesh = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size))
        
        # Generate random trabecular fibers
        for i in range(n_trabeculae):
            fiber_points = self._generate_single_fiber(dura_surface, brain_surface)
            
            # Mark fiber points in mesh
            for point in fiber_points:
                x, y, z = point
                if (0 <= x < self.dims.x_size and 0 <= y < self.dims.y_size and 0 <= z < self.dims.z_size):
                    trabecular_mesh[x, y, z] = 1.0
        
        # Smooth trabecular structure
        trabecular_mesh = ndimage.gaussian_filter(trabecular_mesh, sigma=0.5)
        
        fiber_volume = np.sum(trabecular_mesh > 0.1) * (self.dims.resolution ** 3) / 1e9  # mm³
        logger.info(f"Generated trabecular structure: {fiber_volume:.4f} mm³ fiber volume")
        
        return trabecular_mesh
    
    def _generate_single_fiber(self, dura_surface: np.ndarray, 
                              brain_surface: np.ndarray) -> List[Tuple[int, int, int]]:
        """Generate a single trabecular fiber between surfaces."""
        # Find random start point on brain surface
        brain_points = np.where(brain_surface)
        if len(brain_points[0]) == 0:
            return []
        
        idx = np.random.randint(len(brain_points[0]))
        start_point = (brain_points[0][idx], brain_points[1][idx], brain_points[2][idx])
        
        # Find nearest dura surface point
        dura_points = np.where(dura_surface > 0.1 * np.max(dura_surface))
        if len(dura_points[0]) == 0:
            return []
        
        distances = np.sqrt(
            (dura_points[0] - start_point[0])**2 + 
            (dura_points[1] - start_point[1])**2 + 
            (dura_points[2] - start_point[2])**2
        )
        nearest_idx = np.argmin(distances)
        end_point = (dura_points[0][nearest_idx], dura_points[1][nearest_idx], 
                   dura_points[2][nearest_idx])
        
        # Draw fiber between points
        return self._draw_fiber_line(start_point, end_point)
    
    def _draw_fiber_line(self, start: Tuple[int, int, int], 
                        end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Draw a fiber line between two points using 3D line algorithm."""
        points = []
        
        x0, y0, z0 = start
        x1, y1, z1 = end
        
        # Calculate steps
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        steps = max(dx, dy, dz)
        if steps == 0:
            return [start]
        
        # Interpolate points along the line
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            z = int(z0 + t * (z1 - z0))
            points.append((x, y, z))
        
        return points
    
    def validate_trabecular_connectivity(self, trabecular_mesh: np.ndarray) -> Dict[str, float]:
        """Validate connectivity and structure of trabecular network.
        
        Args:
            trabecular_mesh: Generated trabecular mesh
            
        Returns:
            Dictionary with connectivity metrics
        """
        # Label connected components
        labeled_structure, n_components = ndimage.label(trabecular_mesh > 0.1)
        
        if n_components == 0:
            return {"connectivity_score": 0.0, "component_count": 0, "largest_component_fraction": 0.0}
        
        # Calculate size of each component
        component_sizes = []
        for i in range(1, n_components + 1):
            component_size = np.sum(labeled_structure == i)
            component_sizes.append(component_size)
        
        # Connectivity metrics
        largest_component_size = max(component_sizes)
        total_fiber_voxels = np.sum(trabecular_mesh > 0.1)
        largest_component_fraction = largest_component_size / total_fiber_voxels if total_fiber_voxels > 0 else 0.0
        
        # Good connectivity means most fibers are in one large connected network
        connectivity_score = largest_component_fraction
        
        metrics = {
            "connectivity_score": float(connectivity_score),
            "component_count": int(n_components),
            "largest_component_fraction": float(largest_component_fraction),
            "total_fiber_volume_mm3": float(total_fiber_voxels * (self.dims.resolution ** 3) / 1e9)
        }
        
        logger.info(f"Trabecular connectivity: {connectivity_score:.3f}, "
                   f"{n_components} components")
        
        return metrics
