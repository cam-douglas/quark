#!/usr/bin/env python3
"""Arachnoid Vascular Integration System.

Manages vascular integration points and pathways through arachnoid membrane
including blood vessel pathway establishment and CSF connectivity validation.

Integration: Vascular integration component for arachnoid system
Rationale: Focused vascular modeling separated from main arachnoid system
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import GridDimensions

logger = logging.getLogger(__name__)

class ArachnoidVascularIntegrator:
    """Vascular integration manager for arachnoid membrane.
    
    Handles blood vessel pathway creation through the arachnoid layer
    and validates connectivity with CSF circulation systems.
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize arachnoid vascular integrator.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
        """
        self.dims = grid_dimensions
        logger.info("Initialized ArachnoidVascularIntegrator")
    
    def create_vascular_integration_points(self) -> List[Tuple[float, float, float]]:
        """Create vascular integration points for blood vessel pathways."""
        center_x, center_y, center_z = self.dims.x_size/2, self.dims.y_size/2, self.dims.z_size/2
        
        # Major vascular entry points for embryonic brain
        vascular_points = [
            (center_x - 120, center_y + 80, center_z + 30),   # Left middle cerebral
            (center_x + 120, center_y + 80, center_z + 30),   # Right middle cerebral
            (center_x, center_y + 180, center_z + 60),        # Anterior cerebral
            (center_x - 80, center_y - 100, center_z + 20),   # Left posterior
            (center_x + 80, center_y - 100, center_z + 20),   # Right posterior
            (center_x, center_y - 130, center_z + 40),        # Basilar entry
        ]
        
        logger.info(f"Created {len(vascular_points)} vascular integration points")
        return vascular_points
    
    def integrate_vascular_pathways(self, subarachnoid_space: np.ndarray,
                                   vascular_points: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Integrate vascular pathways through arachnoid layer.
        
        Args:
            subarachnoid_space: Boolean mask of subarachnoid space
            vascular_points: List of vascular integration points
            
        Returns:
            Dictionary with vascular integration analysis
        """
        logger.info("Integrating vascular pathways")
        
        vascular_pathways = {}
        
        # Create pathways for each vascular integration point
        for i, vascular_point in enumerate(vascular_points):
            pathway_id = f"vascular_pathway_{i+1}"
            
            # Convert to voxel coordinates
            vascular_vox = (
                int(vascular_point[0] / self.dims.resolution),
                int(vascular_point[1] / self.dims.resolution),
                int(vascular_point[2] / self.dims.resolution)
            )
            
            # Ensure within bounds
            vascular_vox = (
                np.clip(vascular_vox[0], 0, self.dims.x_size - 1),
                np.clip(vascular_vox[1], 0, self.dims.y_size - 1),
                np.clip(vascular_vox[2], 0, self.dims.z_size - 1)
            )
            
            # Create pathway through subarachnoid space
            pathway_mask = self._create_vascular_pathway(vascular_vox, subarachnoid_space)
            
            # Calculate pathway metrics
            pathway_volume = np.sum(pathway_mask) * (self.dims.resolution ** 3) / 1e9  # mm³
            pathway_length = self._calculate_pathway_length(pathway_mask)
            
            vascular_pathways[pathway_id] = {
                "entry_point_um": vascular_point,
                "volume_mm3": pathway_volume,
                "length_um": pathway_length,
                "connectivity_to_csf": self._validate_csf_connectivity(pathway_mask, subarachnoid_space)
            }
        
        logger.info(f"Integrated {len(vascular_pathways)} vascular pathways")
        
        return vascular_pathways
    
    def _create_vascular_pathway(self, entry_point: Tuple[int, int, int],
                                subarachnoid_space: np.ndarray) -> np.ndarray:
        """Create vascular pathway from entry point through subarachnoid space."""
        pathway_mask = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size), dtype=bool)
        
        # Create cylindrical pathway with radius ~10 µm
        pathway_radius_voxels = int(10.0 / self.dims.resolution)
        
        # Pathway extends from entry point toward brain surface
        x_entry, y_entry, z_entry = entry_point
        
        # Find target point in subarachnoid space
        brain_boundary = ndimage.binary_erosion(~subarachnoid_space)
        brain_points = np.where(brain_boundary)
        
        if len(brain_points[0]) > 0:
            distances = np.sqrt(
                (brain_points[0] - x_entry)**2 + 
                (brain_points[1] - y_entry)**2 + 
                (brain_points[2] - z_entry)**2
            )
            nearest_idx = np.argmin(distances)
            target_point = (brain_points[0][nearest_idx], brain_points[1][nearest_idx], 
                          brain_points[2][nearest_idx])
            
            # Create pathway from entry to target
            pathway_points = self._interpolate_pathway(entry_point, target_point)
            
            # Mark pathway voxels with cylindrical cross-section
            for point in pathway_points:
                x, y, z = point
                
                for dx in range(-pathway_radius_voxels, pathway_radius_voxels + 1):
                    for dy in range(-pathway_radius_voxels, pathway_radius_voxels + 1):
                        px, py = x + dx, y + dy
                        
                        if (0 <= px < self.dims.x_size and 0 <= py < self.dims.y_size and 0 <= z < self.dims.z_size):
                            if dx**2 + dy**2 <= pathway_radius_voxels**2:
                                pathway_mask[px, py, z] = True
        
        return pathway_mask
    
    def _interpolate_pathway(self, start: Tuple[int, int, int], 
                           end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Interpolate pathway points between start and end."""
        x0, y0, z0 = start
        x1, y1, z1 = end
        
        # Calculate number of steps
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        steps = max(1, int(distance))
        
        # Interpolate points
        points = []
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            z = int(z0 + t * (z1 - z0))
            points.append((x, y, z))
        
        return points
    
    def _calculate_pathway_length(self, pathway_mask: np.ndarray) -> float:
        """Calculate length of vascular pathway in micrometers."""
        pathway_points = np.where(pathway_mask)
        
        if len(pathway_points[0]) == 0:
            return 0.0
        
        # Approximate length using point count and resolution
        n_points = len(pathway_points[0])
        avg_length = n_points * self.dims.resolution
        
        return avg_length
    
    def _validate_csf_connectivity(self, pathway_mask: np.ndarray,
                                  subarachnoid_space: np.ndarray) -> bool:
        """Validate that vascular pathway connects to CSF space."""
        # Check if pathway intersects with subarachnoid space
        intersection = pathway_mask & subarachnoid_space
        
        return np.any(intersection)
    
    def validate_trabecular_connectivity(self, trabecular_mesh: np.ndarray) -> Dict[str, float]:
        """Validate connectivity and structure of trabecular network."""
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
