#!/usr/bin/env python3
"""Dura Mater Surface Mesh Generator.

Generates 3D surface meshes for dura mater layer including outer surface
generation, attachment constraint application, and mesh smoothing operations.

Integration: Mesh generation component for dura mater system
Rationale: Focused mesh generation algorithms separated from main dura system
"""

from typing import List, Optional
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import GridDimensions
from .meninges_types import AttachmentPoint, MeningesLayer

logger = logging.getLogger(__name__)

class DuraMeshGenerator:
    """Surface mesh generator for dura mater layer.
    
    Generates 3D surface meshes for the dura mater including outer surface
    computation, attachment point constraint application, and smoothing.
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize dura mesh generator.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
        """
        self.dims = grid_dimensions
        logger.info("Initialized DuraMeshGenerator")
    
    def generate_surface_mesh(self, cavity_mask: np.ndarray, 
                             thickness_um: float,
                             attachment_points: List[AttachmentPoint]) -> np.ndarray:
        """Generate complete dura mater surface mesh.
        
        Args:
            cavity_mask: Boolean mask of ventricular cavities
            thickness_um: Dura layer thickness in micrometers
            attachment_points: List of attachment points
            
        Returns:
            3D surface mesh array
        """
        logger.info("Generating dura mater surface mesh")
        
        # Generate outer surface at specified thickness
        surface_mesh = self._generate_outer_surface(cavity_mask, thickness_um)
        
        # Apply attachment point constraints
        surface_mesh = self._apply_attachment_constraints(surface_mesh, attachment_points)
        
        # Smooth the surface
        surface_mesh = self._smooth_surface_mesh(surface_mesh)
        
        mesh_points = np.sum(surface_mesh > 0)
        logger.info(f"Generated dura surface mesh: {mesh_points} surface points")
        
        return surface_mesh
    
    def _generate_outer_surface(self, cavity_mask: np.ndarray, thickness_um: float) -> np.ndarray:
        """Generate outer surface at specified thickness from brain tissue."""
        # Create brain tissue mask (inverse of cavity mask)
        brain_tissue = ~cavity_mask
        
        # Dilate brain tissue by dura thickness
        thickness_voxels = int(thickness_um / self.dims.resolution)
        structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        
        # Iterative dilation to achieve desired thickness
        dilated_brain = brain_tissue
        for _ in range(thickness_voxels):
            dilated_brain = ndimage.binary_dilation(dilated_brain, structure=structure)
        
        # Dura surface is the boundary of dilated brain
        dura_surface = dilated_brain & ~ndimage.binary_erosion(dilated_brain, structure=structure)
        
        return dura_surface.astype(float)
    
    def _apply_attachment_constraints(self, surface_mesh: np.ndarray,
                                    attachment_points: List[AttachmentPoint]) -> np.ndarray:
        """Apply attachment point constraints to surface mesh."""
        constrained_mesh = surface_mesh.copy()
        
        # Apply constraints at each attachment point
        for attachment in attachment_points:
            # Convert attachment location to voxel coordinates
            attach_vox = (
                int(attachment.location[0] / self.dims.resolution),
                int(attachment.location[1] / self.dims.resolution),
                int(attachment.location[2] / self.dims.resolution)
            )
            
            # Ensure attachment point is within grid bounds
            attach_vox = (
                np.clip(attach_vox[0], 0, self.dims.x_size - 1),
                np.clip(attach_vox[1], 0, self.dims.y_size - 1),
                np.clip(attach_vox[2], 0, self.dims.z_size - 1)
            )
            
            # Create attachment region
            radius_voxels = int(attachment.region_radius_um / self.dims.resolution)
            
            # Mark attachment region in surface mesh
            for dx in range(-radius_voxels, radius_voxels + 1):
                for dy in range(-radius_voxels, radius_voxels + 1):
                    for dz in range(-radius_voxels, radius_voxels + 1):
                        x, y, z = (attach_vox[0] + dx, attach_vox[1] + dy, attach_vox[2] + dz)
                        
                        if (0 <= x < self.dims.x_size and 0 <= y < self.dims.y_size and 0 <= z < self.dims.z_size):
                            distance = np.sqrt(dx**2 + dy**2 + dz**2) * self.dims.resolution
                            
                            if distance <= attachment.region_radius_um:
                                # Strengthen surface at attachment points
                                constrained_mesh[x, y, z] = max(
                                    constrained_mesh[x, y, z], 
                                    attachment.attachment_strength_n * 1e6  # Normalize to mesh units
                                )
        
        return constrained_mesh
    
    def _smooth_surface_mesh(self, surface_mesh: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Apply smoothing to surface mesh."""
        smoothed_mesh = surface_mesh.copy()
        
        for i in range(iterations):
            # Gaussian smoothing
            smoothed_mesh = ndimage.gaussian_filter(smoothed_mesh, sigma=1.0)
            
            # Preserve attachment points (high values)
            attachment_mask = surface_mesh > (0.8 * np.max(surface_mesh))
            smoothed_mesh[attachment_mask] = surface_mesh[attachment_mask]
        
        return smoothed_mesh
    
    def get_thickness_map(self, surface_mesh: np.ndarray, 
                         base_thickness_um: float,
                         stress_field: Optional[np.ndarray] = None) -> np.ndarray:
        """Get 3D thickness map of dura mater layer.
        
        Args:
            surface_mesh: Surface mesh array
            base_thickness_um: Base thickness in micrometers
            stress_field: Optional stress field for thickness variation
            
        Returns:
            3D array with thickness values (µm)
        """
        thickness_map = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size))
        
        # Apply base thickness where surface exists
        surface_mask = surface_mesh > 0.1 * np.max(surface_mesh)
        thickness_map[surface_mask] = base_thickness_um
        
        # Vary thickness based on stress (thicker where more stress)
        if stress_field is not None:
            max_stress = np.max(stress_field)
            if max_stress > 0:
                normalized_stress = stress_field / max_stress
                thickness_variation = 0.3 * base_thickness_um * normalized_stress  # ±30% variation
                thickness_map += thickness_variation
                
                # Ensure minimum thickness
                min_thickness = 0.5 * base_thickness_um
                thickness_map[surface_mask] = np.maximum(
                    thickness_map[surface_mask], min_thickness)
        
        return thickness_map
