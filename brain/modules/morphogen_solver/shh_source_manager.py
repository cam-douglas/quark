#!/usr/bin/env python3
"""SHH Source Region Management.

Manages SHH source regions (floor plate and notochord) for neural tube
morphogen gradient generation during embryonic development.

Integration: Component of SHH gradient system
Rationale: Focused module for source region configuration and management
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .parameter_types import SourceParameters

logger = logging.getLogger(__name__)

@dataclass
class SHHSourceConfiguration:
    """Configuration for SHH source regions."""
    floor_plate_center: Tuple[int, int, int]  # Floor plate center (x, y, z)
    floor_plate_width: int                    # Width in voxels
    notochord_center: Tuple[int, int, int]    # Notochord center (x, y, z)  
    notochord_radius: int                     # Radius in voxels
    production_intensity: float               # Production rate (nM/s)

class SHHSourceManager:
    """Manages SHH source regions for neural tube morphogen gradients.
    
    Handles configuration and application of SHH source regions including:
    - Floor plate: Linear ventral midline source
    - Notochord: Spherical source ventral to neural tube
    - Dynamic source intensity modulation
    - Anatomically-accurate positioning
    
    Key Features:
    - Biologically-accurate source positioning
    - Configurable source geometries
    - Production term calculation for reaction-diffusion
    - Source validation and monitoring
    """
    
    def __init__(self, spatial_grid: SpatialGrid, source_params: SourceParameters):
        """Initialize SHH source manager.
        
        Args:
            spatial_grid: 3D spatial grid for source application
            source_params: Biological source parameters
        """
        self.grid = spatial_grid
        self.source_params = source_params
        self.source_config: Optional[SHHSourceConfiguration] = None
        
        # Source region masks (for efficient production term calculation)
        self.floor_plate_mask: Optional[np.ndarray] = None
        self.notochord_mask: Optional[np.ndarray] = None
        
        logger.info("Initialized SHH source manager")
    
    def configure_sources(self, neural_tube_dimensions: GridDimensions) -> None:
        """Configure SHH source regions based on neural tube geometry.
        
        Args:
            neural_tube_dimensions: Neural tube spatial dimensions
        """
        # Calculate source positions based on neural tube anatomy
        # Floor plate: ventral midline of neural tube
        floor_plate_x = neural_tube_dimensions.x_size // 2  # Center A-P
        floor_plate_y = neural_tube_dimensions.y_size - 10  # Ventral (bottom)
        floor_plate_z = neural_tube_dimensions.z_size // 2  # Center L-R
        
        # Notochord: ventral to neural tube, provides additional SHH
        notochord_x = floor_plate_x
        notochord_y = neural_tube_dimensions.y_size - 5     # Just ventral to floor plate
        notochord_z = floor_plate_z
        
        self.source_config = SHHSourceConfiguration(
            floor_plate_center=(floor_plate_x, floor_plate_y, floor_plate_z),
            floor_plate_width=8,  # voxels (≈8 µm wide floor plate)
            notochord_center=(notochord_x, notochord_y, notochord_z),
            notochord_radius=5,   # voxels (≈5 µm radius notochord)
            production_intensity=self.source_params.intensity
        )
        
        # Create source region masks
        self._create_source_masks()
        
        # Apply initial source regions to spatial grid
        self._apply_source_regions()
        
        logger.info(f"Configured SHH sources:")
        logger.info(f"  Floor plate: {self.source_config.floor_plate_center}")
        logger.info(f"  Notochord: {self.source_config.notochord_center}")
        logger.info(f"  Production intensity: {self.source_config.production_intensity} nM/s")
    
    def _create_source_masks(self) -> None:
        """Create binary masks for source regions for efficient computation."""
        if not self.source_config:
            raise ValueError("Source configuration not set")
        
        grid_shape = (self.grid.dimensions.x_size, 
                     self.grid.dimensions.y_size, 
                     self.grid.dimensions.z_size)
        
        # Initialize masks
        self.floor_plate_mask = np.zeros(grid_shape, dtype=bool)
        self.notochord_mask = np.zeros(grid_shape, dtype=bool)
        
        # Floor plate mask (linear source along A-P axis)
        fp_x, fp_y, fp_z = self.source_config.floor_plate_center
        fp_width = self.source_config.floor_plate_width
        
        for x_offset in range(-fp_width//2, fp_width//2 + 1):
            for z_offset in range(-2, 3):  # 5 voxels wide in L-R direction
                x_pos = fp_x + x_offset
                z_pos = fp_z + z_offset
                
                if (0 <= x_pos < self.grid.dimensions.x_size and 
                    0 <= z_pos < self.grid.dimensions.z_size):
                    self.floor_plate_mask[x_pos, fp_y, z_pos] = True
        
        # Notochord mask (spherical source)
        nc_x, nc_y, nc_z = self.source_config.notochord_center
        nc_radius = self.source_config.notochord_radius
        
        # Create coordinate grids
        x_coords = np.arange(self.grid.dimensions.x_size)
        y_coords = np.arange(self.grid.dimensions.y_size)
        z_coords = np.arange(self.grid.dimensions.z_size)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Calculate distances from notochord center
        distances = np.sqrt((X - nc_x)**2 + (Y - nc_y)**2 + (Z - nc_z)**2)
        self.notochord_mask = distances <= nc_radius
        
        # Log mask statistics
        fp_voxels = np.sum(self.floor_plate_mask)
        nc_voxels = np.sum(self.notochord_mask)
        logger.info(f"Created source masks: FP={fp_voxels} voxels, NC={nc_voxels} voxels")
    
    def _apply_source_regions(self) -> None:
        """Apply SHH source regions to spatial grid."""
        if not self.source_config or self.floor_plate_mask is None:
            raise ValueError("Source configuration or masks not set")
        
        # Floor plate source - full intensity
        self.grid.concentrations['SHH'][self.floor_plate_mask] = self.source_config.production_intensity
        
        # Notochord source - 70% intensity (weaker than floor plate)
        notochord_intensity = self.source_config.production_intensity * 0.7
        self.grid.concentrations['SHH'][self.notochord_mask] = notochord_intensity
        
        logger.info("Applied SHH source regions to spatial grid")
    
    def calculate_production_term(self) -> np.ndarray:
        """Calculate production term for reaction-diffusion equation.
        
        Returns:
            Production term array (nM/s) for each voxel
        """
        if not self.source_config or self.floor_plate_mask is None:
            return np.zeros((self.grid.dimensions.x_size, 
                           self.grid.dimensions.y_size, 
                           self.grid.dimensions.z_size))
        
        production_term = np.zeros((self.grid.dimensions.x_size, 
                                  self.grid.dimensions.y_size, 
                                  self.grid.dimensions.z_size))
        
        # Floor plate production
        production_term[self.floor_plate_mask] = self.source_config.production_intensity
        
        # Notochord production (70% intensity)
        production_term[self.notochord_mask] = self.source_config.production_intensity * 0.7
        
        return production_term
    
    def modulate_source_intensity(self, time_hours: float) -> None:
        """Modulate source intensity based on developmental time.
        
        Args:
            time_hours: Current developmental time in hours
        """
        if not self.source_config:
            return
        
        # Developmental time-dependent modulation
        # Peak expression around E9.0-E9.5 (24-36 hours after E8.5)
        base_intensity = self.source_params.intensity
        
        if time_hours < 12:
            # Early phase: gradual increase
            intensity_factor = 0.3 + 0.7 * (time_hours / 12.0)
        elif time_hours < 48:
            # Peak phase: sustained high expression
            intensity_factor = 1.0
        elif time_hours < 96:
            # Late phase: gradual decrease
            intensity_factor = 1.0 - 0.4 * ((time_hours - 48) / 48.0)
        else:
            # Very late: minimal expression
            intensity_factor = 0.6
        
        self.source_config.production_intensity = base_intensity * intensity_factor
        
        logger.debug(f"Modulated SHH source intensity: {intensity_factor:.2f}x at t={time_hours:.1f}h")
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about SHH source regions.
        
        Returns:
            Dictionary of source statistics
        """
        if not self.source_config or self.floor_plate_mask is None:
            return {"configured": False}
        
        # Calculate source region properties
        fp_voxels = int(np.sum(self.floor_plate_mask))
        nc_voxels = int(np.sum(self.notochord_mask))
        
        # Calculate volumes (µm³)
        voxel_volume = self.grid.resolution ** 3
        fp_volume = fp_voxels * voxel_volume
        nc_volume = nc_voxels * voxel_volume
        
        # Get current concentrations at source centers
        fp_x, fp_y, fp_z = self.source_config.floor_plate_center
        nc_x, nc_y, nc_z = self.source_config.notochord_center
        
        fp_concentration = self.grid.get_concentration('SHH', fp_x, fp_y, fp_z)
        nc_concentration = self.grid.get_concentration('SHH', nc_x, nc_y, nc_z)
        
        return {
            "configured": True,
            "floor_plate": {
                "center": self.source_config.floor_plate_center,
                "width_voxels": self.source_config.floor_plate_width,
                "volume_um3": fp_volume,
                "voxel_count": fp_voxels,
                "current_concentration_nM": float(fp_concentration)
            },
            "notochord": {
                "center": self.source_config.notochord_center,
                "radius_voxels": self.source_config.notochord_radius,
                "volume_um3": nc_volume,
                "voxel_count": nc_voxels,
                "current_concentration_nM": float(nc_concentration)
            },
            "production": {
                "intensity_nM_per_s": self.source_config.production_intensity,
                "total_production_rate": (fp_voxels + nc_voxels * 0.7) * self.source_config.production_intensity
            }
        }
    
    def validate_sources(self) -> Dict[str, Any]:
        """Validate source region configuration and properties.
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not self.source_config:
            validation["is_valid"] = False
            validation["errors"].append("Source configuration not set")
            return validation
        
        # Check source positions are within grid bounds
        fp_x, fp_y, fp_z = self.source_config.floor_plate_center
        nc_x, nc_y, nc_z = self.source_config.notochord_center
        
        if not (0 <= fp_x < self.grid.dimensions.x_size and
                0 <= fp_y < self.grid.dimensions.y_size and
                0 <= fp_z < self.grid.dimensions.z_size):
            validation["is_valid"] = False
            validation["errors"].append("Floor plate center outside grid bounds")
        
        if not (0 <= nc_x < self.grid.dimensions.x_size and
                0 <= nc_y < self.grid.dimensions.y_size and
                0 <= nc_z < self.grid.dimensions.z_size):
            validation["is_valid"] = False
            validation["errors"].append("Notochord center outside grid bounds")
        
        # Check anatomical positioning (notochord should be ventral to floor plate)
        if nc_y <= fp_y:
            validation["warnings"].append("Notochord not positioned ventral to floor plate")
        
        # Check source intensity
        if self.source_config.production_intensity <= 0:
            validation["is_valid"] = False
            validation["errors"].append("Production intensity must be positive")
        
        # Check source overlap
        if (self.floor_plate_mask is not None and self.notochord_mask is not None and
            np.any(self.floor_plate_mask & self.notochord_mask)):
            validation["warnings"].append("Floor plate and notochord sources overlap")
        
        return validation
    
    def reset_sources(self) -> None:
        """Reset source configuration and masks."""
        self.source_config = None
        self.floor_plate_mask = None
        self.notochord_mask = None
        
        # Clear SHH concentrations in source regions
        if 'SHH' in self.grid.concentrations:
            self.grid.concentrations['SHH'].fill(0.0)
        
        logger.info("Reset SHH source configuration")
