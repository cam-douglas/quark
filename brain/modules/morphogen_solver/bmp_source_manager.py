#!/usr/bin/env python3
"""BMP Source Region Management.

Manages BMP source regions (roof plate and dorsal ectoderm) for neural tube
morphogen gradient generation during embryonic development.

Integration: Component of BMP gradient system
Rationale: Focused module for dorsal source region configuration and management
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .parameter_types import SourceParameters

logger = logging.getLogger(__name__)

@dataclass
class BMPSourceConfiguration:
    """Configuration for BMP source regions."""
    roof_plate_center: Tuple[int, int, int]   # Roof plate center (x, y, z)
    roof_plate_width: int                     # Width in voxels
    dorsal_ectoderm_center: Tuple[int, int, int]  # Dorsal ectoderm center (x, y, z)  
    dorsal_ectoderm_extent: int               # Extent in voxels
    production_intensity: float               # Production rate (nM/s)

class BMPSourceManager:
    """Manages BMP source regions for neural tube morphogen gradients.
    
    Handles configuration and application of BMP source regions including:
    - Roof plate: Linear dorsal midline source (opposite to floor plate)
    - Dorsal ectoderm: Broad dorsal source dorsal to neural tube
    - Dynamic source intensity modulation
    - Anatomically-accurate positioning
    
    Key Features:
    - Biologically-accurate dorsal source positioning
    - Configurable source geometries
    - Production term calculation for reaction-diffusion
    - Source validation and monitoring
    """
    
    def __init__(self, spatial_grid: SpatialGrid, source_params: SourceParameters):
        """Initialize BMP source manager.
        
        Args:
            spatial_grid: 3D spatial grid for source application
            source_params: Biological source parameters
        """
        self.grid = spatial_grid
        self.source_params = source_params
        self.source_config: Optional[BMPSourceConfiguration] = None
        
        # Source region masks (for efficient production term calculation)
        self.roof_plate_mask: Optional[np.ndarray] = None
        self.dorsal_ectoderm_mask: Optional[np.ndarray] = None
        
        logger.info("Initialized BMP source manager")
    
    def configure_sources(self, neural_tube_dimensions: GridDimensions) -> None:
        """Configure BMP source regions based on neural tube geometry.
        
        Args:
            neural_tube_dimensions: Neural tube spatial dimensions
        """
        # Calculate source positions based on neural tube anatomy (dorsal sources)
        # Roof plate: dorsal midline of neural tube (opposite to floor plate)
        roof_plate_x = neural_tube_dimensions.x_size // 2  # Center A-P
        roof_plate_y = 10  # Dorsal (top) - opposite to SHH ventral sources
        roof_plate_z = neural_tube_dimensions.z_size // 2  # Center L-R
        
        # Dorsal ectoderm: dorsal to neural tube, provides additional BMP
        dorsal_ectoderm_x = roof_plate_x
        dorsal_ectoderm_y = 5  # Just dorsal to roof plate
        dorsal_ectoderm_z = roof_plate_z
        
        self.source_config = BMPSourceConfiguration(
            roof_plate_center=(roof_plate_x, roof_plate_y, roof_plate_z),
            roof_plate_width=8,  # voxels (≈8 µm wide roof plate)
            dorsal_ectoderm_center=(dorsal_ectoderm_x, dorsal_ectoderm_y, dorsal_ectoderm_z),
            dorsal_ectoderm_extent=12,  # voxels (broader than roof plate)
            production_intensity=self.source_params.intensity
        )
        
        # Create source region masks
        self._create_source_masks()
        
        # Apply initial source regions to spatial grid
        self._apply_source_regions()
        
        logger.info(f"Configured BMP sources:")
        logger.info(f"  Roof plate: {self.source_config.roof_plate_center}")
        logger.info(f"  Dorsal ectoderm: {self.source_config.dorsal_ectoderm_center}")
        logger.info(f"  Production intensity: {self.source_config.production_intensity} nM/s")
    
    def _create_source_masks(self) -> None:
        """Create binary masks for source regions for efficient computation."""
        if not self.source_config:
            raise ValueError("Source configuration not set")
        
        grid_shape = (self.grid.dimensions.x_size, 
                     self.grid.dimensions.y_size, 
                     self.grid.dimensions.z_size)
        
        # Initialize masks
        self.roof_plate_mask = np.zeros(grid_shape, dtype=bool)
        self.dorsal_ectoderm_mask = np.zeros(grid_shape, dtype=bool)
        
        # Roof plate mask (linear source along A-P axis, dorsal)
        rp_x, rp_y, rp_z = self.source_config.roof_plate_center
        rp_width = self.source_config.roof_plate_width
        
        for x_offset in range(-rp_width//2, rp_width//2 + 1):
            for z_offset in range(-2, 3):  # 5 voxels wide in L-R direction
                x_pos = rp_x + x_offset
                z_pos = rp_z + z_offset
                
                if (0 <= x_pos < self.grid.dimensions.x_size and 
                    0 <= z_pos < self.grid.dimensions.z_size):
                    self.roof_plate_mask[x_pos, rp_y, z_pos] = True
        
        # Dorsal ectoderm mask (broader dorsal source)
        de_x, de_y, de_z = self.source_config.dorsal_ectoderm_center
        de_extent = self.source_config.dorsal_ectoderm_extent
        
        # Create coordinate grids
        x_coords = np.arange(self.grid.dimensions.x_size)
        y_coords = np.arange(self.grid.dimensions.y_size)
        z_coords = np.arange(self.grid.dimensions.z_size)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Create elliptical dorsal ectoderm region
        x_dist = (X - de_x) / (de_extent * 0.8)
        y_dist = (Y - de_y) / 3  # Thin in Y direction
        z_dist = (Z - de_z) / de_extent
        
        distances = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
        self.dorsal_ectoderm_mask = distances <= 1.0
        
        # Log mask statistics
        rp_voxels = np.sum(self.roof_plate_mask)
        de_voxels = np.sum(self.dorsal_ectoderm_mask)
        logger.info(f"Created source masks: RP={rp_voxels} voxels, DE={de_voxels} voxels")
    
    def _apply_source_regions(self) -> None:
        """Apply BMP source regions to spatial grid."""
        if not self.source_config or self.roof_plate_mask is None:
            raise ValueError("Source configuration or masks not set")
        
        # Roof plate source - full intensity
        self.grid.concentrations['BMP'][self.roof_plate_mask] = self.source_config.production_intensity
        
        # Dorsal ectoderm source - 80% intensity (slightly weaker than roof plate)
        dorsal_ectoderm_intensity = self.source_config.production_intensity * 0.8
        self.grid.concentrations['BMP'][self.dorsal_ectoderm_mask] = dorsal_ectoderm_intensity
        
        logger.info("Applied BMP source regions to spatial grid")
    
    def calculate_production_term(self) -> np.ndarray:
        """Calculate production term for reaction-diffusion equation.
        
        Returns:
            Production term array (nM/s) for each voxel
        """
        if not self.source_config or self.roof_plate_mask is None:
            return np.zeros((self.grid.dimensions.x_size, 
                           self.grid.dimensions.y_size, 
                           self.grid.dimensions.z_size))
        
        production_term = np.zeros((self.grid.dimensions.x_size, 
                                  self.grid.dimensions.y_size, 
                                  self.grid.dimensions.z_size))
        
        # Roof plate production
        production_term[self.roof_plate_mask] = self.source_config.production_intensity
        
        # Dorsal ectoderm production (80% intensity)
        production_term[self.dorsal_ectoderm_mask] = self.source_config.production_intensity * 0.8
        
        return production_term
    
    def modulate_source_intensity(self, time_hours: float) -> None:
        """Modulate source intensity based on developmental time.
        
        Args:
            time_hours: Current developmental time in hours
        """
        if not self.source_config:
            return
        
        # Developmental time-dependent modulation for BMP
        # Peak expression around E8.5-E10.5 (similar to SHH but slightly delayed)
        base_intensity = self.source_params.intensity
        
        if time_hours < 8:
            # Early phase: gradual increase (slower than SHH)
            intensity_factor = 0.2 + 0.8 * (time_hours / 8.0)
        elif time_hours < 60:
            # Peak phase: sustained high expression
            intensity_factor = 1.0
        elif time_hours < 120:
            # Late phase: gradual decrease
            intensity_factor = 1.0 - 0.3 * ((time_hours - 60) / 60.0)
        else:
            # Very late: moderate expression (BMP maintains longer than SHH)
            intensity_factor = 0.7
        
        self.source_config.production_intensity = base_intensity * intensity_factor
        
        logger.debug(f"Modulated BMP source intensity: {intensity_factor:.2f}x at t={time_hours:.1f}h")
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about BMP source regions.
        
        Returns:
            Dictionary of source statistics
        """
        if not self.source_config or self.roof_plate_mask is None:
            return {"configured": False}
        
        # Calculate source region properties
        rp_voxels = int(np.sum(self.roof_plate_mask))
        de_voxels = int(np.sum(self.dorsal_ectoderm_mask))
        
        # Calculate volumes (µm³)
        voxel_volume = self.grid.resolution ** 3
        rp_volume = rp_voxels * voxel_volume
        de_volume = de_voxels * voxel_volume
        
        # Get current concentrations at source centers
        rp_x, rp_y, rp_z = self.source_config.roof_plate_center
        de_x, de_y, de_z = self.source_config.dorsal_ectoderm_center
        
        rp_concentration = self.grid.get_concentration('BMP', rp_x, rp_y, rp_z)
        de_concentration = self.grid.get_concentration('BMP', de_x, de_y, de_z)
        
        return {
            "configured": True,
            "roof_plate": {
                "center": self.source_config.roof_plate_center,
                "width_voxels": self.source_config.roof_plate_width,
                "volume_um3": rp_volume,
                "voxel_count": rp_voxels,
                "current_concentration_nM": float(rp_concentration)
            },
            "dorsal_ectoderm": {
                "center": self.source_config.dorsal_ectoderm_center,
                "extent_voxels": self.source_config.dorsal_ectoderm_extent,
                "volume_um3": de_volume,
                "voxel_count": de_voxels,
                "current_concentration_nM": float(de_concentration)
            },
            "production": {
                "intensity_nM_per_s": self.source_config.production_intensity,
                "total_production_rate": (rp_voxels + de_voxels * 0.8) * self.source_config.production_intensity
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
        rp_x, rp_y, rp_z = self.source_config.roof_plate_center
        de_x, de_y, de_z = self.source_config.dorsal_ectoderm_center
        
        if not (0 <= rp_x < self.grid.dimensions.x_size and
                0 <= rp_y < self.grid.dimensions.y_size and
                0 <= rp_z < self.grid.dimensions.z_size):
            validation["is_valid"] = False
            validation["errors"].append("Roof plate center outside grid bounds")
        
        if not (0 <= de_x < self.grid.dimensions.x_size and
                0 <= de_y < self.grid.dimensions.y_size and
                0 <= de_z < self.grid.dimensions.z_size):
            validation["is_valid"] = False
            validation["errors"].append("Dorsal ectoderm center outside grid bounds")
        
        # Check anatomical positioning (dorsal ectoderm should be dorsal to roof plate)
        if de_y >= rp_y:
            validation["warnings"].append("Dorsal ectoderm not positioned dorsal to roof plate")
        
        # Check source intensity
        if self.source_config.production_intensity <= 0:
            validation["is_valid"] = False
            validation["errors"].append("Production intensity must be positive")
        
        # Check source overlap
        if (self.roof_plate_mask is not None and self.dorsal_ectoderm_mask is not None and
            np.any(self.roof_plate_mask & self.dorsal_ectoderm_mask)):
            validation["warnings"].append("Roof plate and dorsal ectoderm sources overlap")
        
        return validation
    
    def reset_sources(self) -> None:
        """Reset source configuration and masks."""
        self.source_config = None
        self.roof_plate_mask = None
        self.dorsal_ectoderm_mask = None
        
        # Clear BMP concentrations in source regions
        if 'BMP' in self.grid.concentrations:
            self.grid.concentrations['BMP'].fill(0.0)
        
        logger.info("Reset BMP source configuration")
