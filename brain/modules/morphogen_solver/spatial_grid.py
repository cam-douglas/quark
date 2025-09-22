#!/usr/bin/env python3
"""3D Spatial Grid System for Morphogen Gradient Simulation.

Implements a high-resolution 3D voxel grid system (1µm³ resolution) for 
spatial morphogen concentration modeling during neural tube development.

Integration: Foundation for all morphogen gradient calculations
Rationale: Provides spatial framework for SHH/BMP/WNT/FGF gradient fields
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GridDimensions:
    """3D grid dimensions specification."""
    x_size: int  # Anterior-posterior axis (voxels)
    y_size: int  # Dorsal-ventral axis (voxels)
    z_size: int  # Left-right axis (voxels)
    resolution: float = 1.0  # µm per voxel
    
    @property
    def total_voxels(self) -> int:
        """Total number of voxels in the grid."""
        return self.x_size * self.y_size * self.z_size
    
    @property
    def volume_mm3(self) -> float:
        """Total volume in mm³."""
        return (self.x_size * self.y_size * self.z_size) / (1000**3)

class SpatialGrid:
    """3D spatial grid for morphogen concentration modeling.
    
    Provides high-resolution (1µm³) voxel grid for neural tube morphogen
    gradient simulation during embryonic development weeks 3-8.
    
    Key Features:
    - 1µm³ voxel resolution for sub-millimeter precision
    - Efficient memory management for large grids
    - Coordinate system mapping (biological → computational)
    - Boundary condition handling
    - Integration with morphogen solvers
    """
    
    def __init__(self, dimensions: GridDimensions):
        """Initialize 3D spatial grid.
        
        Args:
            dimensions: Grid size and resolution specification
        """
        self.dimensions = dimensions
        self.resolution = dimensions.resolution
        
        # Initialize concentration arrays for each morphogen
        self.concentrations: Dict[str, np.ndarray] = {}
        
        # Coordinate mapping arrays
        self._init_coordinate_system()
        
        # Boundary conditions
        self.boundary_conditions: Dict[str, str] = {}
        
        logger.info(f"Initialized SpatialGrid: {dimensions.x_size}x{dimensions.y_size}x{dimensions.z_size} "
                   f"voxels at {dimensions.resolution}µm resolution")
        logger.info(f"Total volume: {dimensions.volume_mm3:.3f} mm³, "
                   f"Total voxels: {dimensions.total_voxels:,}")
    
    def _init_coordinate_system(self) -> None:
        """Initialize coordinate mapping arrays."""
        # Biological coordinate system:
        # X-axis: Anterior (rostral) to Posterior (caudal)
        # Y-axis: Dorsal (top) to Ventral (bottom) 
        # Z-axis: Left to Right
        
        x_coords = np.arange(self.dimensions.x_size, dtype=np.float64) * self.resolution
        y_coords = np.arange(self.dimensions.y_size, dtype=np.float64) * self.resolution
        z_coords = np.arange(self.dimensions.z_size, dtype=np.float64) * self.resolution
        
        self.x_coords, self.y_coords, self.z_coords = np.meshgrid(
            x_coords, y_coords, z_coords, indexing='ij'
        )
        
        # Center coordinates around origin for biological relevance
        self.x_coords -= self.x_coords.mean()
        self.y_coords -= self.y_coords.mean()
        self.z_coords -= self.z_coords.mean()
    
    def add_morphogen(self, name: str, initial_concentration: float = 0.0) -> None:
        """Add a morphogen concentration field to the grid.
        
        Args:
            name: Morphogen identifier (e.g., 'SHH', 'BMP', 'WNT', 'FGF')
            initial_concentration: Initial concentration value (nM)
        """
        shape = (self.dimensions.x_size, self.dimensions.y_size, self.dimensions.z_size)
        self.concentrations[name] = np.full(shape, initial_concentration, dtype=np.float32)
        
        # Default boundary conditions (no-flux)
        self.boundary_conditions[name] = 'no_flux'
        
        logger.info(f"Added morphogen '{name}' with initial concentration {initial_concentration} nM")
    
    def set_concentration(self, morphogen: str, x: int, y: int, z: int, 
                         concentration: float) -> None:
        """Set concentration at specific voxel coordinates.
        
        Args:
            morphogen: Morphogen name
            x, y, z: Voxel indices
            concentration: Concentration value (nM)
        """
        if morphogen not in self.concentrations:
            raise ValueError(f"Morphogen '{morphogen}' not found in grid")
        
        if not (0 <= x < self.dimensions.x_size and 
                0 <= y < self.dimensions.y_size and
                0 <= z < self.dimensions.z_size):
            raise IndexError(f"Coordinates ({x}, {y}, {z}) out of grid bounds")
        
        self.concentrations[morphogen][x, y, z] = concentration
    
    def get_concentration(self, morphogen: str, x: int, y: int, z: int) -> float:
        """Get concentration at specific voxel coordinates.
        
        Args:
            morphogen: Morphogen name
            x, y, z: Voxel indices
            
        Returns:
            Concentration value (nM)
        """
        if morphogen not in self.concentrations:
            raise ValueError(f"Morphogen '{morphogen}' not found in grid")
        
        return float(self.concentrations[morphogen][x, y, z])
    
    def has_morphogen(self, morphogen: str) -> bool:
        """Check if morphogen exists in grid.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            True if morphogen exists
        """
        return morphogen in self.concentrations
    
    def get_morphogen_concentration(self, morphogen: str) -> np.ndarray:
        """Get full concentration field for morphogen.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            3D concentration array
        """
        if morphogen not in self.concentrations:
            raise ValueError(f"Morphogen '{morphogen}' not found in grid")
        
        return self.concentrations[morphogen].copy()
    
    def set_morphogen_concentration(self, morphogen: str, concentration_field: np.ndarray) -> None:
        """Set full concentration field for morphogen.
        
        Args:
            morphogen: Morphogen name
            concentration_field: 3D concentration array
        """
        if morphogen not in self.concentrations:
            raise ValueError(f"Morphogen '{morphogen}' not found in grid")
        
        expected_shape = (self.dimensions.x_size, self.dimensions.y_size, self.dimensions.z_size)
        if concentration_field.shape != expected_shape:
            raise ValueError(f"Concentration field shape {concentration_field.shape} does not match grid {expected_shape}")
        
        self.concentrations[morphogen] = concentration_field.astype(np.float32)
    
    def set_source_region(self, morphogen: str, center: Tuple[int, int, int], 
                         radius: int, concentration: float) -> None:
        """Set a spherical source region for morphogen production.
        
        Args:
            morphogen: Morphogen name
            center: Center coordinates (x, y, z) in voxels
            radius: Source radius in voxels
            concentration: Source concentration (nM)
        """
        if morphogen not in self.concentrations:
            raise ValueError(f"Morphogen '{morphogen}' not found in grid")
        
        cx, cy, cz = center
        
        # Create spherical mask
        x_indices = np.arange(self.dimensions.x_size)
        y_indices = np.arange(self.dimensions.y_size)
        z_indices = np.arange(self.dimensions.z_size)
        
        X, Y, Z = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Calculate distance from center
        distances = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        
        # Set concentration in spherical region
        mask = distances <= radius
        self.concentrations[morphogen][mask] = concentration
        
        logger.info(f"Set {morphogen} source region at ({cx}, {cy}, {cz}) "
                   f"with radius {radius} voxels, concentration {concentration} nM")
    
    def get_gradient(self, morphogen: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate concentration gradient using finite differences.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            Gradient components (grad_x, grad_y, grad_z) in nM/µm
        """
        if morphogen not in self.concentrations:
            raise ValueError(f"Morphogen '{morphogen}' not found in grid")
        
        concentration = self.concentrations[morphogen]
        
        # Calculate gradients using central differences
        grad_x = np.gradient(concentration, self.resolution, axis=0)
        grad_y = np.gradient(concentration, self.resolution, axis=1) 
        grad_z = np.gradient(concentration, self.resolution, axis=2)
        
        return grad_x, grad_y, grad_z
    
    def get_laplacian(self, morphogen: str) -> np.ndarray:
        """Calculate Laplacian for diffusion calculations.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            Laplacian array (∇²C) in nM/µm²
        """
        if morphogen not in self.concentrations:
            raise ValueError(f"Morphogen '{morphogen}' not found in grid")
        
        concentration = self.concentrations[morphogen]
        
        # Calculate second derivatives
        d2_dx2 = np.gradient(np.gradient(concentration, self.resolution, axis=0), 
                            self.resolution, axis=0)
        d2_dy2 = np.gradient(np.gradient(concentration, self.resolution, axis=1), 
                            self.resolution, axis=1)
        d2_dz2 = np.gradient(np.gradient(concentration, self.resolution, axis=2), 
                            self.resolution, axis=2)
        
        return d2_dx2 + d2_dy2 + d2_dz2
    
    def apply_boundary_conditions(self, morphogen: str) -> None:
        """Apply boundary conditions to morphogen field.
        
        Args:
            morphogen: Morphogen name
        """
        if morphogen not in self.concentrations:
            return
        
        boundary_type = self.boundary_conditions.get(morphogen, 'no_flux')
        concentration = self.concentrations[morphogen]
        
        if boundary_type == 'no_flux':
            # No-flux boundary conditions (Neumann)
            # Set boundary values equal to adjacent interior values
            
            # X boundaries
            concentration[0, :, :] = concentration[1, :, :]
            concentration[-1, :, :] = concentration[-2, :, :]
            
            # Y boundaries  
            concentration[:, 0, :] = concentration[:, 1, :]
            concentration[:, -1, :] = concentration[:, -2, :]
            
            # Z boundaries
            concentration[:, :, 0] = concentration[:, :, 1]
            concentration[:, :, -1] = concentration[:, :, -2]
            
        elif boundary_type == 'zero_concentration':
            # Dirichlet boundary conditions (zero concentration)
            concentration[0, :, :] = 0
            concentration[-1, :, :] = 0
            concentration[:, 0, :] = 0
            concentration[:, -1, :] = 0
            concentration[:, :, 0] = 0
            concentration[:, :, -1] = 0
    
    def get_memory_usage_mb(self) -> float:
        """Calculate current memory usage in MB."""
        total_bytes = sum(arr.nbytes for arr in self.concentrations.values())
        # Add coordinate arrays
        total_bytes += self.x_coords.nbytes + self.y_coords.nbytes + self.z_coords.nbytes
        return total_bytes / (1024 * 1024)
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Get comprehensive grid information."""
        return {
            "dimensions": {
                "x_size": self.dimensions.x_size,
                "y_size": self.dimensions.y_size, 
                "z_size": self.dimensions.z_size,
                "resolution_um": self.resolution,
                "total_voxels": self.dimensions.total_voxels,
                "volume_mm3": self.dimensions.volume_mm3
            },
            "morphogens": list(self.concentrations.keys()),
            "memory_usage_mb": self.get_memory_usage_mb(),
            "boundary_conditions": self.boundary_conditions.copy()
        }
