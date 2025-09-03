#!/usr/bin/env python3
"""Spatial Grid Module - Spatial organization and cell positioning system.

Manages spatial grid for cell positioning, movement, and spatial queries.

Integration: Spatial management for biological simulation workflows.
Rationale: Specialized spatial logic separate from biological processes.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class SpatialGrid:
    """Manages spatial grid for biological simulation."""
    
    def __init__(self, dimensions: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0),
                 resolution: float = 10.0):
        self.dimensions = dimensions
        self.resolution = resolution
        
        # Calculate grid size
        self.grid_shape = tuple(int(dim / resolution) for dim in dimensions)
        
        # Initialize spatial grid
        self.spatial_grid = self._initialize_spatial_grid()
        self.cell_positions = {}
        self.occupied_positions = set()
    
    def _initialize_spatial_grid(self) -> np.ndarray:
        """Initialize 3D spatial grid for tracking cell positions."""
        # Create 3D grid with cell density tracking
        grid = np.zeros(self.grid_shape, dtype=np.float32)
        
        # Initialize with background tissue density
        background_density = 0.1
        grid.fill(background_density)
        
        return grid
    
    def add_cell_to_grid(self, cell_id: str, position: Tuple[float, float, float]) -> bool:
        """Add a cell to the spatial grid."""
        grid_position = self._world_to_grid_coordinates(position)
        
        if self._is_valid_grid_position(grid_position):
            # Update grid density
            self.spatial_grid[grid_position] += 1.0
            
            # Track cell position
            self.cell_positions[cell_id] = position
            self.occupied_positions.add(grid_position)
            
            return True
        
        return False
    
    def remove_cell_from_grid(self, cell_id: str) -> bool:
        """Remove a cell from the spatial grid."""
        if cell_id in self.cell_positions:
            position = self.cell_positions[cell_id]
            grid_position = self._world_to_grid_coordinates(position)
            
            if self._is_valid_grid_position(grid_position):
                # Update grid density
                self.spatial_grid[grid_position] = max(0.0, self.spatial_grid[grid_position] - 1.0)
                
                # Remove tracking
                del self.cell_positions[cell_id]
                if self.spatial_grid[grid_position] == 0.0:
                    self.occupied_positions.discard(grid_position)
                
                return True
        
        return False
    
    def update_cell_position(self, cell_id: str, new_position: Tuple[float, float, float]) -> bool:
        """Update a cell's position in the grid."""
        if cell_id in self.cell_positions:
            # Remove from old position
            self.remove_cell_from_grid(cell_id)
            
            # Add to new position
            return self.add_cell_to_grid(cell_id, new_position)
        
        return False
    
    def get_local_cell_density(self, position: Tuple[float, float, float], 
                              radius: float = 50.0) -> float:
        """Get cell density in a local region around a position."""
        grid_position = self._world_to_grid_coordinates(position)
        
        if not self._is_valid_grid_position(grid_position):
            return 0.0
        
        # Calculate radius in grid coordinates
        grid_radius = int(radius / self.resolution)
        
        # Get local region
        x, y, z = grid_position
        x_min, x_max = max(0, x - grid_radius), min(self.grid_shape[0], x + grid_radius + 1)
        y_min, y_max = max(0, y - grid_radius), min(self.grid_shape[1], y + grid_radius + 1)
        z_min, z_max = max(0, z - grid_radius), min(self.grid_shape[2], z + grid_radius + 1)
        
        local_region = self.spatial_grid[x_min:x_max, y_min:y_max, z_min:z_max]
        
        return float(np.mean(local_region))
    
    def find_empty_positions(self, count: int = 10, min_distance: float = 20.0) -> List[Tuple[float, float, float]]:
        """Find empty positions suitable for new cells."""
        empty_positions = []
        attempts = 0
        max_attempts = count * 10
        
        while len(empty_positions) < count and attempts < max_attempts:
            # Random position within dimensions
            position = (
                np.random.uniform(0, self.dimensions[0]),
                np.random.uniform(0, self.dimensions[1]),
                np.random.uniform(0, self.dimensions[2])
            )
            
            # Check if position is sufficiently empty
            local_density = self.get_local_cell_density(position, min_distance)
            if local_density < 0.5:  # Low density threshold
                empty_positions.append(position)
            
            attempts += 1
        
        return empty_positions
    
    def _world_to_grid_coordinates(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        return tuple(int(pos / self.resolution) for pos in position)
    
    def _grid_to_world_coordinates(self, grid_position: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid indices to world coordinates."""
        return tuple(pos * self.resolution for pos in grid_position)
    
    def _is_valid_grid_position(self, grid_position: Tuple[int, int, int]) -> bool:
        """Check if grid position is within bounds."""
        return (0 <= grid_position[0] < self.grid_shape[0] and
                0 <= grid_position[1] < self.grid_shape[1] and
                0 <= grid_position[2] < self.grid_shape[2])
    
    def get_spatial_summary(self) -> Dict[str, Any]:
        """Get summary of spatial grid state."""
        return {
            "dimensions": self.dimensions,
            "resolution": self.resolution,
            "grid_shape": self.grid_shape,
            "total_cells": len(self.cell_positions),
            "occupied_positions": len(self.occupied_positions),
            "average_density": float(np.mean(self.spatial_grid)),
            "max_density": float(np.max(self.spatial_grid))
        }
