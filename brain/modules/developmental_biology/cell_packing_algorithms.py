"""
Cell Packing Algorithms

This module implements cell packing and spacing algorithms for
maintaining tissue architecture integrity.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType


class PackingAlgorithm(Enum):
    """Types of cell packing algorithms"""
    SPHERICAL_PACKING = "spherical_packing"
    HEXAGONAL_PACKING = "hexagonal_packing"
    RANDOM_PACKING = "random_packing"
    DENSITY_GRADIENT_PACKING = "density_gradient_packing"


@dataclass
class PackingParameters:
    """Parameters for cell packing algorithms"""
    target_density: float
    min_spacing: float
    max_spacing: float
    packing_efficiency: float
    zone_specific_density: Dict[ZoneType, float]


class CellPackingAlgorithms:
    """
    Implements cell packing and spacing algorithms for
    maintaining tissue architecture integrity.
    """
    
    def __init__(self):
        """Initialize cell packing algorithms"""
        self.packing_parameters: Dict[str, PackingParameters] = {}
        self._setup_packing_parameters()
    
    def _setup_packing_parameters(self) -> None:
        """Setup packing parameters for different zones"""
        self.packing_parameters = {
            "ventricular_zone": PackingParameters(
                target_density=0.8,
                min_spacing=0.1,
                max_spacing=0.3,
                packing_efficiency=0.9,
                zone_specific_density={
                    ZoneType.VENTRICULAR_ZONE: 0.8,
                    ZoneType.SUBVENTRICULAR_ZONE: 0.7,
                    ZoneType.INTERMEDIATE_ZONE: 0.6,
                    ZoneType.MANTLE_ZONE: 0.5
                }
            ),
            "subventricular_zone": PackingParameters(
                target_density=0.7,
                min_spacing=0.15,
                max_spacing=0.4,
                packing_efficiency=0.85,
                zone_specific_density={
                    ZoneType.VENTRICULAR_ZONE: 0.8,
                    ZoneType.SUBVENTRICULAR_ZONE: 0.7,
                    ZoneType.INTERMEDIATE_ZONE: 0.6,
                    ZoneType.MANTLE_ZONE: 0.5
                }
            ),
            "intermediate_zone": PackingParameters(
                target_density=0.6,
                min_spacing=0.2,
                max_spacing=0.5,
                packing_efficiency=0.8,
                zone_specific_density={
                    ZoneType.VENTRICULAR_ZONE: 0.8,
                    ZoneType.SUBVENTRICULAR_ZONE: 0.7,
                    ZoneType.INTERMEDIATE_ZONE: 0.6,
                    ZoneType.MANTLE_ZONE: 0.5
                }
            ),
            "mantle_zone": PackingParameters(
                target_density=0.5,
                min_spacing=0.25,
                max_spacing=0.6,
                packing_efficiency=0.75,
                zone_specific_density={
                    ZoneType.VENTRICULAR_ZONE: 0.8,
                    ZoneType.SUBVENTRICULAR_ZONE: 0.7,
                    ZoneType.INTERMEDIATE_ZONE: 0.6,
                    ZoneType.MANTLE_ZONE: 0.5
                }
            )
        }
    
    def pack_cells_spherical(self, cell_positions: List[Tuple[float, float, float]], 
                           zone_type: ZoneType, target_radius: float) -> List[Tuple[float, float, float]]:
        """Pack cells using spherical packing algorithm"""
        zone_name = zone_type.value
        params = self.packing_parameters.get(zone_name, 
                                           self.packing_parameters["ventricular_zone"])
        
        packed_positions = []
        
        for i, (x, y, z) in enumerate(cell_positions):
            # Calculate optimal position for spherical packing
            angle = 2 * np.pi * i / len(cell_positions)
            radius = target_radius * (1.0 - params.target_density * 0.1)
            
            new_x = x + radius * np.cos(angle) * params.packing_efficiency
            new_y = y + radius * np.sin(angle) * params.packing_efficiency
            new_z = z  # Maintain z-coordinate
            
            packed_positions.append((new_x, new_y, new_z))
        
        return packed_positions
    
    def pack_cells_hexagonal(self, cell_positions: List[Tuple[float, float, float]], 
                           zone_type: ZoneType) -> List[Tuple[float, float, float]]:
        """Pack cells using hexagonal packing algorithm"""
        zone_name = zone_type.value
        params = self.packing_parameters.get(zone_name, 
                                           self.packing_parameters["ventricular_zone"])
        
        packed_positions = []
        spacing = params.min_spacing + (params.max_spacing - params.min_spacing) * 0.5
        
        for i, (x, y, z) in enumerate(cell_positions):
            # Calculate hexagonal grid position
            row = i // 6  # Hexagonal rows
            col = i % 6   # Position within row
            
            # Hexagonal spacing
            hex_x = x + col * spacing
            hex_y = y + row * spacing * np.sqrt(3) / 2
            hex_z = z
            
            # Offset every other row
            if row % 2 == 1:
                hex_x += spacing / 2
            
            packed_positions.append((hex_x, hex_y, hex_z))
        
        return packed_positions
    
    def pack_cells_random(self, cell_positions: List[Tuple[float, float, float]], 
                         zone_type: ZoneType) -> List[Tuple[float, float, float]]:
        """Pack cells using random packing algorithm"""
        zone_name = zone_type.value
        params = self.packing_parameters.get(zone_name, 
                                           self.packing_parameters["ventricular_zone"])
        
        packed_positions = []
        
        for x, y, z in cell_positions:
            # Add random offset within spacing constraints
            random_x = x + np.random.uniform(-params.max_spacing/2, params.max_spacing/2)
            random_y = y + np.random.uniform(-params.max_spacing/2, params.max_spacing/2)
            random_z = z + np.random.uniform(-params.min_spacing/2, params.min_spacing/2)
            
            packed_positions.append((random_x, random_y, random_z))
        
        return packed_positions
    
    def pack_cells_density_gradient(self, cell_positions: List[Tuple[float, float, float]], 
                                  zone_type: ZoneType) -> List[Tuple[float, float, float]]:
        """Pack cells using density gradient packing algorithm"""
        zone_name = zone_type.value
        params = self.packing_parameters.get(zone_name, 
                                           self.packing_parameters["ventricular_zone"])
        
        packed_positions = []
        
        for i, (x, y, z) in enumerate(cell_positions):
            # Calculate density gradient based on position
            density_factor = params.zone_specific_density.get(zone_type, 0.5)
            
            # Adjust spacing based on density
            spacing = params.min_spacing + (params.max_spacing - params.min_spacing) * (1.0 - density_factor)
            
            # Apply density-based positioning
            density_x = x + np.random.uniform(-spacing/2, spacing/2)
            density_y = y + np.random.uniform(-spacing/2, spacing/2)
            density_z = z + np.random.uniform(-spacing/4, spacing/4)
            
            packed_positions.append((density_x, density_y, density_z))
        
        return packed_positions
    
    def calculate_packing_efficiency(self, cell_positions: List[Tuple[float, float, float]], 
                                   zone_type: ZoneType) -> float:
        """Calculate packing efficiency for cell positions"""
        if len(cell_positions) < 2:
            return 1.0
        
        zone_name = zone_type.value
        params = self.packing_parameters.get(zone_name, 
                                           self.packing_parameters["ventricular_zone"])
        
        # Calculate average spacing
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(cell_positions)):
            for j in range(i + 1, len(cell_positions)):
                x1, y1, z1 = cell_positions[i]
                x2, y2, z2 = cell_positions[j]
                
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 1.0
        
        average_spacing = total_distance / pair_count
        
        # Calculate efficiency based on target spacing
        target_spacing = (params.min_spacing + params.max_spacing) / 2
        efficiency = 1.0 - abs(average_spacing - target_spacing) / target_spacing
        
        return max(0.0, min(1.0, efficiency))
    
    def optimize_packing(self, cell_positions: List[Tuple[float, float, float]], 
                        zone_type: ZoneType, algorithm: PackingAlgorithm) -> List[Tuple[float, float, float]]:
        """Optimize cell packing using specified algorithm"""
        if algorithm == PackingAlgorithm.SPHERICAL_PACKING:
            return self.pack_cells_spherical(cell_positions, zone_type, 1.0)
        elif algorithm == PackingAlgorithm.HEXAGONAL_PACKING:
            return self.pack_cells_hexagonal(cell_positions, zone_type)
        elif algorithm == PackingAlgorithm.RANDOM_PACKING:
            return self.pack_cells_random(cell_positions, zone_type)
        elif algorithm == PackingAlgorithm.DENSITY_GRADIENT_PACKING:
            return self.pack_cells_density_gradient(cell_positions, zone_type)
        else:
            return cell_positions  # Return original if unknown algorithm
    
    def get_packing_statistics(self, cell_positions: List[Tuple[float, float, float]], 
                             zone_type: ZoneType) -> Dict[str, float]:
        """Get packing statistics for cell positions"""
        if not cell_positions:
            return {}
        
        zone_name = zone_type.value
        params = self.packing_parameters.get(zone_name, 
                                           self.packing_parameters["ventricular_zone"])
        
        # Calculate statistics
        efficiency = self.calculate_packing_efficiency(cell_positions, zone_type)
        
        # Calculate density
        if len(cell_positions) > 0:
            # Estimate volume (simplified)
            x_coords = [pos[0] for pos in cell_positions]
            y_coords = [pos[1] for pos in cell_positions]
            z_coords = [pos[2] for pos in cell_positions]
            
            volume = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords)) * (max(z_coords) - min(z_coords))
            if volume > 0:
                density = len(cell_positions) / volume
            else:
                density = 0.0
        else:
            density = 0.0
        
        return {
            "total_cells": len(cell_positions),
            "packing_efficiency": efficiency,
            "current_density": density,
            "target_density": params.target_density,
            "density_ratio": density / params.target_density if params.target_density > 0 else 0.0,
            "min_spacing": params.min_spacing,
            "max_spacing": params.max_spacing
        }
