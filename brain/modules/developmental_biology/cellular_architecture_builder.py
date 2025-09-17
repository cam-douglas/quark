"""
Cellular Architecture Builder

This module builds cellular architecture for cells in the ventricular zone,
including surface areas, dimensions, and neighbor connections.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType, SpatialPosition, PolarityState


@dataclass
class CellArchitecture:
    """Cellular architecture information"""
    cell_id: str
    position: SpatialPosition
    polarity_state: PolarityState
    apical_surface_area: float
    basal_surface_area: float
    cell_height: float
    cell_width: float
    neighbor_connections: List[str]


class CellularArchitectureBuilder:
    """
    Builds cellular architecture for cells in the ventricular zone
    including surface areas, dimensions, and neighbor connections.
    """
    
    def __init__(self):
        """Initialize cellular architecture builder"""
        self.polarity_parameters: Dict[str, float] = {}
        self.spatial_parameters: Dict[str, float] = {}
        self._setup_polarity_parameters()
        self._setup_spatial_parameters()
    
    def _setup_polarity_parameters(self) -> None:
        """Setup polarity parameters"""
        self.polarity_parameters = {
            "apical_threshold": 0.1,  # Within 10% of apical surface
            "basal_threshold": 0.1,   # Within 10% of basal surface
            "polarization_threshold": 0.7,  # 70% of cells should be polarized
            "apical_surface_ratio": 0.8,  # 80% of surface area at apical end
            "basal_surface_ratio": 0.2   # 20% of surface area at basal end
        }
    
    def _setup_spatial_parameters(self) -> None:
        """Setup spatial organization parameters"""
        self.spatial_parameters = {
            "cell_density_vz": 0.8,  # High density in VZ
            "cell_density_svz": 0.6,  # Medium density in SVZ
            "cell_density_iz": 0.4,  # Lower density in IZ
            "cell_density_mz": 0.3,  # Lowest density in MZ
            "apical_basal_ratio": 0.7,  # 70% apical, 30% basal
            "neighbor_connectivity": 6  # Average number of neighbors
        }
    
    def create_cellular_architecture(self, cell_id: str, position: SpatialPosition,
                                   polarity_state: PolarityState) -> CellArchitecture:
        """
        Create cellular architecture for cell
        
        Args:
            cell_id: Unique cell identifier
            position: Spatial position
            polarity_state: Polarity state
            
        Returns:
            CellArchitecture with cell structure information
        """
        # Calculate surface areas based on polarity
        apical_surface_area, basal_surface_area = self._calculate_surface_areas(
            polarity_state, position.zone_type
        )
        
        # Calculate cell dimensions
        cell_height, cell_width = self._calculate_cell_dimensions(
            position.zone_type, polarity_state
        )
        
        # Generate neighbor connections (simplified)
        neighbor_connections = self._generate_neighbor_connections(cell_id, position)
        
        return CellArchitecture(
            cell_id=cell_id,
            position=position,
            polarity_state=polarity_state,
            apical_surface_area=apical_surface_area,
            basal_surface_area=basal_surface_area,
            cell_height=cell_height,
            cell_width=cell_width,
            neighbor_connections=neighbor_connections
        )
    
    def _calculate_surface_areas(self, polarity_state: PolarityState,
                               zone_type: ZoneType) -> Tuple[float, float]:
        """Calculate apical and basal surface areas"""
        # Base surface area
        base_area = 1.0
        
        # Adjust based on polarity
        if polarity_state == PolarityState.APICAL:
            apical_ratio = self.polarity_parameters["apical_surface_ratio"]
            basal_ratio = 1.0 - apical_ratio
        elif polarity_state == PolarityState.BASAL:
            basal_ratio = self.polarity_parameters["basal_surface_ratio"]
            apical_ratio = 1.0 - basal_ratio
        else:  # INTERMEDIATE or POLARIZED
            apical_ratio = 0.5
            basal_ratio = 0.5
        
        # Adjust based on zone type
        zone_factor = self._get_zone_surface_factor(zone_type)
        
        apical_area = base_area * apical_ratio * zone_factor
        basal_area = base_area * basal_ratio * zone_factor
        
        return apical_area, basal_area
    
    def _get_zone_surface_factor(self, zone_type: ZoneType) -> float:
        """Get surface area factor for zone type"""
        factors = {
            ZoneType.VENTRICULAR_ZONE: 1.2,  # Larger surface area
            ZoneType.SUBVENTRICULAR_ZONE: 1.0,  # Normal surface area
            ZoneType.INTERMEDIATE_ZONE: 0.8,  # Smaller surface area
            ZoneType.MANTLE_ZONE: 0.6  # Smallest surface area
        }
        return factors.get(zone_type, 1.0)
    
    def _calculate_cell_dimensions(self, zone_type: ZoneType,
                                 polarity_state: PolarityState) -> Tuple[float, float]:
        """Calculate cell height and width"""
        # Base dimensions
        base_height = 1.0
        base_width = 1.0
        
        # Adjust height based on zone type
        height_factors = {
            ZoneType.VENTRICULAR_ZONE: 1.5,  # Taller cells
            ZoneType.SUBVENTRICULAR_ZONE: 1.2,
            ZoneType.INTERMEDIATE_ZONE: 1.0,
            ZoneType.MANTLE_ZONE: 0.8  # Shorter cells
        }
        
        # Adjust width based on polarity
        if polarity_state == PolarityState.POLARIZED:
            width_factor = 0.8  # Narrower when polarized
        else:
            width_factor = 1.0
        
        height = base_height * height_factors.get(zone_type, 1.0)
        width = base_width * width_factor
        
        return height, width
    
    def _generate_neighbor_connections(self, cell_id: str,
                                     position: SpatialPosition) -> List[str]:
        """Generate neighbor connections for cell"""
        # Simplified neighbor generation
        # In reality, this would be based on spatial proximity
        
        num_neighbors = self.spatial_parameters["neighbor_connectivity"]
        neighbors = []
        
        for i in range(num_neighbors):
            neighbor_id = f"{cell_id}_neighbor_{i}"
            neighbors.append(neighbor_id)
        
        return neighbors
    
    def get_architecture_statistics(self, cell_architectures: List[CellArchitecture]) -> Dict[str, float]:
        """Get statistics about cellular architecture"""
        if not cell_architectures:
            return {}
        
        # Calculate statistics
        apical_areas = [arch.apical_surface_area for arch in cell_architectures]
        basal_areas = [arch.basal_surface_area for arch in cell_architectures]
        heights = [arch.cell_height for arch in cell_architectures]
        widths = [arch.cell_width for arch in cell_architectures]
        
        return {
            "total_cells": len(cell_architectures),
            "average_apical_area": np.mean(apical_areas),
            "average_basal_area": np.mean(basal_areas),
            "average_height": np.mean(heights),
            "average_width": np.mean(widths),
            "total_surface_area": sum(apical_areas) + sum(basal_areas),
            "average_neighbors": np.mean([len(arch.neighbor_connections) for arch in cell_architectures])
        }
