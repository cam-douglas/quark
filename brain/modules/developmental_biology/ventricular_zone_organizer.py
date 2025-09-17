"""
Ventricular Zone Organizer

This module organizes neuroepithelial cells in ventricular and subventricular zones
with proper spatial structure and apical-basal polarity.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ZoneType(Enum):
    """Types of ventricular zones"""
    VENTRICULAR_ZONE = "ventricular_zone"  # VZ
    SUBVENTRICULAR_ZONE = "subventricular_zone"  # SVZ
    INTERMEDIATE_ZONE = "intermediate_zone"  # IZ
    MANTLE_ZONE = "mantle_zone"  # MZ


class PolarityState(Enum):
    """Apical-basal polarity states"""
    APICAL = "apical"
    BASAL = "basal"
    INTERMEDIATE = "intermediate"
    POLARIZED = "polarized"


@dataclass
class SpatialPosition:
    """Spatial position in ventricular zone"""
    x: float
    y: float
    z: float
    zone_type: ZoneType
    apical_distance: float  # Distance from apical surface
    basal_distance: float  # Distance from basal surface
    radial_position: float  # 0.0 (apical) to 1.0 (basal)


# Import CellArchitecture and CellularArchitectureBuilder from cellular_architecture_builder
from .cellular_architecture_builder import CellArchitecture, CellularArchitectureBuilder


class VentricularZoneOrganizer:
    """
    Organizes neuroepithelial cells in ventricular and subventricular zones
    with proper spatial structure and apical-basal polarity.
    """
    
    def __init__(self):
        """Initialize ventricular zone organizer"""
        self.zone_boundaries: Dict[ZoneType, Dict[str, float]] = {}
        self.architecture_builder = CellularArchitectureBuilder()
        self._setup_zone_boundaries()
    
    def _setup_zone_boundaries(self) -> None:
        """Setup boundaries for different zones"""
        # Zone boundaries as fractions of total neural tube thickness
        self.zone_boundaries = {
            ZoneType.VENTRICULAR_ZONE: {
                "apical_boundary": 0.0,  # At ventricular surface
                "basal_boundary": 0.3,  # 30% of thickness
                "thickness": 0.3
            },
            ZoneType.SUBVENTRICULAR_ZONE: {
                "apical_boundary": 0.3,  # Below VZ
                "basal_boundary": 0.6,  # 60% of thickness
                "thickness": 0.3
            },
            ZoneType.INTERMEDIATE_ZONE: {
                "apical_boundary": 0.6,  # Below SVZ
                "basal_boundary": 0.8,  # 80% of thickness
                "thickness": 0.2
            },
            ZoneType.MANTLE_ZONE: {
                "apical_boundary": 0.8,  # Below IZ
                "basal_boundary": 1.0,  # At pial surface
                "thickness": 0.2
            }
        }
    
    
    def assign_cell_to_zone(self, cell_id: str, radial_position: float,
                          developmental_stage: str = "mid_embryonic") -> SpatialPosition:
        """
        Assign cell to appropriate zone based on radial position
        
        Args:
            cell_id: Unique cell identifier
            radial_position: Radial position (0.0 = apical, 1.0 = basal)
            developmental_stage: Current developmental stage
            
        Returns:
            SpatialPosition with zone assignment
        """
        # Determine zone based on radial position
        zone_type = self._determine_zone_type(radial_position)
        
        # Calculate distances from boundaries
        apical_distance = radial_position
        basal_distance = 1.0 - radial_position
        
        # Generate spatial coordinates (simplified)
        x, y, z = self._generate_spatial_coordinates(radial_position, zone_type)
        
        return SpatialPosition(
            x=x,
            y=y,
            z=z,
            zone_type=zone_type,
            apical_distance=apical_distance,
            basal_distance=basal_distance,
            radial_position=radial_position
        )
    
    def _determine_zone_type(self, radial_position: float) -> ZoneType:
        """Determine zone type based on radial position"""
        for zone_type, boundaries in self.zone_boundaries.items():
            if (boundaries["apical_boundary"] <= radial_position < 
                boundaries["basal_boundary"]):
                return zone_type
        
        # Default to ventricular zone if position is at boundary
        return ZoneType.VENTRICULAR_ZONE
    
    def _generate_spatial_coordinates(self, radial_position: float,
                                    zone_type: ZoneType) -> Tuple[float, float, float]:
        """Generate 3D spatial coordinates for cell position"""
        # Simplified coordinate generation
        # In reality, this would be more complex based on neural tube geometry
        
        # X and Y coordinates (tangential to neural tube)
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        
        # Z coordinate (radial position)
        z = radial_position
        
        return x, y, z
    
    def establish_apical_basal_polarity(self, cell_id: str, position: SpatialPosition,
                                      developmental_stage: str = "mid_embryonic") -> PolarityState:
        """
        Establish apical-basal polarity for cell
        
        Args:
            cell_id: Unique cell identifier
            position: Spatial position of cell
            developmental_stage: Current developmental stage
            
        Returns:
            PolarityState indicating polarity
        """
        # Determine polarity based on position and developmental stage
        apical_threshold = 0.1  # Within 10% of apical surface
        basal_threshold = 0.1   # Within 10% of basal surface
        
        if position.apical_distance < apical_threshold:
            return PolarityState.APICAL
        elif position.basal_distance < basal_threshold:
            return PolarityState.BASAL
        elif self._is_polarized(position, developmental_stage):
            return PolarityState.POLARIZED
        else:
            return PolarityState.INTERMEDIATE
    
    def _is_polarized(self, position: SpatialPosition, developmental_stage: str) -> bool:
        """Check if cell is properly polarized"""
        # Cells in VZ and SVZ are more likely to be polarized
        if position.zone_type in [ZoneType.VENTRICULAR_ZONE, ZoneType.SUBVENTRICULAR_ZONE]:
            return True
        
        # Later developmental stages have more polarized cells
        if developmental_stage in ["late_embryonic", "fetal"]:
            return position.radial_position < 0.5
        
        return False
    
    def create_cellular_architecture(self, cell_id: str, position: SpatialPosition,
                                   polarity_state: PolarityState) -> CellArchitecture:
        """Create cellular architecture for cell using architecture builder"""
        return self.architecture_builder.create_cellular_architecture(
            cell_id, position, polarity_state
        )
    
    
    def organize_ventricular_zone(self, cell_ids: List[str],
                                developmental_stage: str = "mid_embryonic") -> Dict[str, CellArchitecture]:
        """
        Organize entire ventricular zone with cells
        
        Args:
            cell_ids: List of cell identifiers
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary mapping cell IDs to their architecture
        """
        cell_architectures = {}
        
        for cell_id in cell_ids:
            # Assign random radial position (in real implementation, this would be more sophisticated)
            radial_position = np.random.uniform(0.0, 1.0)
            
            # Assign cell to zone
            position = self.assign_cell_to_zone(cell_id, radial_position, developmental_stage)
            
            # Establish polarity
            polarity_state = self.establish_apical_basal_polarity(
                cell_id, position, developmental_stage
            )
            
            # Create cellular architecture
            architecture = self.create_cellular_architecture(
                cell_id, position, polarity_state
            )
            
            cell_architectures[cell_id] = architecture
        
        return cell_architectures
    
    def get_zone_statistics(self, cell_architectures: Dict[str, CellArchitecture]) -> Dict[str, float]:
        """Get statistics about zone organization"""
        if not cell_architectures:
            return {}
        
        zone_counts = {}
        polarity_counts = {}
        
        for architecture in cell_architectures.values():
            # Count zones
            zone_type = architecture.position.zone_type.value
            zone_counts[zone_type] = zone_counts.get(zone_type, 0) + 1
            
            # Count polarities
            polarity = architecture.polarity_state.value
            polarity_counts[polarity] = polarity_counts.get(polarity, 0) + 1
        
        total_cells = len(cell_architectures)
        
        return {
            "total_cells": total_cells,
            "zone_distribution": {zone: count/total_cells for zone, count in zone_counts.items()},
            "polarity_distribution": {polarity: count/total_cells for polarity, count in polarity_counts.items()},
            "average_apical_distance": np.mean([arch.position.apical_distance for arch in cell_architectures.values()]),
            "average_basal_distance": np.mean([arch.position.basal_distance for arch in cell_architectures.values()])
        }
