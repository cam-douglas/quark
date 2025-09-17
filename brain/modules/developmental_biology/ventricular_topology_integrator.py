"""
Ventricular Topology Integrator

This module integrates ventricular zone organization with the completed
ventricular topology system from the foundation layer.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .ventricular_zone_organizer import ZoneType, SpatialPosition, CellArchitecture


class CavityType(Enum):
    """Types of ventricular cavities"""
    LATERAL_VENTRICLE = "lateral_ventricle"
    THIRD_VENTRICLE = "third_ventricle"
    FOURTH_VENTRICLE = "fourth_ventricle"
    CEREBRAL_AQUEDUCT = "cerebral_aqueduct"
    CENTRAL_CANAL = "central_canal"


@dataclass
class VentricularCavity:
    """Ventricular cavity information"""
    cavity_id: str
    cavity_type: CavityType
    position: Tuple[float, float, float]
    radius: float
    surface_area: float
    volume: float
    connected_cavities: List[str]


@dataclass
class CellCavityRelationship:
    """Relationship between cell and ventricular cavity"""
    cell_id: str
    cavity_id: str
    distance_to_cavity: float
    cavity_influence: float  # 0.0 to 1.0
    surface_contact: bool
    fluid_exposure: float  # 0.0 to 1.0


class VentricularTopologyIntegrator:
    """
    Integrates ventricular zone organization with ventricular topology
    from the foundation layer system.
    """
    
    def __init__(self):
        """Initialize ventricular topology integrator"""
        self.ventricular_cavities: Dict[str, VentricularCavity] = {}
        self.cavity_parameters: Dict[CavityType, Dict[str, float]] = {}
        self.integration_parameters: Dict[str, float] = {}
        self._setup_cavity_parameters()
        self._setup_integration_parameters()
        self._initialize_ventricular_cavities()
    
    def _setup_cavity_parameters(self) -> None:
        """Setup parameters for different cavity types"""
        self.cavity_parameters = {
            CavityType.LATERAL_VENTRICLE: {
                "radius": 2.0,
                "surface_area": 50.0,
                "volume": 100.0,
                "influence_radius": 5.0
            },
            CavityType.THIRD_VENTRICLE: {
                "radius": 1.5,
                "surface_area": 30.0,
                "volume": 50.0,
                "influence_radius": 4.0
            },
            CavityType.FOURTH_VENTRICLE: {
                "radius": 1.0,
                "surface_area": 20.0,
                "volume": 30.0,
                "influence_radius": 3.0
            },
            CavityType.CEREBRAL_AQUEDUCT: {
                "radius": 0.5,
                "surface_area": 10.0,
                "volume": 15.0,
                "influence_radius": 2.0
            },
            CavityType.CENTRAL_CANAL: {
                "radius": 0.3,
                "surface_area": 5.0,
                "volume": 8.0,
                "influence_radius": 1.5
            }
        }
    
    def _setup_integration_parameters(self) -> None:
        """Setup integration parameters"""
        self.integration_parameters = {
            "max_influence_distance": 10.0,  # Maximum distance for cavity influence
            "surface_contact_threshold": 0.1,  # Distance threshold for surface contact
            "fluid_exposure_factor": 0.8,  # Factor for fluid exposure calculation
            "cavity_pressure_influence": 0.6,  # Influence of cavity pressure on cells
            "csf_flow_influence": 0.4  # Influence of CSF flow on cells
        }
    
    def _initialize_ventricular_cavities(self) -> None:
        """Initialize ventricular cavities based on foundation layer"""
        # This would integrate with the completed ventricular topology system
        # For now, create representative cavities
        
        cavities = [
            ("lv_1", CavityType.LATERAL_VENTRICLE, (0.0, 0.0, 0.0)),
            ("lv_2", CavityType.LATERAL_VENTRICLE, (0.0, 0.0, 2.0)),
            ("tv_1", CavityType.THIRD_VENTRICLE, (0.0, 0.0, 1.0)),
            ("fv_1", CavityType.FOURTH_VENTRICLE, (0.0, 0.0, 3.0)),
            ("ca_1", CavityType.CEREBRAL_AQUEDUCT, (0.0, 0.0, 2.5)),
            ("cc_1", CavityType.CENTRAL_CANAL, (0.0, 0.0, 4.0))
        ]
        
        for cavity_id, cavity_type, position in cavities:
            params = self.cavity_parameters[cavity_type]
            
            cavity = VentricularCavity(
                cavity_id=cavity_id,
                cavity_type=cavity_type,
                position=position,
                radius=params["radius"],
                surface_area=params["surface_area"],
                volume=params["volume"],
                connected_cavities=[]
            )
            
            self.ventricular_cavities[cavity_id] = cavity
        
        # Set up cavity connections
        self._setup_cavity_connections()
    
    def _setup_cavity_connections(self) -> None:
        """Setup connections between cavities"""
        # Lateral ventricles connect to third ventricle
        self.ventricular_cavities["lv_1"].connected_cavities = ["tv_1"]
        self.ventricular_cavities["lv_2"].connected_cavities = ["tv_1"]
        
        # Third ventricle connects to cerebral aqueduct
        self.ventricular_cavities["tv_1"].connected_cavities = ["ca_1"]
        
        # Cerebral aqueduct connects to fourth ventricle
        self.ventricular_cavities["ca_1"].connected_cavities = ["fv_1"]
        
        # Fourth ventricle connects to central canal
        self.ventricular_cavities["fv_1"].connected_cavities = ["cc_1"]
    
    def integrate_cell_with_cavity(self, cell_architecture: CellArchitecture) -> CellCavityRelationship:
        """
        Integrate cell with ventricular cavity system
        
        Args:
            cell_architecture: Cell architecture information
            
        Returns:
            CellCavityRelationship with cavity integration details
        """
        # Find closest cavity
        closest_cavity_id, distance = self._find_closest_cavity(cell_architecture.position)
        
        # Calculate cavity influence
        cavity_influence = self._calculate_cavity_influence(
            distance, self.ventricular_cavities[closest_cavity_id]
        )
        
        # Determine surface contact
        surface_contact = distance < self.integration_parameters["surface_contact_threshold"]
        
        # Calculate fluid exposure
        fluid_exposure = self._calculate_fluid_exposure(
            distance, cavity_influence, surface_contact
        )
        
        return CellCavityRelationship(
            cell_id=cell_architecture.cell_id,
            cavity_id=closest_cavity_id,
            distance_to_cavity=distance,
            cavity_influence=cavity_influence,
            surface_contact=surface_contact,
            fluid_exposure=fluid_exposure
        )
    
    def _find_closest_cavity(self, position: SpatialPosition) -> Tuple[str, float]:
        """Find closest ventricular cavity to cell position"""
        min_distance = float('inf')
        closest_cavity_id = None
        
        for cavity_id, cavity in self.ventricular_cavities.items():
            # Calculate 3D distance
            dx = position.x - cavity.position[0]
            dy = position.y - cavity.position[1]
            dz = position.z - cavity.position[2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance < min_distance:
                min_distance = distance
                closest_cavity_id = cavity_id
        
        return closest_cavity_id, min_distance
    
    def _calculate_cavity_influence(self, distance: float, cavity: VentricularCavity) -> float:
        """Calculate influence of cavity on cell"""
        cavity_type = cavity.cavity_type
        influence_radius = self.cavity_parameters[cavity_type]["influence_radius"]
        
        if distance > influence_radius:
            return 0.0
        
        # Exponential decay of influence with distance
        influence = np.exp(-distance / (influence_radius * 0.5))
        
        # Scale by cavity size
        size_factor = cavity.radius / 2.0  # Normalize by typical radius
        
        return min(1.0, influence * size_factor)
    
    def _calculate_fluid_exposure(self, distance: float, cavity_influence: float,
                                surface_contact: bool) -> float:
        """Calculate fluid exposure level"""
        if surface_contact:
            return 1.0
        
        # Base exposure from cavity influence
        base_exposure = cavity_influence * self.integration_parameters["fluid_exposure_factor"]
        
        # Distance-based exposure
        max_distance = self.integration_parameters["max_influence_distance"]
        distance_factor = max(0.0, 1.0 - distance / max_distance)
        
        fluid_exposure = base_exposure * distance_factor
        
        return min(1.0, fluid_exposure)
    
    def integrate_ventricular_zone_with_topology(self, cell_architectures: Dict[str, CellArchitecture]) -> Dict[str, CellCavityRelationship]:
        """
        Integrate entire ventricular zone with ventricular topology
        
        Args:
            cell_architectures: Dictionary of cell architectures
            
        Returns:
            Dictionary mapping cell IDs to cavity relationships
        """
        cell_cavity_relationships = {}
        
        for cell_id, architecture in cell_architectures.items():
            relationship = self.integrate_cell_with_cavity(architecture)
            cell_cavity_relationships[cell_id] = relationship
        
        return cell_cavity_relationships
    
    def get_cavity_statistics(self, cell_cavity_relationships: Dict[str, CellCavityRelationship]) -> Dict[str, float]:
        """Get statistics about cavity integration"""
        if not cell_cavity_relationships:
            return {}
        
        cavity_counts = {}
        surface_contact_count = 0
        total_influence = 0.0
        total_fluid_exposure = 0.0
        
        for relationship in cell_cavity_relationships.values():
            # Count cells per cavity
            cavity_id = relationship.cavity_id
            cavity_counts[cavity_id] = cavity_counts.get(cavity_id, 0) + 1
            
            # Count surface contacts
            if relationship.surface_contact:
                surface_contact_count += 1
            
            # Sum influences and exposures
            total_influence += relationship.cavity_influence
            total_fluid_exposure += relationship.fluid_exposure
        
        total_cells = len(cell_cavity_relationships)
        
        return {
            "total_cells": total_cells,
            "cavity_distribution": {cavity: count/total_cells for cavity, count in cavity_counts.items()},
            "surface_contact_ratio": surface_contact_count / total_cells,
            "average_cavity_influence": total_influence / total_cells,
            "average_fluid_exposure": total_fluid_exposure / total_cells,
            "total_cavities": len(self.ventricular_cavities)
        }
    
    def update_cavity_pressure(self, cavity_id: str, pressure: float) -> None:
        """Update pressure in specific cavity"""
        if cavity_id in self.ventricular_cavities:
            # This would integrate with the CSF flow dynamics system
            # For now, just store the pressure information
            pass
    
    def get_cavity_flow_information(self, cavity_id: str) -> Dict[str, float]:
        """Get flow information for specific cavity"""
        if cavity_id not in self.ventricular_cavities:
            return {}
        
        cavity = self.ventricular_cavities[cavity_id]
        
        # This would integrate with the CSF flow dynamics system
        # For now, return placeholder values
        return {
            "flow_rate": 1.0,
            "pressure": 1.0,
            "viscosity": 1.0,
            "temperature": 37.0
        }
