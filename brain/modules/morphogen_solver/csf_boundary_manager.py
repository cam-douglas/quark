#!/usr/bin/env python3
"""CSF Flow Boundary Condition Manager.

Manages boundary conditions for CSF flow including production sites,
absorption sites, and no-flux boundaries for ventricular system modeling.

Integration: Boundary condition component for CSF flow dynamics
Rationale: Focused boundary condition management separated from main flow solver
"""

from typing import List, Tuple
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import SpatialGrid
from .ventricular_topology import VentricularTopology
from .ventricular_types import VentricleType
from .csf_flow_types import FlowBoundaryCondition, FlowBoundaryType, FlowParameters

logger = logging.getLogger(__name__)

class CSFBoundaryManager:
    """Manager for CSF flow boundary conditions.
    
    Handles identification and setup of CSF production sites (choroid plexus),
    absorption sites (arachnoid granulations), and no-flux boundaries
    (cavity walls) for embryonic ventricular system.
    """
    
    def __init__(self, spatial_grid: SpatialGrid,
                 ventricular_topology: VentricularTopology,
                 flow_params: FlowParameters):
        """Initialize CSF boundary manager.
        
        Args:
            spatial_grid: 3D spatial grid for boundary placement
            ventricular_topology: Ventricular topology system
            flow_params: Flow dynamics parameters
        """
        self.grid = spatial_grid
        self.topology = ventricular_topology
        self.params = flow_params
        
        logger.info("Initialized CSFBoundaryManager")
    
    def setup_boundary_conditions(self, cavity_mask: np.ndarray) -> List[FlowBoundaryCondition]:
        """Setup boundary conditions for CSF flow.
        
        Args:
            cavity_mask: Boolean mask of ventricular cavities
            
        Returns:
            List of boundary conditions
        """
        boundary_conditions = []
        
        # CSF production sites (choroid plexus locations)
        production_sites = self._identify_production_sites()
        for site in production_sites:
            bc = FlowBoundaryCondition(
                boundary_type=FlowBoundaryType.PRODUCTION,
                location=site,
                value=self.params.production_rate_ml_min / 60.0 / 1e6,  # m³/s
                region_mask=None
            )
            boundary_conditions.append(bc)
        
        # CSF absorption sites (arachnoid granulations - simplified)
        absorption_sites = self._identify_absorption_sites()
        for site in absorption_sites:
            bc = FlowBoundaryCondition(
                boundary_type=FlowBoundaryType.ABSORPTION,
                location=site,
                value=-self.params.absorption_rate_ml_min / 60.0 / 1e6,  # m³/s
                region_mask=None
            )
            boundary_conditions.append(bc)
        
        # No-flux boundaries (cavity walls)
        wall_mask = self._identify_cavity_walls(cavity_mask)
        wall_bc = FlowBoundaryCondition(
            boundary_type=FlowBoundaryType.NO_FLUX,
            location=(0, 0, 0),  # Not used for region boundaries
            value=0.0,
            region_mask=wall_mask
        )
        boundary_conditions.append(wall_bc)
        
        logger.info(f"Setup {len(production_sites)} production sites, "
                   f"{len(absorption_sites)} absorption sites")
        
        return boundary_conditions
    
    def _identify_production_sites(self) -> List[Tuple[int, int, int]]:
        """Identify CSF production sites (choroid plexus locations)."""
        production_sites = []
        
        # Lateral ventricles - choroid plexus locations
        for ventricle_type in [VentricleType.LATERAL_LEFT, VentricleType.LATERAL_RIGHT]:
            if ventricle_type in self.topology.ventricular_regions:
                region = self.topology.ventricular_regions[ventricle_type]
                
                # Place production site at medial wall of lateral ventricle
                center = region.center_position
                dims = self.grid.dimensions
                
                production_x = int(center[0] / dims.resolution)
                production_y = int((center[1] - region.dimensions[1]/4) / dims.resolution)
                production_z = int(center[2] / dims.resolution)
                
                # Ensure within grid bounds
                production_x = np.clip(production_x, 0, dims.x_size - 1)
                production_y = np.clip(production_y, 0, dims.y_size - 1)
                production_z = np.clip(production_z, 0, dims.z_size - 1)
                
                production_sites.append((production_x, production_y, production_z))
        
        # Fourth ventricle - choroid plexus
        if VentricleType.FOURTH in self.topology.ventricular_regions:
            region = self.topology.ventricular_regions[VentricleType.FOURTH]
            center = region.center_position
            dims = self.grid.dimensions
            
            production_x = int(center[0] / dims.resolution)
            production_y = int((center[1] + region.dimensions[1]/4) / dims.resolution)
            production_z = int(center[2] / dims.resolution)
            
            production_x = np.clip(production_x, 0, dims.x_size - 1)
            production_y = np.clip(production_y, 0, dims.y_size - 1)
            production_z = np.clip(production_z, 0, dims.z_size - 1)
            
            production_sites.append((production_x, production_y, production_z))
        
        return production_sites
    
    def _identify_absorption_sites(self) -> List[Tuple[int, int, int]]:
        """Identify CSF absorption sites (simplified arachnoid granulations)."""
        # In embryonic development, absorption is limited
        # Place simplified absorption site at fourth ventricle exit
        absorption_sites = []
        
        if VentricleType.FOURTH in self.topology.ventricular_regions:
            region = self.topology.ventricular_regions[VentricleType.FOURTH]
            center = region.center_position
            dims = self.grid.dimensions
            
            # Place at posterior end of fourth ventricle
            absorption_x = int(center[0] / dims.resolution)
            absorption_y = int((center[1] - region.dimensions[1]/2) / dims.resolution)
            absorption_z = int(center[2] / dims.resolution)
            
            absorption_x = np.clip(absorption_x, 0, dims.x_size - 1)
            absorption_y = np.clip(absorption_y, 0, dims.y_size - 1)
            absorption_z = np.clip(absorption_z, 0, dims.z_size - 1)
            
            absorption_sites.append((absorption_x, absorption_y, absorption_z))
        
        return absorption_sites
    
    def _identify_cavity_walls(self, cavity_mask: np.ndarray) -> np.ndarray:
        """Identify cavity wall boundaries for no-flux conditions."""
        # Find boundary voxels (cavity voxels adjacent to non-cavity voxels)
        dilated_cavity = ndimage.binary_dilation(cavity_mask)
        wall_mask = dilated_cavity & ~cavity_mask
        
        return wall_mask
