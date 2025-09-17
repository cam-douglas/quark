#!/usr/bin/env python3
"""Ventricular Cavity Topology System.

Designs and manages ventricular cavity topology for embryonic brain development,
including lateral ventricles, third ventricle, fourth ventricle, and cerebral
aqueduct connectivity for proper CSF flow modeling.

Integration: Spatial structure component of foundation layer
Rationale: Main coordinator for ventricular topology with focused responsibilities
"""

from typing import Dict, Any, Optional, List
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import SpatialGrid
from .ventricular_types import VentricleType, VentricularRegion
from .ventricular_geometry import VentricularGeometryGenerator

logger = logging.getLogger(__name__)

class VentricularTopology:
    """Ventricular cavity topology system for embryonic brain development.
    
    Manages the design and construction of ventricular cavities including
    lateral ventricles, third ventricle, fourth ventricle, and cerebral
    aqueduct with proper connectivity validation.
    """
    
    def __init__(self, spatial_grid: SpatialGrid, 
                 developmental_stage: str = "E8.5-E10.5"):
        """Initialize ventricular topology system.
        
        Args:
            spatial_grid: 3D spatial grid for cavity mapping
            developmental_stage: Embryonic developmental stage
        """
        self.grid = spatial_grid
        self.developmental_stage = developmental_stage
        
        # Initialize geometry generator
        self.geometry_generator = VentricularGeometryGenerator(spatial_grid.dimensions)
        
        # Ventricular regions and connectivity
        self.ventricular_regions: Dict[VentricleType, VentricularRegion] = {}
        self.cavity_mask: Optional[np.ndarray] = None
        self.csf_flow_pathways: List[List[VentricleType]] = []
        
        # Initialize cavity definitions
        self._initialize_ventricular_regions()
        
        logger.info("Initialized VentricularTopology system")
        logger.info(f"Developmental stage: {developmental_stage}")
        logger.info(f"Grid resolution: {spatial_grid.dimensions.resolution} µm")
    
    def _initialize_ventricular_regions(self) -> None:
        """Initialize ventricular cavity region definitions."""
        dims = self.grid.dimensions
        center_x, center_y, center_z = dims.x_size/2, dims.y_size/2, dims.z_size/2
        
        # Lateral ventricles (hemispheres)
        self.ventricular_regions[VentricleType.LATERAL_LEFT] = VentricularRegion(
            ventricle_type=VentricleType.LATERAL_LEFT,
            center_position=(center_x - 200, center_y + 100, center_z),
            dimensions=(150, 80, 200),
            shape_type="ellipsoid",
            connectivity=[VentricleType.THIRD],
            developmental_week=4.5
        )
        
        self.ventricular_regions[VentricleType.LATERAL_RIGHT] = VentricularRegion(
            ventricle_type=VentricleType.LATERAL_RIGHT,
            center_position=(center_x + 200, center_y + 100, center_z),
            dimensions=(150, 80, 200),
            shape_type="ellipsoid",
            connectivity=[VentricleType.THIRD],
            developmental_week=4.5
        )
        
        # Third ventricle (diencephalon)
        self.ventricular_regions[VentricleType.THIRD] = VentricularRegion(
            ventricle_type=VentricleType.THIRD,
            center_position=(center_x, center_y, center_z - 50),
            dimensions=(20, 60, 80),
            shape_type="ellipsoid",
            connectivity=[VentricleType.LATERAL_LEFT, VentricleType.LATERAL_RIGHT, 
                         VentricleType.CEREBRAL_AQUEDUCT],
            developmental_week=5.0
        )
        
        # Cerebral aqueduct (midbrain)
        self.ventricular_regions[VentricleType.CEREBRAL_AQUEDUCT] = VentricularRegion(
            ventricle_type=VentricleType.CEREBRAL_AQUEDUCT,
            center_position=(center_x, center_y - 50, center_z - 100),
            dimensions=(8, 8, 60),
            shape_type="tube",
            connectivity=[VentricleType.THIRD, VentricleType.FOURTH],
            developmental_week=5.5
        )
        
        # Fourth ventricle (rhombencephalon)
        self.ventricular_regions[VentricleType.FOURTH] = VentricularRegion(
            ventricle_type=VentricleType.FOURTH,
            center_position=(center_x, center_y - 100, center_z - 150),
            dimensions=(40, 30, 50),
            shape_type="irregular",
            connectivity=[VentricleType.CEREBRAL_AQUEDUCT],
            developmental_week=6.0
        )
        
        # Define CSF flow pathways
        self.csf_flow_pathways = [
            [VentricleType.LATERAL_LEFT, VentricleType.THIRD, 
             VentricleType.CEREBRAL_AQUEDUCT, VentricleType.FOURTH],
            [VentricleType.LATERAL_RIGHT, VentricleType.THIRD, 
             VentricleType.CEREBRAL_AQUEDUCT, VentricleType.FOURTH]
        ]
        
        logger.info(f"Initialized {len(self.ventricular_regions)} ventricular regions")
    
    def generate_cavity_topology(self) -> np.ndarray:
        """Generate 3D ventricular cavity topology mask.
        
        Returns:
            3D boolean array marking ventricular cavity voxels
        """
        dims = self.grid.dimensions
        cavity_mask = np.zeros((dims.x_size, dims.y_size, dims.z_size), dtype=bool)
        
        # Generate each ventricular cavity
        for ventricle_type, region in self.ventricular_regions.items():
            cavity_voxels = self.geometry_generator.generate_cavity_voxels(region)
            cavity_mask |= cavity_voxels
            
            logger.info(f"Generated {ventricle_type.value} cavity: "
                       f"{np.sum(cavity_voxels)} voxels")
        
        self.cavity_mask = cavity_mask
        
        # Validate connectivity
        self._validate_cavity_connectivity()
        
        total_volume = np.sum(cavity_mask) * (dims.resolution ** 3) / 1e9  # mm³
        logger.info(f"Total ventricular volume: {total_volume:.3f} mm³")
        
        return cavity_mask
    
    def _validate_cavity_connectivity(self) -> None:
        """Validate that ventricular cavities are properly connected."""
        if self.cavity_mask is None:
            return
        
        connectivity_issues = []
        
        for region_type, region in self.ventricular_regions.items():
            for connected_type in region.connectivity:
                if not self._check_cavity_connection(region_type, connected_type):
                    connectivity_issues.append(f"{region_type.value} -> {connected_type.value}")
        
        if connectivity_issues:
            logger.warning(f"Connectivity issues detected: {connectivity_issues}")
        else:
            logger.info("All ventricular connections validated successfully")
    
    def _check_cavity_connection(self, ventricle1: VentricleType, 
                                ventricle2: VentricleType) -> bool:
        """Check if two ventricles are connected through adjacent voxels."""
        if self.cavity_mask is None:
            return False
        
        region1 = self.ventricular_regions[ventricle1]
        region2 = self.ventricular_regions[ventricle2]
        
        # Get cavity voxels for each ventricle
        mask1 = self.geometry_generator.generate_cavity_voxels(region1)
        mask2 = self.geometry_generator.generate_cavity_voxels(region2)
        
        # Check for adjacent voxels (6-connectivity)
        dilated_mask1 = ndimage.binary_dilation(mask1)
        connection_exists = np.any(dilated_mask1 & mask2)
        
        return connection_exists
    
    def get_csf_flow_pathways(self) -> List[List[VentricleType]]:
        """Get CSF flow pathways through ventricular system."""
        return self.csf_flow_pathways.copy()
    
    def get_ventricular_volume(self, ventricle_type: VentricleType) -> float:
        """Get volume of specific ventricle in mm³."""
        if self.cavity_mask is None:
            self.generate_cavity_topology()
        
        region = self.ventricular_regions[ventricle_type]
        cavity_voxels = self.geometry_generator.generate_cavity_voxels(region)
        
        voxel_volume = (self.grid.dimensions.resolution ** 3) / 1e9  # mm³
        total_volume = np.sum(cavity_voxels) * voxel_volume
        
        return total_volume
    
    def export_topology_summary(self) -> Dict[str, Any]:
        """Export comprehensive topology summary."""
        if self.cavity_mask is None:
            self.generate_cavity_topology()
        
        summary = {
            "developmental_stage": self.developmental_stage,
            "grid_resolution_um": self.grid.dimensions.resolution,
            "total_ventricular_volume_mm3": np.sum(self.cavity_mask) * 
                                           (self.grid.dimensions.resolution ** 3) / 1e9,
            "ventricles": {},
            "csf_flow_pathways": [[v.value for v in pathway] 
                                 for pathway in self.csf_flow_pathways]
        }
        
        # Add individual ventricle information
        for ventricle_type, region in self.ventricular_regions.items():
            volume = self.get_ventricular_volume(ventricle_type)
            summary["ventricles"][ventricle_type.value] = {
                "center_position_um": region.center_position,
                "dimensions_um": region.dimensions,
                "shape_type": region.shape_type,
                "volume_mm3": volume,
                "developmental_week": region.developmental_week,
                "connectivity": [c.value for c in region.connectivity]
            }
        
        return summary