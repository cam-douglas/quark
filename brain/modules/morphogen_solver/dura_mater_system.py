#!/usr/bin/env python3
"""Dura Mater Layer System.

Main coordinator for dura mater (outer protective membrane) layer system
including positioning, mechanical properties simulation, and attachment
points to skull primordia for embryonic brain development.

Integration: Dura mater component of meninges scaffold system
Rationale: Main dura mater coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .ventricular_topology import VentricularTopology
from .meninges_types import (
    MeningesLayerType, MechanicalProperties, MeningesLayer
)
from .dura_attachment_manager import DuraAttachmentManager
from .dura_mesh_generator import DuraMeshGenerator
from .dura_stress_analyzer import DuraStressAnalyzer

logger = logging.getLogger(__name__)

class DuraMaterSystem:
    """Dura mater layer system for embryonic brain development.
    
    Main coordinator for the outermost meningeal layer (dura mater) with
    mechanical properties simulation, attachment point management, and
    integration with ventricular topology for complete brain protection.
    """
    
    def __init__(self, spatial_grid: SpatialGrid,
                 ventricular_topology: VentricularTopology,
                 developmental_stage: str = "E8.5-E10.5"):
        """Initialize dura mater system.
        
        Args:
            spatial_grid: 3D spatial grid for positioning
            ventricular_topology: Ventricular topology system
            developmental_stage: Embryonic developmental stage
        """
        self.grid = spatial_grid
        self.topology = ventricular_topology
        self.developmental_stage = developmental_stage
        
        # Initialize component systems
        self.attachment_manager = DuraAttachmentManager(spatial_grid.dimensions)
        self.mesh_generator = DuraMeshGenerator(spatial_grid.dimensions)
        self.stress_analyzer = DuraStressAnalyzer(spatial_grid.dimensions)
        
        # Dura mater layer definition
        self.dura_layer: Optional[MeningesLayer] = None
        self.surface_mesh: Optional[np.ndarray] = None
        self.stress_field: Optional[np.ndarray] = None
        
        # Initialize dura mater properties
        self._initialize_dura_properties()
        
        logger.info("Initialized DuraMaterSystem")
        logger.info(f"Developmental stage: {developmental_stage}")
        logger.info(f"Grid resolution: {spatial_grid.dimensions.resolution} µm")
    
    def _initialize_dura_properties(self) -> None:
        """Initialize dura mater mechanical properties and structure."""
        # Dura mater mechanical properties (embryonic values)
        mechanical_props = MechanicalProperties(
            elastic_modulus_pa=1.5e6,      # ~1.5 MPa (embryonic dura)
            poisson_ratio=0.45,            # Nearly incompressible
            thickness_um=50.0,             # ~50 µm thick in embryo
            density_kg_m3=1050.0,          # Slightly denser than CSF
            permeability_m2=1e-18,         # Very low permeability
            tensile_strength_pa=2e5        # ~200 kPa tensile strength
        )
        
        # Create attachment points
        attachment_points = self.attachment_manager.create_skull_attachment_points()
        
        # Create dura mater layer
        self.dura_layer = MeningesLayer(
            layer_type=MeningesLayerType.DURA_MATER,
            mechanical_properties=mechanical_props,
            attachment_points=attachment_points,
            developmental_week_start=4.0,   # Begins forming at week 4
            developmental_week_mature=7.0,  # Mature by week 7
            surface_mesh=None
        )
        
        logger.info(f"Initialized dura mater with {len(attachment_points)} attachment points")
        logger.info(f"Mechanical properties: E={mechanical_props.elastic_modulus_pa/1e6:.1f} MPa, "
                   f"thickness={mechanical_props.thickness_um} µm")
    
    def generate_dura_surface_mesh(self) -> np.ndarray:
        """Generate 3D surface mesh for dura mater layer.
        
        Returns:
            3D surface mesh coordinates
        """
        if self.dura_layer is None:
            raise ValueError("Dura layer not initialized")
        
        # Get neural tube lumen mask for inner boundary
        lumen_mask = self.topology.get_lumen_mask()
        
        # Generate surface mesh
        self.surface_mesh = self.mesh_generator.generate_surface_mesh(
            cavity_mask=lumen_mask,
            thickness_um=self.dura_layer.mechanical_properties.thickness_um,
            attachment_points=self.dura_layer.attachment_points
        )
        
        # Update layer with mesh
        self.dura_layer.surface_mesh = self.surface_mesh
        
        return self.surface_mesh
    
    def compute_stress_distribution(self, external_forces: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Compute stress distribution in dura mater layer.
        
        Args:
            external_forces: External force fields (optional)
            
        Returns:
            3D stress field array (Pa)
        """
        if self.surface_mesh is None:
            self.generate_dura_surface_mesh()
        
        if self.dura_layer is None:
            raise ValueError("Dura layer not initialized")
        
        # Compute stress distribution
        self.stress_field = self.stress_analyzer.compute_stress_distribution(
            surface_mesh=self.surface_mesh,
            attachment_points=self.dura_layer.attachment_points,
            external_forces=external_forces
        )
        
        return self.stress_field
    
    def validate_mechanical_integrity(self) -> Dict[str, float]:
        """Validate mechanical integrity of dura mater layer.
        
        Returns:
            Dictionary with integrity metrics
        """
        if self.stress_field is None:
            self.compute_stress_distribution()
        
        if self.dura_layer is None or self.stress_field is None:
            return {}
        
        # Validate using stress analyzer
        integrity_metrics = self.stress_analyzer.validate_mechanical_integrity(
            stress_field=self.stress_field,
            surface_mesh=self.surface_mesh,
            mechanical_props=self.dura_layer.mechanical_properties,
            attachment_points=self.dura_layer.attachment_points
        )
        
        return integrity_metrics
    
    def get_dura_thickness_map(self) -> np.ndarray:
        """Get 3D thickness map of dura mater layer.
        
        Returns:
            3D array with thickness values (µm)
        """
        if self.surface_mesh is None:
            self.generate_dura_surface_mesh()
        
        if self.dura_layer is None:
            raise ValueError("Dura layer not initialized")
        
        # Generate thickness map using mesh generator
        thickness_map = self.mesh_generator.get_thickness_map(
            surface_mesh=self.surface_mesh,
            base_thickness_um=self.dura_layer.mechanical_properties.thickness_um,
            stress_field=self.stress_field
        )
        
        return thickness_map
    
    def export_dura_analysis(self) -> Dict[str, Any]:
        """Export comprehensive dura mater analysis.
        
        Returns:
            Dictionary with complete dura analysis
        """
        if self.surface_mesh is None:
            self.generate_dura_surface_mesh()
        
        if self.stress_field is None:
            self.compute_stress_distribution()
        
        # Calculate analysis metrics
        integrity_metrics = self.validate_mechanical_integrity()
        thickness_map = self.get_dura_thickness_map()
        
        # Surface area calculation
        surface_mask = self.surface_mesh > 0.1 * np.max(self.surface_mesh)
        surface_area_um2 = np.sum(surface_mask) * (self.grid.dimensions.resolution ** 2)
        surface_area_mm2 = surface_area_um2 / 1e6
        
        # Volume calculation
        dura_volume_um3 = np.sum(thickness_map) * (self.grid.dimensions.resolution ** 3)
        dura_volume_mm3 = dura_volume_um3 / 1e9
        
        analysis = {
            "developmental_stage": self.developmental_stage,
            "mechanical_properties": {
                "elastic_modulus_mpa": self.dura_layer.mechanical_properties.elastic_modulus_pa / 1e6,
                "poisson_ratio": self.dura_layer.mechanical_properties.poisson_ratio,
                "base_thickness_um": self.dura_layer.mechanical_properties.thickness_um,
                "density_kg_m3": self.dura_layer.mechanical_properties.density_kg_m3,
                "tensile_strength_mpa": self.dura_layer.mechanical_properties.tensile_strength_pa / 1e6
            },
            "geometry": {
                "surface_area_mm2": float(surface_area_mm2),
                "volume_mm3": float(dura_volume_mm3),
                "attachment_points_count": len(self.dura_layer.attachment_points),
                "surface_mesh_points": int(np.sum(surface_mask))
            },
            "integrity_analysis": integrity_metrics,
            "attachment_points": [
                {
                    "type": ap.attachment_type.value,
                    "location_um": ap.location,
                    "strength_n": ap.attachment_strength_n,
                    "developmental_week": ap.developmental_week,
                    "radius_um": ap.region_radius_um
                }
                for ap in self.dura_layer.attachment_points
            ]
        }
        
        return analysis