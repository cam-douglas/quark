#!/usr/bin/env python3
"""Arachnoid Membrane System.

Main coordinator for arachnoid mater (middle meningeal layer) system including
trabecular structure modeling, subarachnoid CSF space creation, and vascular
integration points for embryonic brain development.

Integration: Arachnoid component of meninges scaffold system
Rationale: Main arachnoid coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import SpatialGrid
from .ventricular_topology import VentricularTopology
from .dura_mater_system import DuraMaterSystem
from .meninges_types import (
    MeningesLayerType, MechanicalProperties, MeningesLayer, AttachmentPoint, AttachmentPointType
)
from .arachnoid_trabecular import ArachnoidTrabecularGenerator
from .arachnoid_vascular import ArachnoidVascularIntegrator

logger = logging.getLogger(__name__)

class ArachnoidSystem:
    """Arachnoid membrane system for embryonic brain development.
    
    Main coordinator for the middle meningeal layer (arachnoid mater) with
    trabecular structure, subarachnoid CSF space creation, and vascular
    integration points for complete brain protection and CSF circulation.
    """
    
    def __init__(self, spatial_grid: SpatialGrid,
                 ventricular_topology: VentricularTopology,
                 dura_system: DuraMaterSystem,
                 developmental_stage: str = "E8.5-E10.5"):
        """Initialize arachnoid system.
        
        Args:
            spatial_grid: 3D spatial grid for positioning
            ventricular_topology: Ventricular topology system
            dura_system: Dura mater system for spatial reference
            developmental_stage: Embryonic developmental stage
        """
        self.grid = spatial_grid
        self.topology = ventricular_topology
        self.dura_system = dura_system
        self.developmental_stage = developmental_stage
        
        # Initialize component systems
        self.trabecular_generator = ArachnoidTrabecularGenerator(spatial_grid.dimensions)
        self.vascular_integrator = ArachnoidVascularIntegrator(spatial_grid.dimensions)
        
        # Arachnoid layer definition
        self.arachnoid_layer: Optional[MeningesLayer] = None
        self.trabecular_mesh: Optional[np.ndarray] = None
        self.subarachnoid_space: Optional[np.ndarray] = None
        self.vascular_integration_points: List[Tuple[float, float, float]] = []
        
        # Initialize arachnoid properties
        self._initialize_arachnoid_properties()
        
        logger.info("Initialized ArachnoidSystem")
        logger.info(f"Developmental stage: {developmental_stage}")
    
    def _initialize_arachnoid_properties(self) -> None:
        """Initialize arachnoid membrane properties and structure."""
        # Arachnoid mechanical properties (embryonic values)
        mechanical_props = MechanicalProperties(
            elastic_modulus_pa=5e5,        # ~0.5 MPa (softer than dura)
            poisson_ratio=0.4,             # Compressible membrane
            thickness_um=20.0,             # ~20 µm thick in embryo
            density_kg_m3=1020.0,          # Close to CSF density
            permeability_m2=1e-15,         # Higher permeability than dura
            tensile_strength_pa=1e5        # ~100 kPa tensile strength
        )
        
        # Create minimal attachment points (arachnoid is more loosely attached)
        attachment_points = self._create_arachnoid_attachments()
        
        # Create vascular integration points
        self.vascular_integration_points = self.vascular_integrator.create_vascular_integration_points()
        
        # Create arachnoid layer
        self.arachnoid_layer = MeningesLayer(
            layer_type=MeningesLayerType.ARACHNOID_MATER,
            mechanical_properties=mechanical_props,
            attachment_points=attachment_points,
            developmental_week_start=5.0,   # Begins forming at week 5
            developmental_week_mature=8.0,  # Mature by week 8
            surface_mesh=None
        )
        
        logger.info(f"Initialized arachnoid with {len(attachment_points)} attachments, "
                   f"{len(self.vascular_integration_points)} vascular points")
    
    def _create_arachnoid_attachments(self) -> List[AttachmentPoint]:
        """Create minimal attachment points for arachnoid membrane."""
        dims = self.grid.dimensions
        center_x, center_y, center_z = dims.x_size/2, dims.y_size/2, dims.z_size/2
        
        # Arachnoid has fewer, weaker attachments than dura
        attachment_points = [
            AttachmentPoint(
                attachment_type=AttachmentPointType.CRANIAL_SUTURES,
                location=(center_x, center_y + 150, center_z + 100),  # Superior attachment
                attachment_strength_n=2e-7,  # Weaker than dura
                developmental_week=6.0,
                region_radius_um=10.0
            ),
            AttachmentPoint(
                attachment_type=AttachmentPointType.VASCULAR_ENTRY,
                location=(center_x - 100, center_y, center_z + 50),  # Left vascular entry
                attachment_strength_n=1e-7,
                developmental_week=6.5,
                region_radius_um=8.0
            ),
            AttachmentPoint(
                attachment_type=AttachmentPointType.VASCULAR_ENTRY,
                location=(center_x + 100, center_y, center_z + 50),  # Right vascular entry
                attachment_strength_n=1e-7,
                developmental_week=6.5,
                region_radius_um=8.0
            )
        ]
        
        return attachment_points
    
    def generate_trabecular_structure(self) -> np.ndarray:
        """Generate trabecular structure for arachnoid membrane."""
        # Get dura surface as outer boundary
        if self.dura_system.surface_mesh is None:
            self.dura_system.generate_dura_surface_mesh()
        
        # Get brain surface as inner boundary (inverse of lumen mask)
        lumen_mask = self.topology.get_lumen_mask()
        brain_surface = ~lumen_mask
        
        # Generate trabecular mesh
        self.trabecular_mesh = self.trabecular_generator.generate_trabecular_structure(
            dura_surface=self.dura_system.surface_mesh,
            brain_surface=brain_surface,
            n_trabeculae=50
        )
        
        return self.trabecular_mesh
    
    def create_subarachnoid_space(self) -> np.ndarray:
        """Create subarachnoid CSF space between arachnoid and pia."""
        logger.info("Creating subarachnoid CSF space")
        
        # Get brain surface (inverse of lumen mask)
        lumen_mask = self.topology.get_lumen_mask()
        brain_surface = ~lumen_mask
        
        # Get arachnoid surface (slightly inside dura)
        if self.dura_system.surface_mesh is None:
            self.dura_system.generate_dura_surface_mesh()
        
        # Create subarachnoid space between arachnoid and brain surface
        dims = self.grid.dimensions
        
        # Use distance transform to define layers accurately
        dist_from_brain = ndimage.distance_transform_edt(~brain_surface)

        # Arachnoid surface is a defined distance from the brain surface
        arachnoid_dist_um = 15.0  # Approx. distance for subarachnoid space
        arachnoid_dist_voxels = arachnoid_dist_um / dims.resolution
        
        # Pia mater is directly on the brain surface (0 distance)
        pia_dist_voxels = 1.0 # 1 voxel thick layer on the brain
        
        # Define subarachnoid space using distance thresholds
        subarachnoid_space = (dist_from_brain > pia_dist_voxels) & (dist_from_brain < arachnoid_dist_voxels)

        # Constrain the space to be within the dura mater boundary
        dura_mask = self.dura_system.surface_mesh > 0.1 * np.max(self.dura_system.surface_mesh)
        subarachnoid_space &= dura_mask
        
        self.subarachnoid_space = subarachnoid_space
        
        csf_volume = np.sum(subarachnoid_space) * (dims.resolution ** 3) / 1e9  # mm³
        logger.info(f"Created subarachnoid space: {csf_volume:.4f} mm³ CSF volume")
        
        return subarachnoid_space
    
    def integrate_vascular_pathways(self) -> Dict[str, Any]:
        """Integrate vascular pathways through arachnoid layer."""
        if self.subarachnoid_space is None:
            self.create_subarachnoid_space()
        
        # Use vascular integrator
        vascular_analysis = self.vascular_integrator.integrate_vascular_pathways(
            subarachnoid_space=self.subarachnoid_space,
            vascular_points=self.vascular_integration_points
        )
        
        return vascular_analysis
    
    def export_arachnoid_analysis(self) -> Dict[str, Any]:
        """Export comprehensive arachnoid analysis."""
        if self.trabecular_mesh is None:
            self.generate_trabecular_structure()
        
        if self.subarachnoid_space is None:
            self.create_subarachnoid_space()
        
        # Integrate vascular pathways
        vascular_analysis = self.integrate_vascular_pathways()
        
        # Validate trabecular connectivity
        trabecular_metrics = self.trabecular_generator.validate_trabecular_connectivity(self.trabecular_mesh)
        
        # Calculate geometric metrics
        dims = self.grid.dimensions
        
        # Subarachnoid volume
        csf_volume = np.sum(self.subarachnoid_space) * (dims.resolution ** 3) / 1e9
        
        # Trabecular volume
        trabecular_volume = trabecular_metrics.get("total_fiber_volume_mm3", 0.0)
        
        # Surface area (approximate)
        surface_area = np.sum(self.trabecular_mesh > 0.5) * (dims.resolution ** 2) / 1e6
        
        analysis = {
            "developmental_stage": self.developmental_stage,
            "mechanical_properties": {
                "elastic_modulus_mpa": self.arachnoid_layer.mechanical_properties.elastic_modulus_pa / 1e6,
                "thickness_um": self.arachnoid_layer.mechanical_properties.thickness_um,
                "permeability_m2": self.arachnoid_layer.mechanical_properties.permeability_m2
            },
            "geometry": {
                "trabecular_volume_mm3": float(trabecular_volume),
                "subarachnoid_volume_mm3": float(csf_volume),
                "surface_area_mm2": float(surface_area),
                "attachment_points_count": len(self.arachnoid_layer.attachment_points),
                "vascular_integration_points": len(self.vascular_integration_points)
            },
            "trabecular_connectivity": trabecular_metrics,
            "vascular_integration": vascular_analysis,
            "csf_circulation": {
                "subarachnoid_space_available": self.subarachnoid_space is not None,
                "csf_volume_mm3": float(csf_volume),
                "trabecular_connectivity": trabecular_metrics.get("connectivity_score", 0.0)
            }
        }
        
        return analysis