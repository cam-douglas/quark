#!/usr/bin/env python3
"""Pia Mater Layer System.

Main coordinator for pia mater (innermost meningeal layer) system including
direct neural tissue contact, blood vessel pathway establishment, and
integration with the complete meninges scaffold.

Integration: Pia mater component of meninges scaffold system
Rationale: Main pia mater coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import SpatialGrid
from .ventricular_topology import VentricularTopology
from .arachnoid_system import ArachnoidSystem
from .meninges_types import (
    MeningesLayerType, MechanicalProperties, MeningesLayer
)
from .pia_vascular_network import PiaVascularNetwork

logger = logging.getLogger(__name__)

class PiaMaterSystem:
    """Pia mater layer system for embryonic brain development.
    
    Main coordinator for the innermost meningeal layer (pia mater) with
    direct neural tissue contact, blood vessel pathway establishment,
    and integration with arachnoid and dura layers.
    """
    
    def __init__(self, spatial_grid: SpatialGrid,
                 ventricular_topology: VentricularTopology,
                 arachnoid_system: ArachnoidSystem,
                 developmental_stage: str = "E8.5-E10.5"):
        """Initialize pia mater system.
        
        Args:
            spatial_grid: 3D spatial grid for positioning
            ventricular_topology: Ventricular topology system
            arachnoid_system: Arachnoid system for spatial reference
            developmental_stage: Embryonic developmental stage
        """
        self.grid = spatial_grid
        self.topology = ventricular_topology
        self.arachnoid_system = arachnoid_system
        self.developmental_stage = developmental_stage
        
        # Initialize vascular network system
        self.vascular_network = PiaVascularNetwork(spatial_grid.dimensions)
        
        # Pia mater layer definition
        self.pia_layer: Optional[MeningesLayer] = None
        self.neural_interface: Optional[np.ndarray] = None
        self.vascular_pathways: Dict[str, np.ndarray] = {}
        self.blood_vessel_network: Optional[np.ndarray] = None
        
        # Initialize pia properties
        self._initialize_pia_properties()
        
        logger.info("Initialized PiaMaterSystem")
        logger.info(f"Developmental stage: {developmental_stage}")
    
    def _initialize_pia_properties(self) -> None:
        """Initialize pia mater mechanical properties and structure."""
        # Pia mater mechanical properties (embryonic values)
        mechanical_props = MechanicalProperties(
            elastic_modulus_pa=2e5,        # ~0.2 MPa (softest meningeal layer)
            poisson_ratio=0.35,            # Most compressible
            thickness_um=5.0,              # ~5 µm thick in embryo
            density_kg_m3=1010.0,          # Very close to CSF density
            permeability_m2=1e-12,         # Highest permeability for metabolic exchange
            tensile_strength_pa=5e4        # ~50 kPa tensile strength
        )
        
        # Pia has no traditional attachment points - it adheres directly to neural tissue
        attachment_points = []
        
        # Create pia mater layer
        self.pia_layer = MeningesLayer(
            layer_type=MeningesLayerType.PIA_MATER,
            mechanical_properties=mechanical_props,
            attachment_points=attachment_points,
            developmental_week_start=4.5,   # Forms early with neural tube
            developmental_week_mature=6.0,  # Mature by week 6
            surface_mesh=None
        )
        
        logger.info("Initialized pia mater with direct neural tissue adhesion")
    
    def generate_neural_interface(self) -> np.ndarray:
        """Generate neural tissue interface for pia mater layer."""
        logger.info("Generating pia mater neural interface")
        
        # Get brain tissue surface (inverse of cavity mask)
        cavity_mask = self.topology.generate_cavity_topology()
        brain_tissue = ~cavity_mask
        
        # Pia mater follows brain surface contours exactly
        dims = self.grid.dimensions
        structure = ndimage.generate_binary_structure(3, 1)
        
        # Create pia surface by finding brain tissue boundary
        brain_boundary = brain_tissue & ~ndimage.binary_erosion(brain_tissue, structure=structure)
        
        # Pia extends slightly into brain tissue for tight adhesion
        pia_thickness_voxels = int(self.pia_layer.mechanical_properties.thickness_um / dims.resolution)
        
        pia_interface = brain_boundary
        for _ in range(pia_thickness_voxels):
            pia_interface = ndimage.binary_dilation(pia_interface, structure=structure)
            # Keep within brain tissue boundaries
            pia_interface &= brain_tissue
        
        self.neural_interface = pia_interface
        self.pia_layer.surface_mesh = pia_interface.astype(float)
        
        interface_area = np.sum(pia_interface) * (dims.resolution ** 2) / 1e6  # mm²
        logger.info(f"Generated neural interface: {interface_area:.3f} mm² contact area")
        
        return pia_interface
    
    def establish_blood_vessel_pathways(self) -> Dict[str, np.ndarray]:
        """Establish blood vessel pathways through pia mater layer."""
        if self.neural_interface is None:
            self.generate_neural_interface()
        
        # Get vascular integration points from arachnoid system
        vascular_points = self.arachnoid_system.vascular_integration_points
        
        # Use vascular network system to establish pathways
        self.vascular_pathways = self.vascular_network.establish_blood_vessel_pathways(
            neural_interface=self.neural_interface,
            vascular_points=vascular_points
        )
        
        # Create integrated blood vessel network
        self.blood_vessel_network = self.vascular_network.create_integrated_vessel_network(
            self.vascular_pathways)
        
        return self.vascular_pathways
    
    def validate_metabolic_exchange(self) -> Dict[str, float]:
        """Validate metabolic exchange capacity of pia mater system."""
        if self.neural_interface is None:
            self.generate_neural_interface()
        
        if self.blood_vessel_network is None:
            self.establish_blood_vessel_pathways()
        
        # Use vascular network system for validation
        metabolic_metrics = self.vascular_network.calculate_metabolic_exchange_capacity(
            neural_interface=self.neural_interface,
            blood_vessel_network=self.blood_vessel_network,
            permeability_m2=self.pia_layer.mechanical_properties.permeability_m2
        )
        
        return metabolic_metrics
    
    def _calculate_surface_conformity(self) -> float:
        """Calculate how well pia mater conforms to neural tissue surface."""
        if self.neural_interface is None:
            return 0.0
        
        # Get brain tissue surface
        cavity_mask = self.topology.generate_cavity_topology()
        brain_tissue = ~cavity_mask
        
        # Calculate conformity as overlap between pia interface and brain boundary
        brain_boundary = brain_tissue & ~ndimage.binary_erosion(brain_tissue)
        
        overlap = self.neural_interface & brain_boundary
        total_boundary = np.sum(brain_boundary)
        
        conformity_score = np.sum(overlap) / total_boundary if total_boundary > 0 else 0.0
        
        return conformity_score
    
    def export_pia_analysis(self) -> Dict[str, Any]:
        """Export comprehensive pia mater analysis."""
        if self.neural_interface is None:
            self.generate_neural_interface()
        
        if self.blood_vessel_network is None:
            self.establish_blood_vessel_pathways()
        
        # Validate metabolic exchange
        metabolic_metrics = self.validate_metabolic_exchange()
        
        # Validate vessel network connectivity
        vessel_connectivity = self.vascular_network.validate_vessel_network_connectivity(
            self.blood_vessel_network)
        
        # Calculate geometric metrics
        dims = self.grid.dimensions
        
        # Pia volume
        pia_volume = np.sum(self.neural_interface) * (dims.resolution ** 3) / 1e9
        
        analysis = {
            "developmental_stage": self.developmental_stage,
            "mechanical_properties": {
                "elastic_modulus_mpa": self.pia_layer.mechanical_properties.elastic_modulus_pa / 1e6,
                "thickness_um": self.pia_layer.mechanical_properties.thickness_um,
                "permeability_m2": self.pia_layer.mechanical_properties.permeability_m2,
                "density_kg_m3": self.pia_layer.mechanical_properties.density_kg_m3
            },
            "geometry": {
                "volume_mm3": float(pia_volume),
                "neural_interface_area_mm2": metabolic_metrics["interface_area_mm2"],
                "vessel_pathways_count": len(self.vascular_pathways),
                "total_vessel_volume_mm3": metabolic_metrics["vessel_volume_mm3"]
            },
            "metabolic_exchange": metabolic_metrics,
            "vascular_integration": {
                "pathway_count": len(self.vascular_pathways),
                "network_connectivity": float(vessel_connectivity),
                "arachnoid_integration": True  # Always integrated with arachnoid
            },
            "neural_tissue_interface": {
                "direct_contact": True,
                "adhesion_strength": "biological_adhesion",  # No mechanical attachments
                "surface_conformity": self._calculate_surface_conformity()
            }
        }
        
        return analysis