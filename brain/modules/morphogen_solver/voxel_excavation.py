#!/usr/bin/env python3
"""Voxel Excavation Algorithm for Ventricular System.

Main excavation algorithm for creating ventricular cavities in the 1µm³
resolution neural tube grid with proper CSF flow pathway validation.

Integration: Spatial structure component for cavity construction
Rationale: Main excavation coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import SpatialGrid
from .ventricular_topology import VentricularTopology
from .excavation_parameters import ExcavationParameters, ExcavationResult
from .excavation_validator import ExcavationValidator

logger = logging.getLogger(__name__)

class VoxelExcavationAlgorithm:
    """Voxel excavation algorithm for ventricular cavity construction.
    
    Creates ventricular cavities in the 1µm³ resolution grid through precise
    voxel excavation with biological validation and CSF flow pathway verification.
    """
    
    def __init__(self, spatial_grid: SpatialGrid,
                 ventricular_topology: VentricularTopology,
                 excavation_params: Optional[ExcavationParameters] = None):
        """Initialize voxel excavation algorithm.
        
        Args:
            spatial_grid: 3D spatial grid for excavation
            ventricular_topology: Ventricular topology system
            excavation_params: Excavation algorithm parameters
        """
        self.grid = spatial_grid
        self.topology = ventricular_topology
        self.params = excavation_params or ExcavationParameters()
        
        # Initialize validator
        self.validator = ExcavationValidator(ventricular_topology)
        
        # Excavation state
        self.tissue_mask: Optional[np.ndarray] = None
        self.excavated_mask: Optional[np.ndarray] = None
        
        logger.info("Initialized VoxelExcavationAlgorithm")
        logger.info(f"Grid resolution: {spatial_grid.dimensions.resolution} µm")
    
    def initialize_tissue_mask(self, morphogen_concentrations: Dict[str, np.ndarray]) -> None:
        """Initialize tissue mask from morphogen concentration data."""
        dims = self.grid.dimensions
        
        # Create initial tissue mask (all voxels are tissue initially)
        self.tissue_mask = np.ones((dims.x_size, dims.y_size, dims.z_size), dtype=bool)
        
        # Exclude regions with very low morphogen activity (potential cavities)
        if 'SHH' in morphogen_concentrations and 'BMP' in morphogen_concentrations:
            shh_field = morphogen_concentrations['SHH']
            bmp_field = morphogen_concentrations['BMP']
            
            # Low morphogen activity indicates potential cavity regions
            low_activity_threshold = 0.01  # nM
            low_activity_mask = ((shh_field < low_activity_threshold) & 
                               (bmp_field < low_activity_threshold))
            
            # Mark low activity regions as potential excavation sites
            self.tissue_mask &= ~low_activity_mask
        
        tissue_volume = np.sum(self.tissue_mask) * (dims.resolution ** 3) / 1e9  # mm³
        logger.info(f"Initialized tissue mask: {tissue_volume:.3f} mm³ tissue volume")
    
    def excavate_ventricular_cavities(self) -> ExcavationResult:
        """Excavate ventricular cavities using topology-guided algorithm."""
        if self.tissue_mask is None:
            # Create default tissue mask
            dims = self.grid.dimensions
            self.tissue_mask = np.ones((dims.x_size, dims.y_size, dims.z_size), dtype=bool)
        
        # Generate topology-based cavity template
        topology_mask = self.topology.generate_cavity_topology()
        
        # Perform excavation
        excavated_mask = self._perform_excavation(topology_mask)
        
        # Validate and refine cavities
        refined_mask = self._refine_excavated_cavities(excavated_mask)
        
        # Validate CSF flow pathways
        connectivity_validated = self.validator.validate_csf_pathways(refined_mask)
        
        # Calculate cavity volumes
        cavity_volumes = self.validator.calculate_cavity_volumes(refined_mask)
        
        # Validate against reference volumes
        validation_score = self.validator.validate_against_references(cavity_volumes)
        
        self.excavated_mask = refined_mask
        
        # Create result
        result = ExcavationResult(
            excavated_mask=refined_mask,
            cavity_volumes=cavity_volumes,
            connectivity_validated=connectivity_validated,
            total_excavated_volume=np.sum(refined_mask) * 
                                 (self.grid.dimensions.resolution ** 3) / 1e9,
            validation_score=validation_score
        )
        
        logger.info(f"Excavation complete: {result.total_excavated_volume:.3f} mm³ excavated")
        logger.info(f"Validation score: {result.validation_score:.3f}")
        logger.info(f"CSF connectivity: {'PASS' if result.connectivity_validated else 'FAIL'}")
        
        return result
    
    def _perform_excavation(self, topology_mask: np.ndarray) -> np.ndarray:
        """Perform actual voxel excavation based on topology template."""
        # Start with topology template
        excavated_mask = topology_mask.copy()
        
        # Apply tissue mask constraints (don't excavate where tissue should be)
        excavated_mask &= ~self.tissue_mask
        
        # Remove small disconnected cavities
        excavated_mask = self._remove_small_cavities(excavated_mask)
        
        # Ensure minimum cavity sizes
        excavated_mask = self._ensure_minimum_cavity_sizes(excavated_mask)
        
        logger.info(f"Initial excavation: {np.sum(excavated_mask)} voxels")
        
        return excavated_mask
    
    def _refine_excavated_cavities(self, excavated_mask: np.ndarray) -> np.ndarray:
        """Refine excavated cavities through morphological operations."""
        refined_mask = excavated_mask.copy()
        
        # Apply smoothing iterations
        for i in range(self.params.smoothing_iterations):
            # Morphological opening to remove small protrusions
            refined_mask = ndimage.binary_opening(refined_mask)
            
            # Morphological closing to fill small gaps
            refined_mask = ndimage.binary_closing(refined_mask)
            
            logger.debug(f"Smoothing iteration {i+1}: {np.sum(refined_mask)} voxels")
        
        # Ensure cavities remain within volume constraints
        refined_mask = self._enforce_volume_constraints(refined_mask)
        
        return refined_mask
    
    def _remove_small_cavities(self, cavity_mask: np.ndarray) -> np.ndarray:
        """Remove cavities smaller than minimum volume threshold."""
        # Label connected components
        labeled_cavities, num_cavities = ndimage.label(cavity_mask)
        
        if num_cavities == 0:
            return cavity_mask
        
        # Calculate minimum voxels for minimum volume
        voxel_volume = (self.grid.dimensions.resolution ** 3) / 1e9  # mm³
        min_voxels = int(self.params.min_cavity_volume_mm3 / voxel_volume)
        
        # Filter out small cavities
        filtered_mask = np.zeros_like(cavity_mask)
        
        for cavity_id in range(1, num_cavities + 1):
            cavity_voxels = (labeled_cavities == cavity_id)
            cavity_size = np.sum(cavity_voxels)
            
            if cavity_size >= min_voxels:
                filtered_mask |= cavity_voxels
            else:
                logger.debug(f"Removed small cavity: {cavity_size} voxels")
        
        return filtered_mask
    
    def _ensure_minimum_cavity_sizes(self, cavity_mask: np.ndarray) -> np.ndarray:
        """Ensure each defined ventricle meets minimum size requirements."""
        # This is a placeholder - in practice, would use topology information
        # to identify specific ventricles and ensure they meet size requirements
        return cavity_mask
    
    def _enforce_volume_constraints(self, cavity_mask: np.ndarray) -> np.ndarray:
        """Enforce maximum volume constraints on cavities."""
        voxel_volume = (self.grid.dimensions.resolution ** 3) / 1e9  # mm³
        max_voxels = int(self.params.max_cavity_volume_mm3 / voxel_volume)
        
        # Label connected components
        labeled_cavities, num_cavities = ndimage.label(cavity_mask)
        
        constrained_mask = np.zeros_like(cavity_mask)
        
        for cavity_id in range(1, num_cavities + 1):
            cavity_voxels = (labeled_cavities == cavity_id)
            cavity_size = np.sum(cavity_voxels)
            
            if cavity_size <= max_voxels:
                constrained_mask |= cavity_voxels
            else:
                # Shrink oversized cavity (simple erosion approach)
                eroded_cavity = cavity_voxels
                while np.sum(eroded_cavity) > max_voxels:
                    eroded_cavity = ndimage.binary_erosion(eroded_cavity)
                
                constrained_mask |= eroded_cavity
                logger.warning(f"Shrunk oversized cavity from {cavity_size} to "
                             f"{np.sum(eroded_cavity)} voxels")
        
        return constrained_mask
    
    def get_csf_flow_analysis(self) -> Dict[str, Any]:
        """Analyze CSF flow characteristics of excavated system."""
        if self.excavated_mask is None:
            logger.warning("No excavated cavities available for CSF flow analysis")
            return {}
        
        # Calculate flow pathway analysis
        flow_analysis = {
            "total_cavity_volume_mm3": np.sum(self.excavated_mask) * 
                                     (self.grid.dimensions.resolution ** 3) / 1e9,
            "pathway_analysis": [],
            "connectivity_matrix": self.validator.calculate_connectivity_matrix()
        }
        
        # Analyze each CSF pathway
        for i, pathway in enumerate(self.topology.get_csf_flow_pathways()):
            pathway_info = {
                "pathway_id": i,
                "ventricles": [v.value for v in pathway],
                "total_length_um": self.validator.calculate_pathway_length(pathway),
                "bottleneck_area_um2": self.validator.calculate_bottleneck_area(pathway)
            }
            flow_analysis["pathway_analysis"].append(pathway_info)
        
        return flow_analysis