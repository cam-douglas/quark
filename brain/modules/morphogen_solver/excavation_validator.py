#!/usr/bin/env python3
"""Excavation Validation System.

Validation algorithms for voxel excavation including CSF pathway connectivity,
volume validation against embryonic references, and mass conservation checks.

Integration: Validation component for voxel excavation system
Rationale: Focused validation algorithms separated from main excavation logic
"""

from typing import Dict, List
import numpy as np
from scipy import ndimage
import logging

from .ventricular_types import VentricleType
from .ventricular_topology import VentricularTopology

logger = logging.getLogger(__name__)

class ExcavationValidator:
    """Validation system for excavated ventricular cavities.
    
    Provides comprehensive validation including CSF pathway connectivity,
    volume validation against developmental biology references, and
    structural integrity checks.
    """
    
    def __init__(self, topology: VentricularTopology):
        """Initialize excavation validator.
        
        Args:
            topology: Ventricular topology system
        """
        self.topology = topology
        
        # Reference volumes from developmental biology literature (E8.5-E10.5)
        self.reference_volumes = {
            VentricleType.LATERAL_LEFT: 0.008,    # ~8 mm³ at E10.5
            VentricleType.LATERAL_RIGHT: 0.008,   # ~8 mm³ at E10.5
            VentricleType.THIRD: 0.002,           # ~2 mm³ at E10.5
            VentricleType.FOURTH: 0.003,          # ~3 mm³ at E10.5
            VentricleType.CEREBRAL_AQUEDUCT: 0.0005  # ~0.5 mm³ at E10.5
        }
        
        logger.info("Initialized ExcavationValidator")
    
    def validate_csf_pathways(self, cavity_mask: np.ndarray) -> bool:
        """Validate CSF flow pathways through excavated cavities.
        
        Args:
            cavity_mask: Excavated cavity mask
            
        Returns:
            True if all CSF pathways are connected
        """
        csf_pathways = self.topology.get_csf_flow_pathways()
        
        for pathway in csf_pathways:
            if not self._validate_pathway_connectivity(cavity_mask, pathway):
                logger.warning(f"CSF pathway connectivity failed: "
                             f"{[v.value for v in pathway]}")
                return False
        
        logger.info("All CSF pathways validated successfully")
        return True
    
    def _validate_pathway_connectivity(self, cavity_mask: np.ndarray,
                                     pathway: List[VentricleType]) -> bool:
        """Validate connectivity along a specific CSF pathway."""
        for i in range(len(pathway) - 1):
            current_ventricle = pathway[i]
            next_ventricle = pathway[i + 1]
            
            # Get cavity regions for each ventricle
            current_region = self.topology.ventricular_regions[current_ventricle]
            next_region = self.topology.ventricular_regions[next_ventricle]
            
            # Generate cavity masks for each region
            current_mask = self.topology.geometry_generator.generate_cavity_voxels(current_region)
            next_mask = self.topology.geometry_generator.generate_cavity_voxels(next_region)
            
            # Check for adjacent cavities
            dilated_current = ndimage.binary_dilation(current_mask & cavity_mask)
            connection_exists = np.any(dilated_current & next_mask & cavity_mask)
            
            if not connection_exists:
                return False
        
        return True
    
    def calculate_cavity_volumes(self, cavity_mask: np.ndarray) -> Dict[VentricleType, float]:
        """Calculate volume of each excavated ventricle.
        
        Args:
            cavity_mask: Excavated cavity mask
            
        Returns:
            Dictionary of ventricle volumes in mm³
        """
        cavity_volumes = {}
        voxel_volume = (self.topology.grid.dimensions.resolution ** 3) / 1e9  # mm³
        
        for ventricle_type, region in self.topology.ventricular_regions.items():
            # Get intersection of topology template with excavated cavities
            region_mask = self.topology.geometry_generator.generate_cavity_voxels(region)
            ventricle_cavity = region_mask & cavity_mask
            
            volume = np.sum(ventricle_cavity) * voxel_volume
            cavity_volumes[ventricle_type] = volume
            
            logger.debug(f"{ventricle_type.value} volume: {volume:.4f} mm³")
        
        return cavity_volumes
    
    def validate_against_references(self, cavity_volumes: Dict[VentricleType, float]) -> float:
        """Validate excavated volumes against reference data.
        
        Args:
            cavity_volumes: Calculated cavity volumes
            
        Returns:
            Validation score (0.0 to 1.0)
        """
        if not cavity_volumes:
            return 0.0
        
        validation_scores = []
        
        for ventricle_type, calculated_volume in cavity_volumes.items():
            if ventricle_type in self.reference_volumes:
                reference_volume = self.reference_volumes[ventricle_type]
                
                # Calculate relative error
                if reference_volume > 0:
                    relative_error = abs(calculated_volume - reference_volume) / reference_volume
                    score = max(0.0, 1.0 - relative_error)
                    validation_scores.append(score)
                    
                    logger.debug(f"{ventricle_type.value}: calculated={calculated_volume:.4f}, "
                               f"reference={reference_volume:.4f}, score={score:.3f}")
        
        overall_score = np.mean(validation_scores) if validation_scores else 0.0
        
        return overall_score
    
    def calculate_connectivity_matrix(self) -> List[List[bool]]:
        """Calculate connectivity matrix between all ventricles."""
        ventricle_types = list(VentricleType)
        n_ventricles = len(ventricle_types)
        
        connectivity_matrix = [[False] * n_ventricles for _ in range(n_ventricles)]
        
        for i, ventricle1 in enumerate(ventricle_types):
            region1 = self.topology.ventricular_regions[ventricle1]
            for connected_type in region1.connectivity:
                j = ventricle_types.index(connected_type)
                connectivity_matrix[i][j] = True
                connectivity_matrix[j][i] = True  # Symmetric connectivity
        
        return connectivity_matrix
    
    def calculate_pathway_length(self, pathway: List[VentricleType]) -> float:
        """Calculate total length of CSF pathway in µm."""
        total_length = 0.0
        
        for i in range(len(pathway) - 1):
            current_region = self.topology.ventricular_regions[pathway[i]]
            next_region = self.topology.ventricular_regions[pathway[i + 1]]
            
            # Calculate Euclidean distance between centers
            distance = np.sqrt(sum((a - b)**2 for a, b in 
                                 zip(current_region.center_position, 
                                     next_region.center_position)))
            total_length += distance
        
        return total_length
    
    def calculate_bottleneck_area(self, pathway: List[VentricleType]) -> float:
        """Calculate minimum cross-sectional area in pathway (µm²)."""
        min_area = float('inf')
        
        for ventricle_type in pathway:
            region = self.topology.ventricular_regions[ventricle_type]
            
            # Estimate cross-sectional area from dimensions
            if region.shape_type == "tube":
                # Circular cross-section
                radius = min(region.dimensions[0], region.dimensions[1]) / 2
                area = np.pi * radius**2
            else:
                # Approximate as elliptical cross-section
                area = np.pi * (region.dimensions[0] / 2) * (region.dimensions[1] / 2)
            
            min_area = min(min_area, area)
        
        return min_area if min_area != float('inf') else 0.0
