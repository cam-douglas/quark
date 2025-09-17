#!/usr/bin/env python3
"""Dura Mater Stress Analysis System.

Computes stress distribution and mechanical integrity analysis for dura mater
layer including attachment point stress calculations and safety factor analysis.

Integration: Stress analysis component for dura mater system
Rationale: Focused biomechanical analysis separated from main dura system
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

from .spatial_grid import GridDimensions
from .meninges_types import AttachmentPoint, MechanicalProperties

logger = logging.getLogger(__name__)

class DuraStressAnalyzer:
    """Stress analyzer for dura mater layer.
    
    Computes stress distribution, mechanical integrity, and safety factors
    for the dura mater layer based on attachment points and external forces.
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize dura stress analyzer.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
        """
        self.dims = grid_dimensions
        logger.info("Initialized DuraStressAnalyzer")
    
    def compute_stress_distribution(self, surface_mesh: np.ndarray,
                                   attachment_points: List[AttachmentPoint],
                                   external_forces: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Compute stress distribution in dura mater layer.
        
        Args:
            surface_mesh: Dura surface mesh
            attachment_points: List of attachment points
            external_forces: External force fields (optional)
            
        Returns:
            3D stress field array (Pa)
        """
        logger.info("Computing dura mater stress distribution")
        
        stress_field = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size))
        
        # Compute stress from attachment points
        for attachment in attachment_points:
            attachment_stress = self._compute_attachment_stress(surface_mesh, attachment)
            stress_field += attachment_stress
        
        # Apply external forces if provided
        if external_forces:
            for force_name, force_field in external_forces.items():
                if force_field.shape == stress_field.shape:
                    stress_field += force_field
                    logger.debug(f"Applied external force: {force_name}")
        
        # Calculate stress statistics
        surface_mask = surface_mesh > 0.1 * np.max(surface_mesh)
        surface_stresses = stress_field[surface_mask]
        
        if len(surface_stresses) > 0:
            stress_stats = (np.min(surface_stresses), np.mean(surface_stresses), 
                           np.max(surface_stresses))
            logger.info(f"Stress distribution: min={stress_stats[0]:.1e}, "
                       f"mean={stress_stats[1]:.1e}, max={stress_stats[2]:.1e} Pa")
        
        return stress_field
    
    def _compute_attachment_stress(self, surface_mesh: np.ndarray,
                                  attachment: AttachmentPoint) -> np.ndarray:
        """Compute stress field from single attachment point."""
        stress_field = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size))
        
        attach_vox = (
            int(attachment.location[0] / self.dims.resolution),
            int(attachment.location[1] / self.dims.resolution),
            int(attachment.location[2] / self.dims.resolution)
        )
        
        # Create stress field around attachment point
        x_grid, y_grid, z_grid = np.meshgrid(
            np.arange(self.dims.x_size),
            np.arange(self.dims.y_size), 
            np.arange(self.dims.z_size),
            indexing='ij'
        )
        
        # Distance from attachment point
        distance_voxels = np.sqrt(
            (x_grid - attach_vox[0])**2 + 
            (y_grid - attach_vox[1])**2 + 
            (z_grid - attach_vox[2])**2
        )
        
        # Stress decreases with distance from attachment
        max_stress = attachment.attachment_strength_n * 1e6  # Convert to Pa
        attachment_stress = max_stress * np.exp(-distance_voxels / 20.0)
        
        # Apply only where surface mesh exists
        surface_mask = surface_mesh > 0.1 * np.max(surface_mesh)
        attachment_stress[~surface_mask] = 0.0
        
        return attachment_stress
    
    def validate_mechanical_integrity(self, stress_field: np.ndarray,
                                     surface_mesh: np.ndarray,
                                     mechanical_props: MechanicalProperties,
                                     attachment_points: List[AttachmentPoint]) -> Dict[str, float]:
        """Validate mechanical integrity of dura mater layer.
        
        Args:
            stress_field: Computed stress field
            surface_mesh: Surface mesh array
            mechanical_props: Mechanical properties of dura
            attachment_points: List of attachment points
            
        Returns:
            Dictionary with integrity metrics
        """
        logger.info("Validating dura mater mechanical integrity")
        
        surface_mask = surface_mesh > 0.1 * np.max(surface_mesh)
        surface_stresses = stress_field[surface_mask]
        
        # Calculate integrity metrics
        max_stress = np.max(surface_stresses) if len(surface_stresses) > 0 else 0.0
        safety_factor = mechanical_props.tensile_strength_pa / max_stress if max_stress > 0 else float('inf')
        
        # Stress concentration factor
        mean_stress = np.mean(surface_stresses) if len(surface_stresses) > 0 else 0.0
        stress_concentration = max_stress / mean_stress if mean_stress > 0 else 1.0
        
        # Attachment point integrity
        attachment_integrity = self._validate_attachment_distribution(attachment_points)
        
        integrity_metrics = {
            "max_stress_pa": float(max_stress),
            "safety_factor": float(safety_factor),
            "stress_concentration_factor": float(stress_concentration),
            "attachment_integrity_score": attachment_integrity,
            "overall_integrity_score": min(1.0, safety_factor / 2.0) * attachment_integrity
        }
        
        logger.info(f"Mechanical integrity: safety_factor={safety_factor:.2f}, "
                   f"attachment_integrity={attachment_integrity:.3f}")
        
        return integrity_metrics
    
    def _validate_attachment_distribution(self, attachment_points: List[AttachmentPoint]) -> float:
        """Validate spatial distribution of attachment points."""
        if len(attachment_points) < 3:
            return 0.0  # Need minimum 3 points for stability
        
        # Calculate attachment point distribution score
        positions = np.array([ap.location for ap in attachment_points])
        
        # Check coverage of different regions
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        
        # Good distribution has points spread around the center
        distribution_score = min(1.0, np.std(distances) / np.mean(distances))
        
        return distribution_score
