#!/usr/bin/env python3
"""Pia Mater Vascular Network System.

Manages blood vessel network establishment through pia mater layer including
vessel pathway creation, connectivity validation, and metabolic exchange modeling.

Integration: Vascular network component for pia mater system
Rationale: Focused vascular network logic separated from main pia system
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import ndimage
import logging

from .spatial_grid import GridDimensions

logger = logging.getLogger(__name__)

class PiaVascularNetwork:
    """Vascular network manager for pia mater layer.
    
    Handles blood vessel pathway creation, network connectivity validation,
    and metabolic exchange capacity analysis for the pia mater layer.
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize pia vascular network.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
        """
        self.dims = grid_dimensions
        logger.info("Initialized PiaVascularNetwork")
    
    def establish_blood_vessel_pathways(self, neural_interface: np.ndarray,
                                       vascular_points: List[Tuple[float, float, float]]) -> Dict[str, np.ndarray]:
        """Establish blood vessel pathways through pia mater layer.
        
        Args:
            neural_interface: Pia mater neural interface mask
            vascular_points: List of vascular integration points
            
        Returns:
            Dictionary of vascular pathway masks
        """
        logger.info("Establishing blood vessel pathways")
        
        vascular_pathways = {}
        
        # Create blood vessel pathways
        for i, vascular_point in enumerate(vascular_points):
            pathway_id = f"pia_vessel_{i+1}"
            
            # Convert to voxel coordinates
            vascular_vox = (
                int(vascular_point[0] / self.dims.resolution),
                int(vascular_point[1] / self.dims.resolution),
                int(vascular_point[2] / self.dims.resolution)
            )
            
            # Create vessel pathway from vascular point to neural tissue
            pathway_mask = self._create_vessel_pathway(vascular_vox, neural_interface)
            vascular_pathways[pathway_id] = pathway_mask
        
        total_pathways = len(vascular_pathways)
        logger.info(f"Established {total_pathways} vessel pathways")
        
        return vascular_pathways
    
    def _create_vessel_pathway(self, entry_point: Tuple[int, int, int],
                              neural_interface: np.ndarray) -> np.ndarray:
        """Create single blood vessel pathway through pia mater."""
        pathway_mask = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size), dtype=bool)
        
        # Vessel radius ~5 µm (capillary scale)
        vessel_radius_voxels = max(1, int(5.0 / self.dims.resolution))
        
        x_entry, y_entry, z_entry = entry_point
        
        # Find target point on neural interface
        interface_points = np.where(neural_interface)
        
        if len(interface_points[0]) > 0:
            distances = np.sqrt(
                (interface_points[0] - x_entry)**2 + 
                (interface_points[1] - y_entry)**2 + 
                (interface_points[2] - z_entry)**2
            )
            nearest_idx = np.argmin(distances)
            target_point = (interface_points[0][nearest_idx], interface_points[1][nearest_idx], 
                          interface_points[2][nearest_idx])
            
            # Create pathway from entry to target
            pathway_points = self._interpolate_vessel_path(entry_point, target_point)
            
            # Mark vessel voxels with cylindrical cross-section
            for point in pathway_points:
                x, y, z = point
                
                for dx in range(-vessel_radius_voxels, vessel_radius_voxels + 1):
                    for dy in range(-vessel_radius_voxels, vessel_radius_voxels + 1):
                        px, py = x + dx, y + dy
                        
                        if (0 <= px < self.dims.x_size and 0 <= py < self.dims.y_size and 0 <= z < self.dims.z_size):
                            if dx**2 + dy**2 <= vessel_radius_voxels**2:
                                pathway_mask[px, py, z] = True
        
        return pathway_mask
    
    def _interpolate_vessel_path(self, start: Tuple[int, int, int], 
                               end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Interpolate vessel pathway points with slight curvature."""
        x0, y0, z0 = start
        x1, y1, z1 = end
        
        # Calculate number of steps
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        steps = max(1, int(distance))
        
        # Interpolate points with slight random curvature for realism
        points = []
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            
            # Base interpolation
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            z = z0 + t * (z1 - z0)
            
            # Add slight curvature (±2 voxels random deviation)
            if i > 0 and i < steps:
                x += np.random.randint(-2, 3)
                y += np.random.randint(-2, 3)
                z += np.random.randint(-2, 3)
            
            points.append((int(x), int(y), int(z)))
        
        return points
    
    def create_integrated_vessel_network(self, vascular_pathways: Dict[str, np.ndarray]) -> np.ndarray:
        """Create integrated blood vessel network from individual pathways.
        
        Args:
            vascular_pathways: Dictionary of individual vessel pathways
            
        Returns:
            Integrated blood vessel network mask
        """
        blood_vessel_network = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size), dtype=bool)
        
        for pathway_mask in vascular_pathways.values():
            blood_vessel_network |= pathway_mask
        
        total_vessel_volume = np.sum(blood_vessel_network) * (self.dims.resolution ** 3) / 1e9  # mm³
        logger.info(f"Created integrated vessel network: {total_vessel_volume:.4f} mm³ total volume")
        
        return blood_vessel_network
    
    def validate_vessel_network_connectivity(self, blood_vessel_network: np.ndarray) -> float:
        """Validate connectivity of blood vessel network."""
        # Label connected components
        labeled_network, n_components = ndimage.label(blood_vessel_network)
        
        if n_components == 0:
            return 0.0
        
        # Calculate connectivity score based on largest connected component
        component_sizes = []
        for i in range(1, n_components + 1):
            component_size = np.sum(labeled_network == i)
            component_sizes.append(component_size)
        
        largest_component = max(component_sizes)
        total_vessel_voxels = np.sum(blood_vessel_network)
        
        connectivity_score = largest_component / total_vessel_voxels if total_vessel_voxels > 0 else 0.0
        
        logger.info(f"Vessel network connectivity: {connectivity_score:.3f}")
        
        return connectivity_score
    
    def calculate_metabolic_exchange_capacity(self, neural_interface: np.ndarray,
                                            blood_vessel_network: np.ndarray,
                                            permeability_m2: float) -> Dict[str, float]:
        """Calculate metabolic exchange capacity of pia mater system."""
        # Calculate exchange surface area
        interface_area = np.sum(neural_interface) * (self.dims.resolution ** 2) / 1e6  # mm²
        
        # Calculate vascular density
        vessel_volume = np.sum(blood_vessel_network) * (self.dims.resolution ** 3) / 1e9  # mm³
        
        # Estimate exchange capacity based on surface area and permeability
        exchange_capacity = interface_area * permeability_m2 * 1e6  # Normalized units
        
        # Vascular density (vessels per unit brain volume)
        # Approximate brain volume as non-cavity space
        brain_volume = np.sum(neural_interface) * (self.dims.resolution ** 3) / 1e9  # Approximate
        vascular_density = vessel_volume / brain_volume if brain_volume > 0 else 0.0
        
        metrics = {
            "interface_area_mm2": float(interface_area),
            "vessel_volume_mm3": float(vessel_volume),
            "exchange_capacity": float(exchange_capacity),
            "vascular_density": float(vascular_density),
            "permeability_m2": float(permeability_m2)
        }
        
        logger.info(f"Metabolic exchange: interface={interface_area:.3f} mm², "
                   f"vascular_density={vascular_density:.4f}")
        
        return metrics
