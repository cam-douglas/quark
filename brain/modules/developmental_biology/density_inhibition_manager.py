"""
Density Inhibition Manager

Manages density-dependent inhibition of cell proliferation.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional

from .density_inhibition_types import (
    InhibitionType, DensityContext, InhibitionResult, InhibitionParameters
)
from .neuroepithelial_cells import NeuroepithelialCell


class DensityInhibitionManager:
    """
    Manages density-dependent inhibition of cell proliferation
    """
    
    def __init__(self):
        """Initialize density inhibition manager"""
        self.inhibition_parameters = InhibitionParameters(
            contact_threshold=0.8,
            crowding_threshold=0.7,
            nutrient_threshold=0.5,
            spatial_threshold=0.9,
            inhibition_strength=0.8,
            recovery_rate=0.1
        )
        self.inhibition_history: Dict[str, List[InhibitionResult]] = {}
    
    def calculate_density_inhibition(self,
                                   target_cell: NeuroepithelialCell,
                                   neighboring_cells: List[NeuroepithelialCell],
                                   tissue_context: Dict[str, float]) -> InhibitionResult:
        """
        Calculate density inhibition for a target cell
        
        Args:
            target_cell: Cell to calculate inhibition for
            neighboring_cells: Nearby cells
            tissue_context: Tissue-level context parameters
        
        Returns:
            Inhibition result with components and total inhibition
        """
        # Create density context
        density_context = self._create_density_context(
            target_cell, neighboring_cells, tissue_context
        )
        
        # Calculate inhibition components
        inhibition_components = {}
        
        # Contact inhibition
        if density_context.contact_area >= self.inhibition_parameters.contact_threshold:
            contact_inhibition = (density_context.contact_area - self.inhibition_parameters.contact_threshold) * 2.0
            inhibition_components[InhibitionType.CONTACT_INHIBITION] = min(1.0, contact_inhibition)
        
        # Crowding inhibition
        if density_context.local_density >= self.inhibition_parameters.crowding_threshold:
            crowding_inhibition = (density_context.local_density - self.inhibition_parameters.crowding_threshold) * 3.0
            inhibition_components[InhibitionType.CROWDING_INHIBITION] = min(1.0, crowding_inhibition)
        
        # Nutrient competition
        if density_context.nutrient_gradient <= self.inhibition_parameters.nutrient_threshold:
            nutrient_inhibition = (self.inhibition_parameters.nutrient_threshold - density_context.nutrient_gradient) * 1.5
            inhibition_components[InhibitionType.NUTRIENT_COMPETITION] = min(1.0, nutrient_inhibition)
        
        # Spatial constraint
        if density_context.global_density >= self.inhibition_parameters.spatial_threshold:
            spatial_inhibition = (density_context.global_density - self.inhibition_parameters.spatial_threshold) * 4.0
            inhibition_components[InhibitionType.SPATIAL_CONSTRAINT] = min(1.0, spatial_inhibition)
        
        # Calculate total inhibition
        total_inhibition = min(1.0, sum(inhibition_components.values()) * self.inhibition_parameters.inhibition_strength)
        
        # Determine inhibition strength category
        if total_inhibition < 0.3:
            strength = "weak"
        elif total_inhibition < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        # Calculate recovery time
        recovery_time = total_inhibition / self.inhibition_parameters.recovery_rate
        
        result = InhibitionResult(
            total_inhibition=total_inhibition,
            inhibition_components=inhibition_components,
            inhibition_strength=strength,
            recovery_time=recovery_time
        )
        
        # Store in history
        if target_cell.cell_id not in self.inhibition_history:
            self.inhibition_history[target_cell.cell_id] = []
        self.inhibition_history[target_cell.cell_id].append(result)
        
        return result
    
    def _create_density_context(self,
                              target_cell: NeuroepithelialCell,
                              neighboring_cells: List[NeuroepithelialCell],
                              tissue_context: Dict[str, float]) -> DensityContext:
        """Create density context for inhibition calculation"""
        # Calculate local density
        nearby_cells = [
            cell for cell in neighboring_cells
            if np.linalg.norm(np.array(cell.position) - np.array(target_cell.position)) <= 0.1
        ]
        local_density = len(nearby_cells) / 10.0  # Normalize to expected max neighbors
        
        # Get tissue context values
        global_density = tissue_context.get('global_density', 0.5)
        tissue_elasticity = tissue_context.get('tissue_elasticity', 0.8)
        nutrient_gradient = tissue_context.get('nutrient_gradient', 0.7)
        
        # Calculate contact area (simplified)
        contact_area = min(1.0, len(nearby_cells) * 0.1)
        
        return DensityContext(
            local_density=min(1.0, local_density),
            global_density=global_density,
            cell_size=1.0,  # Normalized cell size
            tissue_elasticity=tissue_elasticity,
            nutrient_gradient=nutrient_gradient,
            spatial_position=target_cell.position,
            neighbor_count=len(nearby_cells),
            contact_area=contact_area
        )
    
    def apply_density_inhibition(self,
                               cell: NeuroepithelialCell,
                               inhibition_result: InhibitionResult) -> None:
        """Apply density inhibition to cell proliferation"""
        # Reduce proliferation rate based on inhibition
        original_rate = cell.proliferation_rate
        inhibited_rate = original_rate * (1.0 - inhibition_result.total_inhibition)
        cell.proliferation_rate = max(0.01, inhibited_rate)  # Minimum rate
        
        # Extend cell cycle length
        if inhibition_result.total_inhibition > 0:
            cycle_extension = inhibition_result.total_inhibition * 5.0  # Up to 5 hours extension
            cell.cell_cycle_length += cycle_extension
    
    def get_inhibition_statistics(self) -> Dict[str, float]:
        """Get statistics on density inhibition"""
        if not self.inhibition_history:
            return {}
        
        all_results = [result for results_list in self.inhibition_history.values() for result in results_list]
        
        return {
            "total_inhibition_events": len(all_results),
            "average_inhibition": np.mean([r.total_inhibition for r in all_results]),
            "strong_inhibition_fraction": sum(1 for r in all_results if r.inhibition_strength == "strong") / len(all_results),
            "average_recovery_time": np.mean([r.recovery_time for r in all_results])
        }