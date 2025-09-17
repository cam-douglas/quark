"""
Proliferation Rate Controller

Controls proliferation rates with density-dependent inhibition and growth factors.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional

from .proliferation_control_types import (
    ProliferationState, ProliferationContext, ProliferationRate, ProliferationControlParameters
)
from .density_inhibition_manager import DensityInhibitionManager
from .neuroepithelial_cells import NeuroepithelialCell
from .apoptosis_manager import ApoptosisManager


class ProliferationRateController:
    """
    Controls overall proliferation rates with density-dependent inhibition
    """
    
    def __init__(self):
        """Initialize proliferation rate controller"""
        self.control_parameters = ProliferationControlParameters(
            base_proliferation_rate=0.08,  # Base rate per hour
            density_inhibition_threshold=0.7,
            growth_factor_sensitivity=1.5,
            nutrient_dependence_factor=0.8,
            cell_cycle_checkpoint_strength=0.9
        )
        self.density_manager = DensityInhibitionManager()
        self.apoptosis_manager = ApoptosisManager()  # NEW: apoptosis integration
        self.proliferation_history: Dict[str, List[ProliferationRate]] = {}
    
    def control_cell_proliferation(self,
                                 target_cell: NeuroepithelialCell,
                                 neighboring_cells: List[NeuroepithelialCell],
                                 tissue_context: Dict[str, float],
                                 growth_factors: Dict[str, float]) -> ProliferationRate:
        """
        Control proliferation rate for a target cell
        
        Args:
            target_cell: Cell to control proliferation for
            neighboring_cells: Nearby cells
            tissue_context: Tissue-level context
            growth_factors: Growth factor concentrations
        
        Returns:
            Calculated proliferation rate
        """
        # Create proliferation context
        context = ProliferationContext(
            cell_density=tissue_context.get('cell_density', 0.5),
            growth_factor_levels=growth_factors,
            nutrient_availability=tissue_context.get('nutrient_availability', 0.8),
            developmental_stage=tissue_context.get('developmental_stage', 'E10.0'),
            tissue_type='neuroepithelium',
            spatial_position=target_cell.position,
            cell_age=tissue_context.get('cell_age', 5.0)
        )
        
        # Calculate density inhibition
        inhibition_result = self.density_manager.calculate_density_inhibition(
            target_cell, neighboring_cells, tissue_context
        )
        
        # Calculate growth factor response
        growth_factor_response = self._calculate_growth_factor_response(growth_factors)
        
        # Calculate final proliferation rate
        base_rate = self.control_parameters.base_proliferation_rate
        density_factor = 1.0 - inhibition_result.total_inhibition
        nutrient_factor = context.nutrient_availability * self.control_parameters.nutrient_dependence_factor
        
        current_rate = base_rate * density_factor * growth_factor_response * nutrient_factor
        current_rate = max(0.001, min(0.2, current_rate))  # Clamp to reasonable range
        
        proliferation_rate = ProliferationRate(
            base_rate=base_rate,
            current_rate=current_rate,
            inhibition_factor=inhibition_result.total_inhibition,
            growth_factor_response=growth_factor_response,
            density_inhibition=inhibition_result.total_inhibition,
            nutrient_dependence=nutrient_factor
        )
        
        # Store in history
        if target_cell.cell_id not in self.proliferation_history:
            self.proliferation_history[target_cell.cell_id] = []
        self.proliferation_history[target_cell.cell_id].append(proliferation_rate)
        
        # Apply to cell
        target_cell.proliferation_rate = current_rate
        
        return proliferation_rate
    
    def _calculate_growth_factor_response(self, growth_factors: Dict[str, float]) -> float:
        """Calculate response to growth factors"""
        # Growth factor effects (based on literature)
        fgf_effect = growth_factors.get('FGF', 0.5) * 1.2  # FGF promotes proliferation
        egf_effect = growth_factors.get('EGF', 0.5) * 1.1  # EGF promotes proliferation
        wnt_effect = growth_factors.get('WNT', 0.5) * 1.3  # WNT promotes proliferation
        shh_effect = growth_factors.get('SHH', 0.5) * 0.8  # SHH can reduce proliferation
        bmp_effect = growth_factors.get('BMP', 0.5) * 0.7  # BMP inhibits proliferation
        
        total_effect = (fgf_effect + egf_effect + wnt_effect + shh_effect + bmp_effect) / 5.0
        return max(0.1, min(2.0, total_effect * self.control_parameters.growth_factor_sensitivity))
    
    def update_proliferation_parameters(self, new_parameters: ProliferationControlParameters) -> None:
        """Update proliferation control parameters"""
        self.control_parameters = new_parameters
    
    def get_proliferation_statistics(self) -> Dict[str, float]:
        """Get proliferation statistics"""
        if not self.proliferation_history:
            return {}
        
        all_rates = [rate for rates_list in self.proliferation_history.values() for rate in rates_list]
        
        return {
            "total_proliferation_events": len(all_rates),
            "average_proliferation_rate": np.mean([r.current_rate for r in all_rates]),
            "average_inhibition": np.mean([r.inhibition_factor for r in all_rates]),
            "average_growth_factor_response": np.mean([r.growth_factor_response for r in all_rates])
        }