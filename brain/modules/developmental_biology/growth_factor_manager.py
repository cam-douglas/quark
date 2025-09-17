"""
Growth Factor Manager

This module manages growth factor responsiveness, signaling pathways,
and their effects on cell proliferation and differentiation.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .signaling_types import GrowthFactorType, GrowthFactorSignal, SignalingPathway
from .signaling_pathway_manager import SignalingPathwayManager


@dataclass
class SignalingContext:
    """Context for growth factor signaling"""
    growth_factors: Dict[GrowthFactorType, GrowthFactorSignal]
    developmental_stage: str
    tissue_type: str
    cell_state: str  # "proliferating", "quiescent", "differentiating"
    spatial_position: Tuple[float, float, float]
    receptor_expression_profile: Dict[GrowthFactorType, float]


@dataclass
class SignalingResponse:
    """Response to growth factor signaling"""
    proliferation_response: float  # 0.0 to 2.0
    differentiation_response: float  # 0.0 to 2.0
    survival_response: float  # 0.0 to 2.0
    pathway_activities: Dict[SignalingPathway, float]
    overall_response: float  # 0.0 to 2.0


class GrowthFactorManager:
    """
    Manages growth factor responsiveness and signaling pathways
    for cell proliferation and differentiation control.
    """
    
    def __init__(self):
        """Initialize growth factor manager"""
        self.pathway_manager = SignalingPathwayManager()
        self.tissue_sensitivities: Dict[str, Dict[GrowthFactorType, float]] = {}
        self.stage_dependencies: Dict[str, Dict[GrowthFactorType, float]] = {}
        self._setup_tissue_sensitivities()
        self._setup_stage_dependencies()
    
    
    def _setup_tissue_sensitivities(self) -> None:
        """Setup tissue-specific growth factor sensitivities"""
        self.tissue_sensitivities = {
            "neural_tube": {
                GrowthFactorType.FGF: 1.5,
                GrowthFactorType.WNT: 1.2,
                GrowthFactorType.SHH: 1.8,
                GrowthFactorType.BMP: 0.5,
                GrowthFactorType.EGF: 1.0,
                GrowthFactorType.TGF_BETA: 0.8,
                GrowthFactorType.PDGF: 0.7,
                GrowthFactorType.VEGF: 0.6
            },
            "mesoderm": {
                GrowthFactorType.FGF: 1.0,
                GrowthFactorType.WNT: 1.3,
                GrowthFactorType.SHH: 1.1,
                GrowthFactorType.BMP: 1.4,
                GrowthFactorType.EGF: 0.8,
                GrowthFactorType.TGF_BETA: 1.2,
                GrowthFactorType.PDGF: 1.1,
                GrowthFactorType.VEGF: 1.0
            },
            "endoderm": {
                GrowthFactorType.FGF: 0.9,
                GrowthFactorType.WNT: 1.1,
                GrowthFactorType.SHH: 0.7,
                GrowthFactorType.BMP: 1.2,
                GrowthFactorType.EGF: 1.3,
                GrowthFactorType.TGF_BETA: 1.0,
                GrowthFactorType.PDGF: 0.9,
                GrowthFactorType.VEGF: 0.8
            },
            "default": {
                GrowthFactorType.FGF: 1.0,
                GrowthFactorType.WNT: 1.0,
                GrowthFactorType.SHH: 1.0,
                GrowthFactorType.BMP: 1.0,
                GrowthFactorType.EGF: 1.0,
                GrowthFactorType.TGF_BETA: 1.0,
                GrowthFactorType.PDGF: 1.0,
                GrowthFactorType.VEGF: 1.0
            }
        }
    
    def _setup_stage_dependencies(self) -> None:
        """Setup developmental stage dependencies"""
        self.stage_dependencies = {
            "early_embryonic": {
                GrowthFactorType.FGF: 1.5,
                GrowthFactorType.WNT: 1.3,
                GrowthFactorType.SHH: 0.8,
                GrowthFactorType.BMP: 0.6,
                GrowthFactorType.EGF: 0.7,
                GrowthFactorType.TGF_BETA: 0.5,
                GrowthFactorType.PDGF: 0.6,
                GrowthFactorType.VEGF: 0.4
            },
            "mid_embryonic": {
                GrowthFactorType.FGF: 1.2,
                GrowthFactorType.WNT: 1.1,
                GrowthFactorType.SHH: 1.2,
                GrowthFactorType.BMP: 1.0,
                GrowthFactorType.EGF: 1.0,
                GrowthFactorType.TGF_BETA: 0.9,
                GrowthFactorType.PDGF: 0.8,
                GrowthFactorType.VEGF: 0.7
            },
            "late_embryonic": {
                GrowthFactorType.FGF: 0.9,
                GrowthFactorType.WNT: 0.8,
                GrowthFactorType.SHH: 1.4,
                GrowthFactorType.BMP: 1.3,
                GrowthFactorType.EGF: 1.2,
                GrowthFactorType.TGF_BETA: 1.1,
                GrowthFactorType.PDGF: 1.0,
                GrowthFactorType.VEGF: 0.9
            },
            "fetal": {
                GrowthFactorType.FGF: 0.7,
                GrowthFactorType.WNT: 0.6,
                GrowthFactorType.SHH: 1.1,
                GrowthFactorType.BMP: 1.5,
                GrowthFactorType.EGF: 1.4,
                GrowthFactorType.TGF_BETA: 1.3,
                GrowthFactorType.PDGF: 1.2,
                GrowthFactorType.VEGF: 1.1
            }
        }
    
    def calculate_signaling_response(self, context: SignalingContext) -> SignalingResponse:
        """
        Calculate cellular response to growth factor signaling
        
        Args:
            context: Signaling context with growth factors and cell state
            
        Returns:
            SignalingResponse with calculated responses
        """
        # Calculate pathway activities using pathway manager
        pathway_activities = self.pathway_manager.calculate_pathway_activities(
            context.growth_factors, context.tissue_type, context.developmental_stage
        )
        
        # Calculate responses using pathway manager
        proliferation_response = self.pathway_manager.calculate_cellular_response(
            pathway_activities, "proliferation"
        )
        
        differentiation_response = self.pathway_manager.calculate_cellular_response(
            pathway_activities, "differentiation"
        )
        
        survival_response = self.pathway_manager.calculate_cellular_response(
            pathway_activities, "survival"
        )
        
        # Calculate overall response
        overall_response = self._calculate_overall_response(
            proliferation_response, differentiation_response, survival_response, context
        )
        
        return SignalingResponse(
            proliferation_response=proliferation_response,
            differentiation_response=differentiation_response,
            survival_response=survival_response,
            pathway_activities=pathway_activities,
            overall_response=overall_response
        )
    
    
    def _calculate_overall_response(self, proliferation_response: float,
                                  differentiation_response: float,
                                  survival_response: float,
                                  context: SignalingContext) -> float:
        """Calculate overall cellular response"""
        # Weight responses based on cell state
        if context.cell_state == "proliferating":
            weights = [0.6, 0.2, 0.2]  # Proliferation, differentiation, survival
        elif context.cell_state == "differentiating":
            weights = [0.2, 0.6, 0.2]  # Differentiation, proliferation, survival
        else:  # quiescent
            weights = [0.1, 0.3, 0.6]  # Survival, differentiation, proliferation
        
        overall_response = (
            proliferation_response * weights[0] +
            differentiation_response * weights[1] +
            survival_response * weights[2]
        )
        
        return min(2.0, overall_response)
    
    def update_growth_factor_levels(self, context: SignalingContext,
                                  time_delta: float,
                                  new_concentrations: Optional[Dict[GrowthFactorType, float]] = None) -> SignalingContext:
        """Update growth factor levels over time"""
        # Update concentrations if provided
        if new_concentrations is not None:
            updated_factors = {}
            for factor_type, concentration in new_concentrations.items():
                if factor_type in context.growth_factors:
                    signal = context.growth_factors[factor_type]
                    updated_factors[factor_type] = GrowthFactorSignal(
                        factor_type=factor_type,
                        concentration=concentration,
                        receptor_expression=signal.receptor_expression,
                        pathway_activity=signal.pathway_activity,
                        downstream_effects=signal.downstream_effects
                    )
                else:
                    updated_factors[factor_type] = GrowthFactorSignal(
                        factor_type=factor_type,
                        concentration=concentration,
                        receptor_expression=0.5,  # Default
                        pathway_activity=0.0,
                        downstream_effects={}
                    )
        else:
            updated_factors = context.growth_factors.copy()
        
        return SignalingContext(
            growth_factors=updated_factors,
            developmental_stage=context.developmental_stage,
            tissue_type=context.tissue_type,
            cell_state=context.cell_state,
            spatial_position=context.spatial_position,
            receptor_expression_profile=context.receptor_expression_profile
        )
    
    def get_signaling_statistics(self, contexts: List[SignalingContext]) -> Dict[str, float]:
        """Get signaling statistics for multiple cells"""
        if not contexts:
            return {}
        
        responses = []
        pathway_activities = {pathway: [] for pathway in SignalingPathway}
        
        for context in contexts:
            response = self.calculate_signaling_response(context)
            responses.append(response.overall_response)
            
            for pathway, activity in response.pathway_activities.items():
                pathway_activities[pathway].append(activity)
        
        # Calculate statistics
        stats = {
            "average_response": np.mean(responses),
            "median_response": np.median(responses),
            "std_response": np.std(responses),
            "min_response": min(responses),
            "max_response": max(responses),
            "total_cells": len(contexts)
        }
        
        # Add pathway statistics
        for pathway, activities in pathway_activities.items():
            if activities:
                stats[f"{pathway.value}_avg"] = np.mean(activities)
                stats[f"{pathway.value}_max"] = max(activities)
        
        return stats
