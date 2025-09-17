"""
Signaling Pathway Manager

This module manages signaling pathway activities and their connections
to growth factors for cellular responses.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .signaling_types import GrowthFactorType, SignalingPathway, GrowthFactorSignal


@dataclass
class PathwayActivity:
    """Activity level of a signaling pathway"""
    pathway: SignalingPathway
    activity_level: float  # 0.0 to 1.0
    upstream_factors: List[GrowthFactorType]
    downstream_effects: Dict[str, float]


class SignalingPathwayManager:
    """
    Manages signaling pathway activities and their connections
    to growth factors for cellular responses.
    """
    
    def __init__(self):
        """Initialize signaling pathway manager"""
        self.pathway_connections: Dict[GrowthFactorType, List[SignalingPathway]] = {}
        self.pathway_weights: Dict[SignalingPathway, Dict[str, float]] = {}
        self._setup_pathway_connections()
        self._setup_pathway_weights()
    
    def _setup_pathway_connections(self) -> None:
        """Setup growth factor to pathway connections"""
        self.pathway_connections = {
            GrowthFactorType.FGF: [SignalingPathway.MAPK, SignalingPathway.PI3K_AKT],
            GrowthFactorType.EGF: [SignalingPathway.MAPK, SignalingPathway.PI3K_AKT],
            GrowthFactorType.WNT: [SignalingPathway.WNT_BETA_CATENIN],
            GrowthFactorType.SHH: [SignalingPathway.HEDGEHOG],
            GrowthFactorType.BMP: [SignalingPathway.SMAD],
            GrowthFactorType.TGF_BETA: [SignalingPathway.SMAD],
            GrowthFactorType.PDGF: [SignalingPathway.MAPK, SignalingPathway.PI3K_AKT],
            GrowthFactorType.VEGF: [SignalingPathway.MAPK, SignalingPathway.PI3K_AKT]
        }
    
    def _setup_pathway_weights(self) -> None:
        """Setup pathway weights for different cellular responses"""
        self.pathway_weights = {
            SignalingPathway.MAPK: {
                "proliferation": 0.8,
                "differentiation": 0.2,
                "survival": 0.6,
                "migration": 0.7
            },
            SignalingPathway.PI3K_AKT: {
                "proliferation": 0.6,
                "differentiation": 0.1,
                "survival": 0.9,
                "migration": 0.5
            },
            SignalingPathway.WNT_BETA_CATENIN: {
                "proliferation": 0.7,
                "differentiation": 0.8,
                "survival": 0.4,
                "migration": 0.3
            },
            SignalingPathway.HEDGEHOG: {
                "proliferation": 0.3,
                "differentiation": 0.9,
                "survival": 0.5,
                "migration": 0.4
            },
            SignalingPathway.SMAD: {
                "proliferation": 0.2,
                "differentiation": 0.9,
                "survival": 0.6,
                "migration": 0.8
            },
            SignalingPathway.NOTCH: {
                "proliferation": 0.4,
                "differentiation": 0.7,
                "survival": 0.7,
                "migration": 0.2
            }
        }
    
    def calculate_pathway_activities(self, growth_factors: Dict[GrowthFactorType, GrowthFactorSignal],
                                   tissue_type: str = "default",
                                   developmental_stage: str = "mid_embryonic") -> Dict[SignalingPathway, float]:
        """
        Calculate activity levels of signaling pathways
        
        Args:
            growth_factors: Dictionary of growth factor signals
            tissue_type: Type of tissue for sensitivity adjustment
            developmental_stage: Developmental stage for dependency adjustment
            
        Returns:
            Dictionary of pathway activities
        """
        pathway_activities = {}
        
        # Initialize all pathways
        for pathway in SignalingPathway:
            pathway_activities[pathway] = 0.0
        
        # Calculate activity for each growth factor
        for factor_type, signal in growth_factors.items():
            if factor_type in self.pathway_connections:
                pathways = self.pathway_connections[factor_type]
                
                # Calculate signal strength
                signal_strength = signal.concentration * signal.receptor_expression
                
                # Apply tissue sensitivity (simplified)
                tissue_sensitivity = self._get_tissue_sensitivity(factor_type, tissue_type)
                
                # Apply stage dependency (simplified)
                stage_dependency = self._get_stage_dependency(factor_type, developmental_stage)
                
                # Calculate effective signal strength
                effective_strength = signal_strength * tissue_sensitivity * stage_dependency
                
                # Distribute to pathways
                for pathway in pathways:
                    pathway_activities[pathway] += effective_strength / len(pathways)
        
        # Normalize pathway activities
        for pathway in pathway_activities:
            pathway_activities[pathway] = min(1.0, pathway_activities[pathway])
        
        return pathway_activities
    
    def _get_tissue_sensitivity(self, factor_type: GrowthFactorType, tissue_type: str) -> float:
        """Get tissue sensitivity for a growth factor"""
        # Simplified tissue sensitivities
        sensitivities = {
            "neural_tube": {
                GrowthFactorType.FGF: 1.5,
                GrowthFactorType.WNT: 1.2,
                GrowthFactorType.SHH: 1.8,
                GrowthFactorType.BMP: 0.5
            },
            "mesoderm": {
                GrowthFactorType.FGF: 1.0,
                GrowthFactorType.WNT: 1.3,
                GrowthFactorType.SHH: 1.1,
                GrowthFactorType.BMP: 1.4
            },
            "endoderm": {
                GrowthFactorType.FGF: 0.9,
                GrowthFactorType.WNT: 1.1,
                GrowthFactorType.SHH: 0.7,
                GrowthFactorType.BMP: 1.2
            }
        }
        
        tissue_sens = sensitivities.get(tissue_type, sensitivities["neural_tube"])
        return tissue_sens.get(factor_type, 1.0)
    
    def _get_stage_dependency(self, factor_type: GrowthFactorType, developmental_stage: str) -> float:
        """Get developmental stage dependency for a growth factor"""
        # Simplified stage dependencies
        dependencies = {
            "early_embryonic": {
                GrowthFactorType.FGF: 1.5,
                GrowthFactorType.WNT: 1.3,
                GrowthFactorType.SHH: 0.8,
                GrowthFactorType.BMP: 0.6
            },
            "mid_embryonic": {
                GrowthFactorType.FGF: 1.2,
                GrowthFactorType.WNT: 1.1,
                GrowthFactorType.SHH: 1.2,
                GrowthFactorType.BMP: 1.0
            },
            "late_embryonic": {
                GrowthFactorType.FGF: 0.9,
                GrowthFactorType.WNT: 0.8,
                GrowthFactorType.SHH: 1.4,
                GrowthFactorType.BMP: 1.3
            },
            "fetal": {
                GrowthFactorType.FGF: 0.7,
                GrowthFactorType.WNT: 0.6,
                GrowthFactorType.SHH: 1.1,
                GrowthFactorType.BMP: 1.5
            }
        }
        
        stage_dep = dependencies.get(developmental_stage, dependencies["mid_embryonic"])
        return stage_dep.get(factor_type, 1.0)
    
    def calculate_cellular_response(self, pathway_activities: Dict[SignalingPathway, float],
                                 response_type: str) -> float:
        """
        Calculate cellular response based on pathway activities
        
        Args:
            pathway_activities: Dictionary of pathway activities
            response_type: Type of response ("proliferation", "differentiation", "survival", "migration")
            
        Returns:
            Response strength (0.0 to 2.0)
        """
        total_response = 0.0
        total_weight = 0.0
        
        for pathway, activity in pathway_activities.items():
            if pathway in self.pathway_weights:
                weight = self.pathway_weights[pathway].get(response_type, 0.0)
                total_response += activity * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize by total weight
        normalized_response = total_response / total_weight
        
        # Scale to 0.0 to 2.0 range
        return min(2.0, normalized_response * 2.0)
    
    def get_pathway_statistics(self, pathway_activities: Dict[SignalingPathway, float]) -> Dict[str, float]:
        """Get statistics about pathway activities"""
        if not pathway_activities:
            return {}
        
        activities = list(pathway_activities.values())
        
        return {
            "average_activity": np.mean(activities),
            "median_activity": np.median(activities),
            "std_activity": np.std(activities),
            "min_activity": min(activities),
            "max_activity": max(activities),
            "active_pathways": sum(1 for activity in activities if activity > 0.1),
            "total_pathways": len(activities)
        }
    
    def get_pathway_contribution(self, pathway: SignalingPathway,
                               pathway_activities: Dict[SignalingPathway, float],
                               response_type: str) -> float:
        """Get contribution of a specific pathway to a response type"""
        if pathway not in pathway_activities:
            return 0.0
        
        activity = pathway_activities[pathway]
        weight = self.pathway_weights.get(pathway, {}).get(response_type, 0.0)
        
        return activity * weight
    
    def identify_dominant_pathways(self, pathway_activities: Dict[SignalingPathway, float],
                                 threshold: float = 0.3) -> List[SignalingPathway]:
        """Identify pathways with activity above threshold"""
        dominant_pathways = []
        
        for pathway, activity in pathway_activities.items():
            if activity >= threshold:
                dominant_pathways.append(pathway)
        
        # Sort by activity level
        dominant_pathways.sort(key=lambda p: pathway_activities[p], reverse=True)
        
        return dominant_pathways
