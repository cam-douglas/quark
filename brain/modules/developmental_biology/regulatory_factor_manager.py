"""
Regulatory Factor Manager

This module manages regulatory factors that influence cell cycle checkpoint
sensitivity, including p53, p21, CDK activity, and stress responses.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .cell_cycle_timing_engine import CheckpointType


@dataclass
class RegulatoryFactor:
    """A regulatory factor that influences checkpoint sensitivity"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    checkpoint_influence: Dict[CheckpointType, float]
    developmental_modulation: Dict[str, float]


class RegulatoryFactorManager:
    """
    Manages regulatory factors that influence cell cycle checkpoint sensitivity.
    Handles p53, p21, CDK activity, cyclin levels, and stress responses.
    """
    
    def __init__(self):
        """Initialize regulatory factor manager"""
        self.factors: Dict[str, RegulatoryFactor] = {}
        self._setup_default_factors()
    
    def _setup_default_factors(self) -> None:
        """Setup default regulatory factors"""
        # p53 activity (tumor suppressor)
        self.factors["p53_activity"] = RegulatoryFactor(
            name="p53_activity",
            current_value=1.0,
            min_value=0.0,
            max_value=2.0,
            checkpoint_influence={
                CheckpointType.G1_S: 1.0,
                CheckpointType.G2_M: 0.5,
                CheckpointType.SPINDLE_ASSEMBLY: 0.0,
                CheckpointType.DNA_DAMAGE: 1.5
            },
            developmental_modulation={
                "early_embryonic": 0.8,
                "mid_embryonic": 1.0,
                "late_embryonic": 1.2,
                "fetal": 1.5
            }
        )
        
        # p21 expression (CDK inhibitor)
        self.factors["p21_expression"] = RegulatoryFactor(
            name="p21_expression",
            current_value=1.0,
            min_value=0.0,
            max_value=2.0,
            checkpoint_influence={
                CheckpointType.G1_S: -0.5,  # Negative influence
                CheckpointType.G2_M: 0.0,
                CheckpointType.SPINDLE_ASSEMBLY: 0.0,
                CheckpointType.DNA_DAMAGE: 0.3
            },
            developmental_modulation={
                "early_embryonic": 0.5,
                "mid_embryonic": 1.0,
                "late_embryonic": 1.3,
                "fetal": 1.8
            }
        )
        
        # CDK activity
        self.factors["cdk_activity"] = RegulatoryFactor(
            name="cdk_activity",
            current_value=1.0,
            min_value=0.0,
            max_value=2.0,
            checkpoint_influence={
                CheckpointType.G1_S: 1.2,
                CheckpointType.G2_M: 1.5,
                CheckpointType.SPINDLE_ASSEMBLY: 0.8,
                CheckpointType.DNA_DAMAGE: 0.0
            },
            developmental_modulation={
                "early_embryonic": 1.5,
                "mid_embryonic": 1.0,
                "late_embryonic": 0.8,
                "fetal": 0.6
            }
        )
        
        # Cyclin levels
        self.factors["cyclin_levels"] = RegulatoryFactor(
            name="cyclin_levels",
            current_value=1.0,
            min_value=0.0,
            max_value=2.0,
            checkpoint_influence={
                CheckpointType.G1_S: 1.1,
                CheckpointType.G2_M: 1.3,
                CheckpointType.SPINDLE_ASSEMBLY: 0.9,
                CheckpointType.DNA_DAMAGE: 0.0
            },
            developmental_modulation={
                "early_embryonic": 1.3,
                "mid_embryonic": 1.0,
                "late_embryonic": 0.9,
                "fetal": 0.7
            }
        )
        
        # Growth factor signaling
        self.factors["growth_factor_signaling"] = RegulatoryFactor(
            name="growth_factor_signaling",
            current_value=1.0,
            min_value=0.0,
            max_value=2.0,
            checkpoint_influence={
                CheckpointType.G1_S: 1.0,
                CheckpointType.G2_M: 0.0,
                CheckpointType.SPINDLE_ASSEMBLY: 0.0,
                CheckpointType.DNA_DAMAGE: 0.0
            },
            developmental_modulation={
                "early_embryonic": 1.2,
                "mid_embryonic": 1.0,
                "late_embryonic": 0.9,
                "fetal": 0.8
            }
        )
        
        # DNA damage level (negative influence)
        self.factors["dna_damage_level"] = RegulatoryFactor(
            name="dna_damage_level",
            current_value=0.0,
            min_value=0.0,
            max_value=1.0,
            checkpoint_influence={
                CheckpointType.G1_S: -0.8,
                CheckpointType.G2_M: -0.6,
                CheckpointType.SPINDLE_ASSEMBLY: -0.4,
                CheckpointType.DNA_DAMAGE: -1.0
            },
            developmental_modulation={
                "early_embryonic": 0.5,  # More tolerant
                "mid_embryonic": 1.0,
                "late_embryonic": 1.2,
                "fetal": 1.5
            }
        )
        
        # Oxidative stress (negative influence)
        self.factors["oxidative_stress"] = RegulatoryFactor(
            name="oxidative_stress",
            current_value=0.0,
            min_value=0.0,
            max_value=1.0,
            checkpoint_influence={
                CheckpointType.G1_S: -0.3,
                CheckpointType.G2_M: -0.2,
                CheckpointType.SPINDLE_ASSEMBLY: -0.1,
                CheckpointType.DNA_DAMAGE: -0.5
            },
            developmental_modulation={
                "early_embryonic": 0.7,
                "mid_embryonic": 1.0,
                "late_embryonic": 1.1,
                "fetal": 1.3
            }
        )
        
        # Nutrient availability
        self.factors["nutrient_availability"] = RegulatoryFactor(
            name="nutrient_availability",
            current_value=1.0,
            min_value=0.0,
            max_value=2.0,
            checkpoint_influence={
                CheckpointType.G1_S: 0.8,
                CheckpointType.G2_M: 0.6,
                CheckpointType.SPINDLE_ASSEMBLY: 0.4,
                CheckpointType.DNA_DAMAGE: 0.0
            },
            developmental_modulation={
                "early_embryonic": 1.1,
                "mid_embryonic": 1.0,
                "late_embryonic": 0.9,
                "fetal": 0.8
            }
        )
    
    def calculate_regulatory_modifier(self, checkpoint_type: CheckpointType, 
                                    developmental_stage: str) -> float:
        """
        Calculate regulatory modifier for a specific checkpoint and stage
        
        Args:
            checkpoint_type: Type of checkpoint
            developmental_stage: Current developmental stage
            
        Returns:
            Regulatory modifier value
        """
        base_modifier = 1.0
        
        for factor_name, factor in self.factors.items():
            # Get influence on this checkpoint
            influence = factor.checkpoint_influence.get(checkpoint_type, 0.0)
            
            # Get developmental modulation
            dev_modulation = factor.developmental_modulation.get(developmental_stage, 1.0)
            
            # Calculate contribution
            if influence > 0:
                # Positive influence
                contribution = factor.current_value * influence * dev_modulation
                base_modifier *= (1.0 + contribution * 0.5)
            elif influence < 0:
                # Negative influence
                contribution = factor.current_value * abs(influence) * dev_modulation
                base_modifier *= (1.0 - contribution * 0.5)
        
        # Clamp between 0.1 and 2.0
        return max(0.1, min(2.0, base_modifier))
    
    def update_factor(self, factor_name: str, value: float) -> bool:
        """
        Update a regulatory factor value
        
        Args:
            factor_name: Name of the factor to update
            value: New value for the factor
            
        Returns:
            True if successful, False if factor not found
        """
        if factor_name not in self.factors:
            return False
        
        factor = self.factors[factor_name]
        factor.current_value = max(factor.min_value, min(factor.max_value, value))
        return True
    
    def get_factor_value(self, factor_name: str) -> Optional[float]:
        """Get current value of a regulatory factor"""
        if factor_name in self.factors:
            return self.factors[factor_name].current_value
        return None
    
    def get_all_factor_values(self) -> Dict[str, float]:
        """Get all current regulatory factor values"""
        return {
            name: factor.current_value 
            for name, factor in self.factors.items()
        }
    
    def simulate_stress_response(self, stress_type: str, intensity: float) -> None:
        """
        Simulate cellular stress response by updating relevant factors
        
        Args:
            stress_type: Type of stress (dna_damage, oxidative, nutrient_deprivation)
            intensity: Intensity of stress (0.0 to 1.0)
        """
        if stress_type == "dna_damage":
            self.update_factor("dna_damage_level", intensity)
            self.update_factor("p53_activity", 1.0 + intensity * 0.5)
            self.update_factor("p21_expression", 1.0 + intensity * 0.3)
        
        elif stress_type == "oxidative":
            self.update_factor("oxidative_stress", intensity)
            self.update_factor("p53_activity", 1.0 + intensity * 0.3)
        
        elif stress_type == "nutrient_deprivation":
            self.update_factor("nutrient_availability", 1.0 - intensity)
            self.update_factor("cdk_activity", 1.0 - intensity * 0.4)
            self.update_factor("cyclin_levels", 1.0 - intensity * 0.3)
    
    def simulate_growth_factor_stimulation(self, intensity: float) -> None:
        """
        Simulate growth factor stimulation
        
        Args:
            intensity: Intensity of stimulation (0.0 to 1.0)
        """
        self.update_factor("growth_factor_signaling", 1.0 + intensity)
        self.update_factor("cdk_activity", 1.0 + intensity * 0.2)
        self.update_factor("cyclin_levels", 1.0 + intensity * 0.3)
    
    def get_checkpoint_sensitivity_profile(self, checkpoint_type: CheckpointType, 
                                         developmental_stage: str) -> Dict[str, float]:
        """
        Get sensitivity profile for a checkpoint showing factor contributions
        
        Args:
            checkpoint_type: Type of checkpoint
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary of factor contributions to checkpoint sensitivity
        """
        profile = {}
        
        for factor_name, factor in self.factors.items():
            influence = factor.checkpoint_influence.get(checkpoint_type, 0.0)
            dev_modulation = factor.developmental_modulation.get(developmental_stage, 1.0)
            
            if influence != 0:
                contribution = factor.current_value * influence * dev_modulation
                profile[factor_name] = contribution
        
        return profile
