#!/usr/bin/env python3
"""Competency Curve Calculator.

Calculates temporal competency curves for different cell fates including
mathematical modeling of competency profiles, morphogen responsiveness,
and restriction mechanisms over developmental time.

Integration: Calculation component for competency modeling system
Rationale: Focused mathematical competency calculations separated from main system
"""

from typing import Dict, List
import numpy as np
import random
import logging

from .competency_window_types import CompetencyWindow, CompetencyProfile, FateType

logger = logging.getLogger(__name__)

class CompetencyCurveCalculator:
    """Calculator for temporal competency curves.
    
    Provides mathematical modeling of competency curves over developmental
    time including different profile shapes, morphogen responsiveness,
    and temporal restriction mechanisms.
    """
    
    def __init__(self):
        """Initialize competency curve calculator."""
        logger.info("Initialized CompetencyCurveCalculator")
    
    def calculate_competency_level(self, window: CompetencyWindow, 
                                  current_time: float,
                                  morphogen_levels: Dict[str, float]) -> float:
        """Calculate current competency level for a fate.
        
        Args:
            window: Competency window definition
            current_time: Current developmental time (weeks)
            morphogen_levels: Current morphogen concentrations
            
        Returns:
            Current competency level (0-1)
        """
        # Check if within temporal window
        if current_time < window.competency_start or current_time > window.competency_end:
            return 0.0
        
        # Calculate base temporal competency
        base_competency = self._calculate_temporal_profile(window, current_time)
        
        # Apply morphogen modulation
        morphogen_factor = self._calculate_morphogen_modulation(window, morphogen_levels)
        
        # Final competency level
        competency_level = base_competency * morphogen_factor * window.max_competency_strength
        
        return np.clip(competency_level, 0.0, 1.0)
    
    def _calculate_temporal_profile(self, window: CompetencyWindow, current_time: float) -> float:
        """Calculate temporal competency profile."""
        start_time = window.competency_start
        peak_time = window.competency_peak
        end_time = window.competency_end
        
        # Normalize time within window
        if current_time <= peak_time:
            # Rising phase
            if peak_time > start_time:
                t_norm = (current_time - start_time) / (peak_time - start_time)
            else:
                t_norm = 1.0
        else:
            # Declining phase
            if end_time > peak_time:
                t_norm = 1.0 - (current_time - peak_time) / (end_time - peak_time)
            else:
                t_norm = 0.0
        
        # Apply profile shape
        if window.competency_profile == CompetencyProfile.EARLY_PEAK:
            # Exponential decay from early peak
            if current_time <= peak_time:
                profile_value = 1.0
            else:
                decay_rate = 2.0
                profile_value = np.exp(-decay_rate * (current_time - peak_time) / (end_time - peak_time))
                
        elif window.competency_profile == CompetencyProfile.SUSTAINED:
            # Sustained high level with gradual decline
            if current_time <= peak_time:
                profile_value = t_norm
            else:
                profile_value = 0.8 + 0.2 * (1.0 - t_norm)  # Slow decline
                
        elif window.competency_profile == CompetencyProfile.LATE_ONSET:
            # Sigmoid increase to late peak
            sigmoid_center = (start_time + peak_time) / 2
            sigmoid_width = (peak_time - start_time) / 4
            profile_value = 1.0 / (1.0 + np.exp(-(current_time - sigmoid_center) / sigmoid_width))
            
        elif window.competency_profile == CompetencyProfile.BELL_CURVE:
            # Gaussian curve centered on peak
            sigma = (end_time - start_time) / 6  # 6-sigma window
            profile_value = np.exp(-0.5 * ((current_time - peak_time) / sigma) ** 2)
            
        elif window.competency_profile == CompetencyProfile.STEP_FUNCTION:
            # Sharp step function
            if start_time <= current_time <= end_time:
                profile_value = 1.0
            else:
                profile_value = 0.0
        else:
            # Default triangular profile
            profile_value = t_norm
        
        return np.clip(profile_value, 0.0, 1.0)
    
    def _calculate_morphogen_modulation(self, window: CompetencyWindow,
                                       morphogen_levels: Dict[str, float]) -> float:
        """Calculate morphogen-dependent modulation of competency."""
        modulation_factor = 1.0
        
        # Positive modulation from required morphogens
        required_signals = 0.0
        for morphogen in window.required_morphogens:
            if morphogen in morphogen_levels:
                signal_strength = morphogen_levels[morphogen]
                required_signals += signal_strength
        
        if window.required_morphogens:
            avg_required = required_signals / len(window.required_morphogens)
            modulation_factor *= (0.1 + 0.9 * avg_required)  # Minimum 10% without signals
        
        # Negative modulation from inhibitory morphogens
        inhibitory_signals = 0.0
        for morphogen in window.inhibitory_morphogens:
            if morphogen in morphogen_levels:
                inhibition_strength = morphogen_levels[morphogen]
                inhibitory_signals += inhibition_strength
        
        if window.inhibitory_morphogens:
            avg_inhibitory = inhibitory_signals / len(window.inhibitory_morphogens)
            modulation_factor *= (1.0 - 0.8 * avg_inhibitory)  # Up to 80% inhibition
        
        return np.clip(modulation_factor, 0.0, 2.0)
    
    def calculate_competency_restriction_rate(self, window: CompetencyWindow,
                                            current_time: float,
                                            morphogen_levels: Dict[str, float]) -> float:
        """Calculate rate of competency restriction.
        
        Args:
            window: Competency window definition
            current_time: Current developmental time
            morphogen_levels: Current morphogen levels
            
        Returns:
            Restriction rate (per week)
        """
        restriction_rate = 0.0
        
        for mechanism in window.restriction_mechanisms:
            if mechanism.value == "temporal_decay":
                # Natural temporal decline
                if current_time > window.competency_peak:
                    time_past_peak = current_time - window.competency_peak
                    window_duration = window.competency_end - window.competency_peak
                    if window_duration > 0:
                        restriction_rate += 0.1 * (time_past_peak / window_duration)
                        
            elif mechanism.value == "morphogen_inhibition":
                # Morphogen-mediated restriction
                for inhibitory_morphogen in window.inhibitory_morphogens:
                    if inhibitory_morphogen in morphogen_levels:
                        inhibition = morphogen_levels[inhibitory_morphogen]
                        restriction_rate += 0.2 * inhibition
                        
            elif mechanism.value == "epigenetic_silencing":
                # Progressive epigenetic silencing
                age_factor = (current_time - window.competency_start) / (window.competency_end - window.competency_start)
                restriction_rate += 0.05 * age_factor
                
            elif mechanism.value == "transcriptional_repression":
                # Transcriptional repressor accumulation
                restriction_rate += 0.03 * (current_time - window.competency_start)
                
            elif mechanism.value == "metabolic_constraint":
                # Metabolic limitations over time
                restriction_rate += 0.02 * np.log(1 + current_time - window.competency_start)
        
        return np.clip(restriction_rate, 0.0, 1.0)  # Maximum 100% restriction per week
    
    def model_competency_inheritance(self, parent_competencies: Dict[FateType, float],
                                   division_type: str) -> Dict[FateType, float]:
        """Model how competency is inherited during division.
        
        Args:
            parent_competencies: Parent cell competency levels
            division_type: Type of division (symmetric/asymmetric)
            
        Returns:
            Daughter cell competency levels
        """
        daughter_competencies = {}
        
        for fate_type, parent_level in parent_competencies.items():
            if division_type == "symmetric_proliferative":
                # Both daughters inherit full competency
                daughter_level = parent_level * 0.95  # Slight dilution
                
            elif division_type == "asymmetric":
                # Asymmetric inheritance - one daughter gets more competency
                if random.random() < 0.5:
                    daughter_level = parent_level * 0.8  # Reduced competency
                else:
                    daughter_level = parent_level * 1.0  # Full competency
                    
            elif division_type == "symmetric_differentiative":
                # Both daughters lose competency (committing)
                daughter_level = parent_level * 0.3  # Major reduction
                
            else:
                daughter_level = parent_level * 0.9  # Default slight reduction
            
            daughter_competencies[fate_type] = np.clip(daughter_level, 0.0, 1.0)
        
        return daughter_competencies
    
    def calculate_fate_competition(self, competencies: Dict[FateType, float],
                                  morphogen_levels: Dict[str, float]) -> Dict[FateType, float]:
        """Calculate competitive fate specification.
        
        Args:
            competencies: Current competency levels for all fates
            morphogen_levels: Current morphogen concentrations
            
        Returns:
            Competitive fate probabilities
        """
        if not competencies:
            return {}
        
        # Apply morphogen weighting to competencies
        weighted_competencies = {}
        
        for fate_type, competency in competencies.items():
            morphogen_weight = self._get_morphogen_weight_for_fate(fate_type, morphogen_levels)
            weighted_competencies[fate_type] = competency * morphogen_weight
        
        # Normalize to probabilities (competitive exclusion)
        total_weight = sum(weighted_competencies.values())
        
        if total_weight > 0:
            fate_probabilities = {
                fate: weight / total_weight 
                for fate, weight in weighted_competencies.items()
            }
        else:
            # Equal probabilities if no signals
            num_fates = len(competencies)
            fate_probabilities = {fate: 1.0 / num_fates for fate in competencies.keys()}
        
        return fate_probabilities
    
    def _get_morphogen_weight_for_fate(self, fate_type: FateType, 
                                      morphogen_levels: Dict[str, float]) -> float:
        """Get morphogen weighting factor for specific fate."""
        # Fate-specific morphogen dependencies
        fate_morphogen_weights = {
            FateType.MOTOR_NEURON: {'SHH': 2.0, 'BMP': -1.0},  # High SHH, low BMP
            FateType.INTERNEURON_V0: {'SHH': 1.5, 'BMP': 0.5},
            FateType.INTERNEURON_V1: {'SHH': 1.0, 'BMP': 0.5},
            FateType.INTERNEURON_V2: {'SHH': 0.8, 'BMP': 0.3},
            FateType.INTERNEURON_DORSAL: {'SHH': -0.5, 'BMP': 2.0},  # Low SHH, high BMP
            FateType.NEURAL_CREST: {'BMP': 2.0, 'WNT': 1.5, 'SHH': -1.0},
            FateType.OLIGODENDROCYTE: {'SHH': 1.0, 'FGF': 0.5},
            FateType.ASTROCYTE: {'BMP': 1.0, 'FGF': -0.5},
            FateType.EPENDYMAL: {'FGF': 1.0, 'WNT': 0.5}
        }
        
        if fate_type not in fate_morphogen_weights:
            return 1.0  # Default weight
        
        weights = fate_morphogen_weights[fate_type]
        total_weight = 1.0
        
        for morphogen, weight_factor in weights.items():
            if morphogen in morphogen_levels:
                morphogen_level = morphogen_levels[morphogen]
                
                if weight_factor > 0:
                    # Positive influence
                    total_weight *= (1.0 + weight_factor * morphogen_level)
                else:
                    # Negative influence
                    total_weight *= (1.0 + weight_factor * morphogen_level)
        
        return max(0.1, total_weight)  # Minimum 10% weight
