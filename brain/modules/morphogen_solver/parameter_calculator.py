#!/usr/bin/env python3
"""Parameter calculation utilities for morphogen interactions.

Provides mathematical functions for calculating morphogen interactions,
Hill functions, and parameter validation utilities.

Integration: Mathematical utilities for morphogen parameter systems
Rationale: Focused module for parameter calculations and validation
"""

from typing import Dict, Any, List
import numpy as np
import logging

from .parameter_types import InteractionParameters, ParameterValidationResult

logger = logging.getLogger(__name__)

class ParameterCalculator:
    """Utilities for morphogen parameter calculations.
    
    Provides mathematical functions for:
    - Hill function calculations
    - Interaction strength computations
    - Parameter validation
    - Cross-regulation effects
    """
    
    @staticmethod
    def calculate_hill_function(concentration: float, threshold: float, 
                              hill_coefficient: float, max_effect: float = 1.0) -> float:
        """Calculate Hill function for cooperative binding interactions.
        
        Args:
            concentration: Current morphogen concentration (nM)
            threshold: Half-maximal concentration (nM)
            hill_coefficient: Cooperativity parameter
            max_effect: Maximum effect magnitude
            
        Returns:
            Hill function value (0 to max_effect)
        """
        if concentration <= 0:
            return 0.0
        
        numerator = max_effect * (concentration ** hill_coefficient)
        denominator = (threshold ** hill_coefficient) + (concentration ** hill_coefficient)
        
        return numerator / denominator
    
    @staticmethod
    def calculate_interaction_strength(source_concentration: float,
                                     interaction: InteractionParameters) -> float:
        """Calculate interaction strength based on source morphogen concentration.
        
        Args:
            source_concentration: Source morphogen concentration (nM)
            interaction: Interaction parameters
            
        Returns:
            Interaction strength coefficient
        """
        if interaction.interaction_type == 'inhibition':
            # Inhibition: decreases with increasing source concentration
            hill_value = ParameterCalculator.calculate_hill_function(
                source_concentration, 
                interaction.threshold,
                interaction.hill_coefficient
            )
            return interaction.strength * (1.0 - hill_value)
        
        elif interaction.interaction_type == 'activation':
            # Activation: increases with increasing source concentration
            hill_value = ParameterCalculator.calculate_hill_function(
                source_concentration,
                interaction.threshold, 
                interaction.hill_coefficient
            )
            return interaction.strength * hill_value
        
        elif interaction.interaction_type == 'competition':
            # Competition: mutual inhibition
            hill_value = ParameterCalculator.calculate_hill_function(
                source_concentration,
                interaction.threshold,
                interaction.hill_coefficient
            )
            return interaction.strength * hill_value
        
        elif interaction.interaction_type == 'modulation':
            # Modulation: weak non-linear effect
            return interaction.strength * np.tanh(source_concentration / interaction.threshold)
        
        return 0.0
    
    @staticmethod
    def calculate_interaction_matrix(morphogens: List[str], 
                                   concentrations: Dict[str, float],
                                   all_interactions: Dict[str, List[InteractionParameters]]) -> np.ndarray:
        """Calculate morphogen interaction matrix.
        
        Args:
            morphogens: List of morphogen names
            concentrations: Current concentrations for each morphogen
            all_interactions: Interaction parameters for each morphogen
            
        Returns:
            Interaction matrix (morphogens x morphogens)
        """
        n_morphogens = len(morphogens)
        interaction_matrix = np.zeros((n_morphogens, n_morphogens))
        
        for i, source_morphogen in enumerate(morphogens):
            source_conc = concentrations.get(source_morphogen, 0.0)
            interactions = all_interactions.get(source_morphogen, [])
            
            for interaction in interactions:
                if interaction.target_morphogen in morphogens:
                    j = morphogens.index(interaction.target_morphogen)
                    strength = ParameterCalculator.calculate_interaction_strength(
                        source_conc, interaction
                    )
                    interaction_matrix[i, j] = strength
        
        return interaction_matrix
    
    @staticmethod
    def validate_parameter_ranges(diffusion_params, source_params, 
                                interactions: List[InteractionParameters]) -> ParameterValidationResult:
        """Validate parameter ranges and consistency.
        
        Args:
            diffusion_params: Diffusion parameters object
            source_params: Source parameters object
            interactions: List of interaction parameters
            
        Returns:
            ParameterValidationResult object
        """
        validation = ParameterValidationResult()
        
        # Validate diffusion parameters
        if diffusion_params.diffusion_coefficient <= 0:
            validation.add_error("Diffusion coefficient must be positive")
        
        if diffusion_params.degradation_rate <= 0:
            validation.add_error("Degradation rate must be positive")
        
        if diffusion_params.production_rate < 0:
            validation.add_error("Production rate must be non-negative")
        
        if diffusion_params.half_life <= 0:
            validation.add_error("Half-life must be positive")
        
        # Check consistency between degradation rate and half-life
        expected_half_life = np.log(2) / diffusion_params.degradation_rate
        half_life_error = abs(diffusion_params.half_life - expected_half_life) / expected_half_life
        
        if half_life_error > 0.1:  # 10% tolerance
            validation.add_warning(f"Half-life inconsistent with degradation rate: "
                                 f"expected {expected_half_life:.1f}s, got {diffusion_params.half_life:.1f}s")
        
        # Validate source parameters
        if source_params.intensity < 0:
            validation.add_error("Source intensity must be non-negative")
        
        if source_params.spatial_extent <= 0:
            validation.add_error("Spatial extent must be positive")
        
        # Validate interactions
        for i, interaction in enumerate(interactions):
            if interaction.strength <= 0:
                validation.add_error(f"Interaction {i}: strength must be positive")
            
            if interaction.hill_coefficient <= 0:
                validation.add_error(f"Interaction {i}: hill coefficient must be positive")
            
            if interaction.threshold <= 0:
                validation.add_error(f"Interaction {i}: threshold must be positive")
        
        # Add validation info
        validation.add_info("diffusion_coefficient", diffusion_params.diffusion_coefficient)
        validation.add_info("degradation_rate", diffusion_params.degradation_rate)
        validation.add_info("source_intensity", source_params.intensity)
        validation.add_info("interaction_count", len(interactions))
        
        return validation
    
    @staticmethod
    def calculate_steady_state_concentration(production_rate: float, 
                                           degradation_rate: float) -> float:
        """Calculate steady-state concentration for simple production-degradation.
        
        Args:
            production_rate: Production rate (nM/s)
            degradation_rate: Degradation rate (1/s)
            
        Returns:
            Steady-state concentration (nM)
        """
        if degradation_rate <= 0:
            raise ValueError("Degradation rate must be positive")
        
        return production_rate / degradation_rate
    
    @staticmethod
    def calculate_diffusion_length(diffusion_coefficient: float, 
                                 degradation_rate: float) -> float:
        """Calculate characteristic diffusion length.
        
        Args:
            diffusion_coefficient: Diffusion coefficient (µm²/s)
            degradation_rate: Degradation rate (1/s)
            
        Returns:
            Diffusion length (µm)
        """
        if degradation_rate <= 0:
            raise ValueError("Degradation rate must be positive")
        
        return np.sqrt(diffusion_coefficient / degradation_rate)
    
    @staticmethod
    def estimate_gradient_steepness(diffusion_length: float, 
                                  source_extent: float) -> float:
        """Estimate gradient steepness based on diffusion length and source size.
        
        Args:
            diffusion_length: Characteristic diffusion length (µm)
            source_extent: Source region extent (µm)
            
        Returns:
            Gradient steepness parameter (dimensionless)
        """
        if source_extent <= 0:
            raise ValueError("Source extent must be positive")
        
        return diffusion_length / source_extent
    
    @staticmethod
    def calculate_parameter_sensitivity(base_params: Dict[str, float], 
                                      param_name: str, 
                                      perturbation: float = 0.01) -> Dict[str, Any]:
        """Calculate parameter sensitivity analysis.
        
        Args:
            base_params: Base parameter values
            param_name: Parameter to perturb
            perturbation: Relative perturbation size (default 1%)
            
        Returns:
            Dictionary of sensitivity analysis results
        """
        if param_name not in base_params:
            raise ValueError(f"Parameter {param_name} not found in base parameters")
        
        base_value = base_params[param_name]
        perturbed_value = base_value * (1 + perturbation)
        
        # Calculate relative sensitivity
        relative_change = (perturbed_value - base_value) / base_value
        
        return {
            "parameter": param_name,
            "base_value": base_value,
            "perturbed_value": perturbed_value,
            "absolute_change": perturbed_value - base_value,
            "relative_change": relative_change,
            "perturbation_size": perturbation
        }
    
    @staticmethod
    def optimize_time_step(diffusion_coefficient: float, grid_spacing: float,
                         degradation_rate: float, safety_factor: float = 0.5) -> float:
        """Calculate optimal time step for numerical stability.
        
        Args:
            diffusion_coefficient: Diffusion coefficient (µm²/s)
            grid_spacing: Spatial grid spacing (µm)
            degradation_rate: Degradation rate (1/s)
            safety_factor: Safety factor for stability (default 0.5)
            
        Returns:
            Optimal time step (seconds)
        """
        # CFL condition for diffusion: dt < dx²/(2*D) for 1D, dx²/(6*D) for 3D
        diffusion_dt = (grid_spacing ** 2) / (6 * diffusion_coefficient)
        
        # Degradation stability: dt < 1/k
        degradation_dt = 1.0 / degradation_rate if degradation_rate > 0 else float('inf')
        
        # Take minimum and apply safety factor
        optimal_dt = min(diffusion_dt, degradation_dt) * safety_factor
        
        return optimal_dt
