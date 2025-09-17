#!/usr/bin/env python3
"""Biological Parameters System Coordinator.

Coordinates biological parameter components including morphogen parameters,
parameter calculations, and validation utilities.

Integration: Main interface for biological parameter management
Rationale: Unified coordinator maintaining backward compatibility
"""

from typing import Dict, Any, Optional, List
import logging

from .parameter_types import DiffusionParameters, SourceParameters, InteractionParameters
from .morphogen_parameters import MorphogenParametersDatabase
from .parameter_calculator import ParameterCalculator

logger = logging.getLogger(__name__)

class BiologicalParameters:
    """Biological parameters system coordinator.
    
    Provides unified interface for morphogen parameter management including
    parameter lookup, validation, and mathematical calculations.
    
    Key Components:
    - MorphogenParametersDatabase: Parameter storage and management
    - ParameterCalculator: Mathematical utilities and validation
    
    Maintains backward compatibility with existing code.
    """
    
    def __init__(self, species: str = "mouse", stage: str = "E8.5-E10.5"):
        """Initialize biological parameters system.
        
        Args:
            species: Model organism ('mouse', 'human', 'zebrafish')
            stage: Developmental stage ('E8.5-E10.5', 'E10.5-E12.5', etc.)
        """
        self.species = species
        self.stage = stage
        
        # Initialize component systems
        self.morphogen_db = MorphogenParametersDatabase(species, stage)
        self.calculator = ParameterCalculator()
        
        logger.info("Initialized biological parameters system coordinator")
        logger.info(f"Species: {species}, Stage: {stage}")
        logger.info(f"Available morphogens: {len(self.get_all_morphogens())}")
    
    def get_diffusion_parameters(self, morphogen: str) -> DiffusionParameters:
        """Get diffusion parameters for specified morphogen.
        
        Args:
            morphogen: Morphogen name ('SHH', 'BMP', 'WNT', 'FGF')
            
        Returns:
            DiffusionParameters object
        """
        return self.morphogen_db.get_diffusion_parameters(morphogen)
    
    def get_source_parameters(self, morphogen: str) -> SourceParameters:
        """Get source parameters for specified morphogen.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            SourceParameters object
        """
        return self.morphogen_db.get_source_parameters(morphogen)
    
    def get_interaction_parameters(self, morphogen: str) -> List[InteractionParameters]:
        """Get interaction parameters for specified morphogen.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            List of InteractionParameters objects
        """
        return self.morphogen_db.get_interaction_parameters(morphogen)
    
    def calculate_hill_function(self, concentration: float, threshold: float, 
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
        return self.calculator.calculate_hill_function(
            concentration, threshold, hill_coefficient, max_effect
        )
    
    def calculate_interaction_strength(self, source_concentration: float,
                                     interaction: InteractionParameters) -> float:
        """Calculate interaction strength based on source morphogen concentration.
        
        Args:
            source_concentration: Source morphogen concentration (nM)
            interaction: Interaction parameters
            
        Returns:
            Interaction strength coefficient
        """
        return self.calculator.calculate_interaction_strength(source_concentration, interaction)
    
    def get_all_morphogens(self) -> List[str]:
        """Get list of all available morphogens."""
        return self.morphogen_db.get_all_morphogens()
    
    def validate_parameters(self) -> Dict[str, bool]:
        """Validate biological parameter consistency.
        
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check parameter completeness
        morphogens = self.get_all_morphogens()
        results['parameters_complete'] = len(morphogens) > 0
        
        # Validate each morphogen's parameters
        for morphogen in morphogens:
            try:
                diffusion_params = self.get_diffusion_parameters(morphogen)
                source_params = self.get_source_parameters(morphogen)
                interactions = self.get_interaction_parameters(morphogen)
                
                validation = self.calculator.validate_parameter_ranges(
                    diffusion_params, source_params, interactions
                )
                
                results[f'{morphogen}_valid'] = validation.is_valid
                
            except Exception as e:
                logger.warning(f"Validation failed for {morphogen}: {e}")
                results[f'{morphogen}_valid'] = False
        
        # Overall validation
        results['all_valid'] = all(
            value for key, value in results.items() 
            if key.endswith('_valid')
        )
        
        return results
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get comprehensive parameter summary for debugging."""
        return {
            "system_info": {
                "species": self.species,
                "developmental_stage": self.stage,
                "morphogen_count": len(self.get_all_morphogens()),
                "morphogens": self.get_all_morphogens()
            },
            "database_summary": self.morphogen_db.get_database_summary(),
            "validation": self.validate_parameters()
        }
    
    def calculate_steady_state_concentration(self, morphogen: str) -> float:
        """Calculate steady-state concentration for morphogen.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            Steady-state concentration (nM)
        """
        diffusion_params = self.get_diffusion_parameters(morphogen)
        return self.calculator.calculate_steady_state_concentration(
            diffusion_params.production_rate,
            diffusion_params.degradation_rate
        )
    
    def calculate_diffusion_length(self, morphogen: str) -> float:
        """Calculate characteristic diffusion length for morphogen.
        
        Args:
            morphogen: Morphogen name
            
        Returns:
            Diffusion length (µm)
        """
        diffusion_params = self.get_diffusion_parameters(morphogen)
        return self.calculator.calculate_diffusion_length(
            diffusion_params.diffusion_coefficient,
            diffusion_params.degradation_rate
        )
    
    def optimize_time_step(self, morphogen: str, grid_spacing: float) -> float:
        """Calculate optimal time step for morphogen simulation.
        
        Args:
            morphogen: Morphogen name
            grid_spacing: Spatial grid spacing (µm)
            
        Returns:
            Optimal time step (seconds)
        """
        diffusion_params = self.get_diffusion_parameters(morphogen)
        return self.calculator.optimize_time_step(
            diffusion_params.diffusion_coefficient,
            grid_spacing,
            diffusion_params.degradation_rate
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "coordinator_info": {
                "species": self.species,
                "stage": self.stage,
                "available_morphogens": len(self.get_all_morphogens())
            },
            "components": {
                "morphogen_database": self.morphogen_db.get_database_summary(),
                "calculator_available": True
            },
            "validation": self.validate_parameters()
        }