#!/usr/bin/env python3
"""SHH Reaction-Diffusion Dynamics Engine.

Implements the reaction-diffusion dynamics for SHH morphogen gradient
simulation including diffusion, degradation, and cross-regulation.

Integration: Component of SHH gradient system
Rationale: Focused module for SHH dynamics simulation
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .parameter_types import DiffusionParameters, InteractionParameters
from .shh_source_manager import SHHSourceManager

logger = logging.getLogger(__name__)

class SHHDynamicsEngine:
    """SHH reaction-diffusion dynamics simulation engine.
    
    Implements the complete reaction-diffusion equation for SHH:
    ∂C/∂t = D∇²C - kC + P + I
    
    Where:
    - C = SHH concentration
    - D = diffusion coefficient  
    - k = degradation rate
    - P = production term (from sources)
    - I = interaction terms (BMP antagonism, etc.)
    
    Key Features:
    - Efficient finite difference implementation
    - Adaptive time stepping
    - Cross-regulation with other morphogens
    - Stability monitoring and control
    """
    
    def __init__(self, spatial_grid: SpatialGrid, source_manager: SHHSourceManager,
                 diffusion_params: DiffusionParameters, interactions: list):
        """Initialize SHH dynamics engine.
        
        Args:
            spatial_grid: 3D spatial grid for concentration storage
            source_manager: SHH source region manager
            diffusion_params: SHH diffusion parameters
            interactions: List of interaction parameters
        """
        self.grid = spatial_grid
        self.source_manager = source_manager
        self.diffusion_params = diffusion_params
        self.interactions = interactions
        
        # Simulation state
        self.current_time = 0.0  # seconds
        self.time_step = 1.0     # seconds
        self.max_time_step = 5.0 # maximum allowed time step
        self.min_time_step = 0.1 # minimum allowed time step
        
        # Stability monitoring
        self.max_concentration_change = 0.0
        self.stability_threshold = 1.0  # nM/s
        
        logger.info("Initialized SHH dynamics engine")
        logger.info(f"Diffusion coefficient: {self.diffusion_params.diffusion_coefficient} µm²/s")
        logger.info(f"Degradation rate: {self.diffusion_params.degradation_rate} s⁻¹")
        logger.info(f"Initial time step: {self.time_step} s")
    
    def simulate_time_step(self, dt: Optional[float] = None) -> Dict[str, float]:
        """Simulate one time step of SHH dynamics.
        
        Args:
            dt: Time step size (seconds). Uses adaptive if None.
            
        Returns:
            Dictionary of simulation metrics
        """
        if dt is None:
            dt = self._calculate_adaptive_time_step()
        
        # Get current SHH concentration
        concentration = self.grid.concentrations['SHH'].copy()
        
        # Calculate all terms in reaction-diffusion equation
        diffusion_term = self._calculate_diffusion_term()
        degradation_term = self._calculate_degradation_term(concentration)
        production_term = self._calculate_production_term()
        interaction_term = self._calculate_interaction_terms(concentration)
        
        # Update concentration using forward Euler integration
        concentration_change = dt * (diffusion_term + degradation_term + 
                                   production_term + interaction_term)
        
        concentration_new = concentration + concentration_change
        
        # Ensure non-negative concentrations
        concentration_new = np.maximum(concentration_new, 0.0)
        
        # Update grid
        self.grid.concentrations['SHH'] = concentration_new
        
        # Apply boundary conditions
        self.grid.apply_boundary_conditions('SHH')
        
        # Update time
        self.current_time += dt
        
        # Calculate simulation metrics
        max_change = np.max(np.abs(concentration_change))
        mean_change = np.mean(np.abs(concentration_change))
        max_conc = np.max(concentration_new)
        mean_conc = np.mean(concentration_new)
        
        self.max_concentration_change = max_change
        
        # Log progress periodically
        if int(self.current_time) % 60 == 0:  # Every minute
            logger.debug(f"SHH dynamics t={self.current_time:.1f}s: "
                        f"max_conc={max_conc:.2f} nM, max_change={max_change:.3f} nM/s")
        
        return {
            "time_step_s": dt,
            "max_concentration_nM": max_conc,
            "mean_concentration_nM": mean_conc,
            "max_change_rate": max_change / dt,
            "mean_change_rate": mean_change / dt,
            "simulation_time_s": self.current_time
        }
    
    def _calculate_diffusion_term(self) -> np.ndarray:
        """Calculate diffusion term: D∇²C."""
        laplacian = self.grid.get_laplacian('SHH')
        return self.diffusion_params.diffusion_coefficient * laplacian
    
    def _calculate_degradation_term(self, concentration: np.ndarray) -> np.ndarray:
        """Calculate degradation term: -kC."""
        return -self.diffusion_params.degradation_rate * concentration
    
    def _calculate_production_term(self) -> np.ndarray:
        """Calculate production term from source regions."""
        return self.source_manager.calculate_production_term()
    
    def _calculate_interaction_terms(self, shh_concentration: np.ndarray) -> np.ndarray:
        """Calculate cross-regulation interaction terms.
        
        Args:
            shh_concentration: Current SHH concentration array
            
        Returns:
            Combined interaction term array
        """
        total_interaction = np.zeros_like(shh_concentration)
        
        for interaction in self.interactions:
            if interaction.target_morphogen not in self.grid.concentrations:
                continue
            
            target_concentration = self.grid.concentrations[interaction.target_morphogen]
            interaction_effect = self._calculate_single_interaction(
                shh_concentration, target_concentration, interaction
            )
            total_interaction += interaction_effect
        
        return total_interaction
    
    def _calculate_single_interaction(self, source_conc: np.ndarray, 
                                    target_conc: np.ndarray,
                                    interaction: InteractionParameters) -> np.ndarray:
        """Calculate single morphogen interaction effect.
        
        Args:
            source_conc: Source morphogen concentration (SHH)
            target_conc: Target morphogen concentration
            interaction: Interaction parameters
            
        Returns:
            Interaction effect array
        """
        if interaction.interaction_type == 'inhibition':
            # SHH inhibits target, which reduces target's effect on SHH
            hill_values = self._calculate_hill_function(
                source_conc, interaction.threshold, interaction.hill_coefficient
            )
            # Reduced inhibition from target due to SHH
            return interaction.strength * hill_values * target_conc
        
        elif interaction.interaction_type == 'activation':
            # SHH activates target, which increases target's positive effect on SHH
            hill_values = self._calculate_hill_function(
                source_conc, interaction.threshold, interaction.hill_coefficient
            )
            return interaction.strength * hill_values
        
        elif interaction.interaction_type == 'competition':
            # Competitive inhibition between SHH and target
            competitive_factor = target_conc / (interaction.threshold + target_conc)
            return -interaction.strength * competitive_factor * source_conc
        
        elif interaction.interaction_type == 'modulation':
            # Non-linear modulation
            modulation_factor = np.tanh(source_conc / interaction.threshold)
            return interaction.strength * modulation_factor
        
        return np.zeros_like(source_conc)
    
    def _calculate_hill_function(self, concentration: np.ndarray, threshold: float,
                               hill_coefficient: float) -> np.ndarray:
        """Calculate Hill function for cooperative binding.
        
        Args:
            concentration: Morphogen concentration array
            threshold: Half-maximal concentration
            hill_coefficient: Cooperativity parameter
            
        Returns:
            Hill function values (0 to 1)
        """
        # Avoid division by zero
        safe_concentration = np.maximum(concentration, 1e-10)
        
        numerator = safe_concentration ** hill_coefficient
        denominator = (threshold ** hill_coefficient) + numerator
        
        return numerator / denominator
    
    def _calculate_adaptive_time_step(self) -> float:
        """Calculate adaptive time step based on stability criteria.
        
        Returns:
            Optimal time step size (seconds)
        """
        # CFL condition for diffusion stability
        # dt < (dx²)/(2*D) for 3D diffusion
        grid_spacing = self.grid.resolution  # µm
        diffusion_dt = (grid_spacing ** 2) / (6 * self.diffusion_params.diffusion_coefficient)
        
        # Degradation stability condition
        # dt < 1/k for stability
        degradation_dt = 1.0 / self.diffusion_params.degradation_rate
        
        # Production stability condition
        # Limit based on maximum production rate
        max_production = self.diffusion_params.production_rate
        production_dt = 1.0 / max_production if max_production > 0 else float('inf')
        
        # Take minimum of all stability conditions
        stability_dt = min(diffusion_dt, degradation_dt, production_dt)
        
        # Apply bounds
        optimal_dt = np.clip(stability_dt, self.min_time_step, self.max_time_step)
        
        # Adaptive adjustment based on recent concentration changes
        if self.max_concentration_change > self.stability_threshold:
            # Reduce time step if system is changing rapidly
            optimal_dt *= 0.8
        elif self.max_concentration_change < self.stability_threshold * 0.1:
            # Increase time step if system is stable
            optimal_dt *= 1.2
        
        # Final bounds check
        optimal_dt = np.clip(optimal_dt, self.min_time_step, self.max_time_step)
        
        return optimal_dt
    
    def check_stability(self) -> Dict[str, Any]:
        """Check numerical stability of the simulation.
        
        Returns:
            Dictionary of stability metrics
        """
        concentration = self.grid.concentrations['SHH']
        
        # Check for NaN or infinite values
        has_nan = np.any(np.isnan(concentration))
        has_inf = np.any(np.isinf(concentration))
        
        # Check for negative values (should be impossible after clipping)
        has_negative = np.any(concentration < 0)
        
        # Check concentration range
        max_conc = np.max(concentration)
        min_conc = np.min(concentration)
        
        # Check for excessive concentrations (biological plausibility)
        excessive_conc = max_conc > 1000.0  # nM
        
        # Calculate gradients for stability assessment
        grad_x, grad_y, grad_z = self.grid.get_gradient('SHH')
        max_gradient = np.max(np.sqrt(grad_x**2 + grad_y**2 + grad_z**2))
        
        is_stable = not (has_nan or has_inf or has_negative or excessive_conc)
        
        return {
            "is_stable": is_stable,
            "has_nan": has_nan,
            "has_infinite": has_inf,
            "has_negative": has_negative,
            "excessive_concentration": excessive_conc,
            "concentration_range": {
                "min_nM": float(min_conc),
                "max_nM": float(max_conc)
            },
            "max_gradient": float(max_gradient),
            "max_change_rate": self.max_concentration_change,
            "current_time_step": self.time_step
        }
    
    def set_time_step(self, dt: float) -> None:
        """Set fixed time step size.
        
        Args:
            dt: Time step size (seconds)
        """
        if dt <= 0:
            raise ValueError("Time step must be positive")
        
        self.time_step = np.clip(dt, self.min_time_step, self.max_time_step)
        logger.info(f"Set SHH dynamics time step: {self.time_step} s")
    
    def reset_simulation(self) -> None:
        """Reset simulation state."""
        self.current_time = 0.0
        self.max_concentration_change = 0.0
        
        # Reset SHH concentrations
        if 'SHH' in self.grid.concentrations:
            self.grid.concentrations['SHH'].fill(0.0)
        
        logger.info("Reset SHH dynamics simulation state")
    
    def get_dynamics_summary(self) -> Dict[str, Any]:
        """Get comprehensive dynamics summary.
        
        Returns:
            Dictionary of dynamics information
        """
        stability = self.check_stability()
        
        return {
            "simulation_state": {
                "current_time_s": self.current_time,
                "time_step_s": self.time_step,
                "time_step_bounds": {
                    "min_s": self.min_time_step,
                    "max_s": self.max_time_step
                }
            },
            "parameters": {
                "diffusion_coefficient": self.diffusion_params.diffusion_coefficient,
                "degradation_rate": self.diffusion_params.degradation_rate,
                "half_life_min": self.diffusion_params.half_life / 60.0
            },
            "interactions": {
                "count": len(self.interactions),
                "types": [i.interaction_type for i in self.interactions]
            },
            "stability": stability
        }
