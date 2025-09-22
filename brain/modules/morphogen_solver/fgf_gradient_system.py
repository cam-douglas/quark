#!/usr/bin/env python3
"""FGF Gradient System Implementation.

Implements FGF (Fibroblast Growth Factor) morphogen gradient system for
neural induction and progenitor maintenance during embryonic development
with integration for anterior-posterior patterning.

Integration: FGF component of multi-morphogen gradient system
Rationale: Focused FGF gradient implementation with neural maintenance
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .parameter_types import DiffusionParameters, SourceParameters
from .wnt_fgf_types import FGFGradientParameters, FGFReceptorType

logger = logging.getLogger(__name__)

class FGFSourceManager:
    """Source manager for FGF morphogen production."""
    
    def __init__(self, spatial_grid: SpatialGrid, source_params: SourceParameters):
        """Initialize FGF source manager."""
        self.grid = spatial_grid
        self.source_params = source_params
        self.source_regions: Dict[str, np.ndarray] = {}
        
    def configure_neural_sources(self, neural_tube_dimensions: GridDimensions) -> None:
        """Configure FGF sources for neural induction and maintenance."""
        dims = self.grid.dimensions
        
        # Neural plate border sources (early neural induction)
        border_width = max(1, int(0.1 * dims.x_size))  # 10% of width
        
        neural_border_source = np.zeros((dims.x_size, dims.y_size, dims.z_size))
        neural_border_source[:border_width, :, :] = self.source_params.intensity * 0.8
        neural_border_source[-border_width:, :, :] = self.source_params.intensity * 0.8
        
        self.source_regions['neural_border'] = neural_border_source
        
        # Isthmic organizer (midbrain-hindbrain boundary)
        isthmus_position = int(0.4 * dims.y_size)  # ~40% along A-P axis
        isthmus_width = max(1, int(0.05 * dims.y_size))  # 5% width
        
        isthmus_source = np.zeros((dims.x_size, dims.y_size, dims.z_size))
        y_start = max(0, isthmus_position - isthmus_width)
        y_end = min(dims.y_size, isthmus_position + isthmus_width)
        isthmus_source[:, y_start:y_end, :] = self.source_params.intensity
        
        self.source_regions['isthmus'] = isthmus_source
        
        self.grid.add_morphogen('FGF', initial_concentration=0.0)
        
        total_sources = sum(np.sum(source > 0) for source in self.source_regions.values())
        logger.info(f"Configured FGF sources: {total_sources} total source voxels")

class FGFDynamicsEngine:
    """Dynamics engine for FGF reaction-diffusion simulation."""
    
    def __init__(self, spatial_grid: SpatialGrid, source_manager: FGFSourceManager,
                 diffusion_params: DiffusionParameters, interactions: List):
        """Initialize FGF dynamics engine."""
        self.grid = spatial_grid
        self.source_manager = source_manager
        self.diffusion_params = diffusion_params
        self.interactions = interactions
        
    def simulate_timestep(self, dt: float) -> None:
        """Simulate one timestep of FGF dynamics."""
        # Get current FGF concentration
        fgf_field = self.grid.get_morphogen_concentration('FGF')
        
        # Apply diffusion
        diffusion_term = self._compute_diffusion(fgf_field, dt)
        
        # Apply degradation
        degradation_term = -self.diffusion_params.degradation_rate * fgf_field * dt
        
        # Apply production from sources
        production_term = np.zeros_like(fgf_field)
        for source_region in self.source_manager.source_regions.values():
            production_term += source_region * dt
        
        # Apply interactions with other morphogens
        interaction_term = self._compute_interactions(fgf_field, dt)
        
        # Update concentration
        new_concentration = fgf_field + diffusion_term + degradation_term + production_term + interaction_term
        new_concentration = np.maximum(new_concentration, 0.0)  # Non-negative concentrations
        
        self.grid.set_morphogen_concentration('FGF', new_concentration)
    
    def _compute_diffusion(self, concentration: np.ndarray, dt: float) -> np.ndarray:
        """Compute diffusion term using finite differences."""
        dims = self.grid.dimensions
        dx = dy = dz = dims.resolution * 1e-6  # Convert µm to m
        D = self.diffusion_params.diffusion_coefficient * 1e-12  # Convert µm²/s to m²/s
        
        # 3D Laplacian using finite differences
        laplacian = np.zeros_like(concentration)
        
        # X direction
        laplacian[1:-1, :, :] += (concentration[2:, :, :] - 2*concentration[1:-1, :, :] + 
                                 concentration[:-2, :, :]) / (dx**2)
        
        # Y direction
        laplacian[:, 1:-1, :] += (concentration[:, 2:, :] - 2*concentration[:, 1:-1, :] + 
                                 concentration[:, :-2, :]) / (dy**2)
        
        # Z direction
        laplacian[:, :, 1:-1] += (concentration[:, :, 2:] - 2*concentration[:, :, 1:-1] + 
                                 concentration[:, :, :-2]) / (dz**2)
        
        return D * laplacian * dt
    
    def _compute_interactions(self, fgf_concentration: np.ndarray, dt: float) -> np.ndarray:
        """Compute interaction terms with other morphogens."""
        total_interaction = np.zeros_like(fgf_concentration)

        for interaction in self.interactions:
            if not self.grid.has_morphogen(interaction.target_morphogen):
                continue
            
            target_conc = self.grid.get_morphogen_concentration(interaction.target_morphogen)
            
            if interaction.interaction_type == 'activation':
                hill = self._calculate_hill_function(target_conc, interaction.threshold, interaction.hill_coefficient)
                total_interaction += interaction.strength * hill * dt

            elif interaction.interaction_type == 'inhibition':
                hill = self._calculate_hill_function(target_conc, interaction.threshold, interaction.hill_coefficient)
                total_interaction -= interaction.strength * hill * fgf_concentration * dt

        return total_interaction

    def _calculate_hill_function(self, concentration: np.ndarray, threshold: float,
                               hill_coefficient: float) -> np.ndarray:
        """Calculate Hill function for cooperative binding."""
        safe_concentration = np.maximum(concentration, 1e-10)
        numerator = safe_concentration ** hill_coefficient
        denominator = (threshold ** hill_coefficient) + numerator
        return numerator / denominator

class FGFGeneExpressionMapper:
    """Gene expression mapper for FGF signaling."""
    
    def __init__(self, spatial_grid: SpatialGrid):
        """Initialize FGF gene expression mapper."""
        self.grid = spatial_grid
        self.expression_thresholds = self._initialize_fgf_thresholds()
        
    def _initialize_fgf_thresholds(self) -> Dict[str, float]:
        """Initialize FGF target gene expression thresholds."""
        return {
            'Sox2': 0.3,      # Neural progenitor maintenance (nM)
            'Pax6': 0.4,      # Neural specification
            'Nestin': 0.2,    # Neural stem cell marker
            'Fgf8': 0.5,      # Isthmic organizer
            'Spry2': 0.6,     # FGF signaling feedback inhibitor
        }
    
    def map_gene_expression(self, fgf_concentration: np.ndarray) -> Dict[str, np.ndarray]:
        """Map FGF concentration to target gene expression."""
        gene_expression = {}
        
        for gene, threshold in self.expression_thresholds.items():
            if gene == 'Spry2':
                # Spry2 provides negative feedback
                expression = np.minimum(1.0, fgf_concentration / threshold)
            else:
                # Most FGF targets are activated
                expression = np.minimum(1.0, fgf_concentration / threshold)
            
            gene_expression[gene] = expression
        
        return gene_expression

class FGFGradientSystem:
    """Complete FGF gradient system coordinator.
    
    Coordinates FGF morphogen gradient components for neural induction
    and progenitor maintenance during embryonic development.
    """
    
    def __init__(self, spatial_grid: SpatialGrid, 
                 diffusion_params: DiffusionParameters,
                 source_params: SourceParameters,
                 interactions: List = None):
        """Initialize FGF gradient system."""
        self.grid = spatial_grid
        self.diffusion_params = diffusion_params
        self.source_params = source_params
        self.interactions = interactions or []
        
        # Initialize component systems
        self.source_manager = FGFSourceManager(spatial_grid, source_params)
        self.dynamics_engine = FGFDynamicsEngine(
            spatial_grid, self.source_manager, diffusion_params, self.interactions
        )
        self.gene_expression = FGFGeneExpressionMapper(spatial_grid)
        
        logger.info("Initialized FGF gradient system coordinator")
        logger.info(f"FGF diffusion coefficient: {diffusion_params.diffusion_coefficient} µm²/s")
    
    def configure_sources(self, neural_tube_dimensions: GridDimensions) -> None:
        """Configure FGF source regions."""
        self.source_manager.configure_neural_sources(neural_tube_dimensions)
        logger.info("FGF sources configured for neural maintenance")
    
    def simulate_time_step(self, dt: float) -> None:
        """Simulate one timestep of FGF gradient dynamics."""
        self.dynamics_engine.simulate_timestep(dt)
    
    def get_concentration_field(self) -> np.ndarray:
        """Get current FGF concentration field."""
        return self.grid.get_morphogen_concentration('FGF')
    
    def get_gene_expression_map(self) -> Dict[str, np.ndarray]:
        """Get FGF target gene expression map."""
        fgf_concentration = self.get_concentration_field()
        return self.gene_expression.map_gene_expression(fgf_concentration)
