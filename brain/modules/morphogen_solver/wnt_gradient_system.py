#!/usr/bin/env python3
"""WNT Gradient System Implementation.

Implements WNT (Wingless-related) morphogen gradient system for anterior-posterior
neural tube patterning during embryonic development including canonical and
non-canonical signaling pathways.

Integration: WNT component of multi-morphogen gradient system
Rationale: Focused WNT gradient implementation with A-P patterning
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .parameter_types import DiffusionParameters, SourceParameters
from .wnt_fgf_types import WNTGradientParameters, WNTSignalingType, RegionalMarker

logger = logging.getLogger(__name__)

class WNTSourceManager:
    """Source manager for WNT morphogen production."""
    
    def __init__(self, spatial_grid: SpatialGrid, source_params: SourceParameters):
        """Initialize WNT source manager."""
        self.grid = spatial_grid
        self.source_params = source_params
        self.source_regions: Dict[str, np.ndarray] = {}
        
    def configure_posterior_sources(self, neural_tube_dimensions: GridDimensions) -> None:
        """Configure WNT sources in posterior neural tube regions."""
        dims = self.grid.dimensions
        
        # Posterior neural tube (spinal cord region)
        posterior_start = int(0.6 * dims.y_size)  # Posterior 40% of neural tube
        
        # Create posterior WNT source
        posterior_source = np.zeros((dims.x_size, dims.y_size, dims.z_size))
        posterior_source[:, posterior_start:, :] = self.source_params.intensity
        
        self.source_regions['posterior'] = posterior_source
        self.grid.add_morphogen('WNT', initial_concentration=0.0)
        
        logger.info(f"Configured WNT posterior source: {np.sum(posterior_source > 0)} voxels")

class WNTDynamicsEngine:
    """Dynamics engine for WNT reaction-diffusion simulation."""
    
    def __init__(self, spatial_grid: SpatialGrid, source_manager: WNTSourceManager,
                 diffusion_params: DiffusionParameters, interactions: List):
        """Initialize WNT dynamics engine."""
        self.grid = spatial_grid
        self.source_manager = source_manager
        self.diffusion_params = diffusion_params
        self.interactions = interactions
        
    def simulate_timestep(self, dt: float) -> None:
        """Simulate one timestep of WNT dynamics."""
        # Get current WNT concentration
        wnt_field = self.grid.get_morphogen_concentration('WNT')
        
        # Apply diffusion
        diffusion_term = self._compute_diffusion(wnt_field, dt)
        
        # Apply degradation
        degradation_term = -self.diffusion_params.degradation_rate * wnt_field * dt
        
        # Apply production from sources
        production_term = np.zeros_like(wnt_field)
        for source_region in self.source_manager.source_regions.values():
            production_term += source_region * dt
        
        # Apply interactions with other morphogens
        interaction_term = self._compute_interactions(wnt_field, dt)
        
        # Update concentration
        new_concentration = wnt_field + diffusion_term + degradation_term + production_term + interaction_term
        new_concentration = np.maximum(new_concentration, 0.0)  # Non-negative concentrations
        
        self.grid.set_morphogen_concentration('WNT', new_concentration)
    
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
    
    def _compute_interactions(self, wnt_concentration: np.ndarray, dt: float) -> np.ndarray:
        """Compute interaction terms with other morphogens."""
        total_interaction = np.zeros_like(wnt_concentration)
        
        for interaction in self.interactions:
            if not self.grid.has_morphogen(interaction.target_morphogen):
                continue
            
            target_conc = self.grid.get_morphogen_concentration(interaction.target_morphogen)
            
            if interaction.interaction_type == 'activation':
                hill = self._calculate_hill_function(target_conc, interaction.threshold, interaction.hill_coefficient)
                total_interaction += interaction.strength * hill * dt
            
            elif interaction.interaction_type == 'inhibition':
                hill = self._calculate_hill_function(target_conc, interaction.threshold, interaction.hill_coefficient)
                total_interaction -= interaction.strength * hill * wnt_concentration * dt

        return total_interaction

    def _calculate_hill_function(self, concentration: np.ndarray, threshold: float,
                               hill_coefficient: float) -> np.ndarray:
        """Calculate Hill function for cooperative binding."""
        safe_concentration = np.maximum(concentration, 1e-10)
        numerator = safe_concentration ** hill_coefficient
        denominator = (threshold ** hill_coefficient) + numerator
        return numerator / denominator

class WNTGeneExpressionMapper:
    """Gene expression mapper for WNT signaling."""
    
    def __init__(self, spatial_grid: SpatialGrid):
        """Initialize WNT gene expression mapper."""
        self.grid = spatial_grid
        self.expression_thresholds = self._initialize_wnt_thresholds()
        
    def _initialize_wnt_thresholds(self) -> Dict[str, float]:
        """Initialize WNT target gene expression thresholds."""
        return {
            'Cdx2': 0.5,      # Posterior identity (nM)
            'Hoxb1': 0.3,     # Hindbrain specification
            'Gbx2': 0.4,      # Midbrain-hindbrain boundary
            'En1': 0.6,       # Midbrain specification
            'Otx2': 0.2,      # Forebrain specification (WNT inhibits)
        }
    
    def map_gene_expression(self, wnt_concentration: np.ndarray) -> Dict[str, np.ndarray]:
        """Map WNT concentration to target gene expression."""
        gene_expression = {}
        
        for gene, threshold in self.expression_thresholds.items():
            if gene == 'Otx2':
                # Otx2 is inhibited by WNT (anterior genes)
                expression = np.maximum(0.0, 1.0 - wnt_concentration / threshold)
            else:
                # Most WNT targets are activated
                expression = np.minimum(1.0, wnt_concentration / threshold)
            
            gene_expression[gene] = expression
        
        return gene_expression

class WNTGradientSystem:
    """Complete WNT gradient system coordinator.
    
    Coordinates WNT morphogen gradient components for anterior-posterior
    neural tube patterning during embryonic development.
    
    Key Components:
    - WNTSourceManager: Posterior neural tube sources
    - WNTDynamicsEngine: Reaction-diffusion simulation
    - WNTGeneExpressionMapper: A-P gene expression mapping
    """
    
    def __init__(self, spatial_grid: SpatialGrid, 
                 diffusion_params: DiffusionParameters,
                 source_params: SourceParameters,
                 interactions: List = None):
        """Initialize WNT gradient system.
        
        Args:
            spatial_grid: 3D spatial grid for concentration storage
            diffusion_params: WNT diffusion parameters
            source_params: WNT source parameters
            interactions: List of interaction parameters
        """
        self.grid = spatial_grid
        self.diffusion_params = diffusion_params
        self.source_params = source_params
        self.interactions = interactions or []
        
        # Initialize WNT concentration field
        self.grid.add_morphogen('WNT', initial_concentration=0.0)
        
        # Initialize component systems
        self.source_manager = WNTSourceManager(spatial_grid, source_params)
        self.dynamics_engine = WNTDynamicsEngine(
            spatial_grid, self.source_manager, diffusion_params, self.interactions
        )
        self.gene_expression = WNTGeneExpressionMapper(spatial_grid)
        
        logger.info("Initialized WNT gradient system coordinator")
        logger.info(f"WNT diffusion coefficient: {diffusion_params.diffusion_coefficient} µm²/s")
        logger.info(f"WNT degradation rate: {diffusion_params.degradation_rate} s⁻¹")
    
    def configure_sources(self, neural_tube_dimensions: GridDimensions) -> None:
        """Configure WNT source regions."""
        self.source_manager.configure_posterior_sources(neural_tube_dimensions)
        logger.info("WNT sources configured for posterior patterning")
    
    def simulate_time_step(self, dt: float) -> None:
        """Simulate one timestep of WNT gradient dynamics."""
        self.dynamics_engine.simulate_timestep(dt)
    
    def get_concentration_field(self) -> np.ndarray:
        """Get current WNT concentration field."""
        return self.grid.get_morphogen_concentration('WNT')
    
    def get_gene_expression_map(self) -> Dict[str, np.ndarray]:
        """Get WNT target gene expression map."""
        wnt_concentration = self.get_concentration_field()
        return self.gene_expression.map_gene_expression(wnt_concentration)
    
    def analyze_regional_specification(self) -> Dict[str, Any]:
        """Analyze regional specification along A-P axis."""
        wnt_field = self.get_concentration_field()
        gene_expression = self.get_gene_expression_map()
        
        # Analyze concentration along A-P axis (y-direction)
        dims = self.grid.dimensions
        ap_profile = np.mean(wnt_field, axis=(0, 2))  # Average over x,z
        
        # Identify regional boundaries based on WNT concentration
        regions = {}
        for i, concentration in enumerate(ap_profile):
            position = i / dims.y_size  # Normalized A-P position
            
            if concentration > 0.6:
                region = RegionalMarker.SPINAL_CORD
            elif concentration > 0.4:
                region = RegionalMarker.HINDBRAIN
            elif concentration > 0.2:
                region = RegionalMarker.MIDBRAIN
            else:
                region = RegionalMarker.FOREBRAIN
            
            if region not in regions:
                regions[region] = []
            regions[region].append(position)
        
        # Calculate region boundaries
        regional_analysis = {}
        for region, positions in regions.items():
            regional_analysis[region.value] = {
                "anterior_boundary": min(positions),
                "posterior_boundary": max(positions),
                "mean_position": np.mean(positions),
                "extent": max(positions) - min(positions)
            }
        
        return {
            "ap_concentration_profile": ap_profile.tolist(),
            "regional_boundaries": regional_analysis,
            "gene_expression_summary": {
                gene: {
                    "max_expression": float(np.max(expr)),
                    "mean_expression": float(np.mean(expr)),
                    "expression_domain_size": float(np.sum(expr > 0.1))
                }
                for gene, expr in gene_expression.items()
            }
        }
