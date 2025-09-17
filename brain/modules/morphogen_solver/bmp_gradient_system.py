#!/usr/bin/env python3
"""BMP (Bone Morphogenetic Protein) Gradient System Coordinator.

Coordinates BMP morphogen gradient system components for dorsal neural tube
patterning during embryonic development (weeks 3-8).

Key Features:
- Integrates source management, dynamics, and gene expression
- Provides unified interface for BMP system
- Coordinates simulation and validation
- Implements SHH antagonism interactions

Integration: Core component of foundation layer morphogen solver
Rationale: BMP is the primary morphogen for dorsal neural tube specification
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .parameter_types import DiffusionParameters, SourceParameters
from .bmp_source_manager import BMPSourceManager
from .bmp_dynamics_engine import BMPDynamicsEngine  
from .bmp_gene_expression import BMPGeneExpressionMapper

logger = logging.getLogger(__name__)

class BMPGradientSystem:
    """Complete BMP gradient system coordinator.
    
    Coordinates BMP morphogen gradient components including source management,
    reaction-diffusion dynamics, and gene expression mapping for dorsal neural
    tube specification during embryonic development.
    
    Key Components:
    - BMPSourceManager: Roof plate and dorsal ectoderm source regions
    - BMPDynamicsEngine: Reaction-diffusion simulation with SHH antagonism
    - BMPGeneExpressionMapper: Gene expression threshold mapping for dorsal fates
    """
    
    def __init__(self, spatial_grid: SpatialGrid, 
                 diffusion_params: DiffusionParameters,
                 source_params: SourceParameters,
                 interactions: List = None):
        """Initialize BMP gradient system coordinator.
        
        Args:
            spatial_grid: 3D spatial grid for concentration storage
            diffusion_params: BMP diffusion parameters
            source_params: BMP source parameters
            interactions: List of interaction parameters
        """
        self.grid = spatial_grid
        self.diffusion_params = diffusion_params
        self.source_params = source_params
        self.interactions = interactions or []
        
        # Initialize BMP concentration field
        self.grid.add_morphogen('BMP', initial_concentration=0.0)
        
        # Initialize component systems
        self.source_manager = BMPSourceManager(spatial_grid, source_params)
        self.dynamics_engine = BMPDynamicsEngine(
            spatial_grid, self.source_manager, diffusion_params, self.interactions
        )
        self.gene_expression = BMPGeneExpressionMapper(spatial_grid)
        
        logger.info("Initialized BMP gradient system coordinator")
        logger.info(f"Components: SourceManager, DynamicsEngine, GeneExpression")
        logger.info(f"BMP diffusion coefficient: {diffusion_params.diffusion_coefficient} µm²/s")
        logger.info(f"BMP degradation rate: {diffusion_params.degradation_rate} s⁻¹")
    
    def configure_sources(self, neural_tube_dimensions: GridDimensions) -> None:
        """Configure BMP source regions based on neural tube geometry.
        
        Args:
            neural_tube_dimensions: Neural tube spatial dimensions
        """
        self.source_manager.configure_sources(neural_tube_dimensions)
        logger.info("Configured BMP source regions")
    
    def simulate_time_step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Simulate one time step of BMP gradient dynamics.
        
        Args:
            dt: Time step size (seconds). Uses adaptive if None.
            
        Returns:
            Dictionary of simulation metrics
        """
        return self.dynamics_engine.simulate_time_step(dt)
    
    def run_simulation(self, duration_hours: float, 
                      progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Run complete BMP gradient simulation.
        
        Args:
            duration_hours: Simulation duration in hours
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of simulation results
        """
        duration_seconds = duration_hours * 3600.0
        target_time = self.dynamics_engine.current_time + duration_seconds
        
        results = {
            "start_time": self.dynamics_engine.current_time,
            "target_time": target_time,
            "time_steps": 0,
            "final_metrics": {}
        }
        
        logger.info(f"Starting BMP simulation for {duration_hours:.1f} hours")
        
        while self.dynamics_engine.current_time < target_time:
            # Simulate time step
            step_metrics = self.simulate_time_step()
            results["time_steps"] += 1
            
            # Progress callback
            if progress_callback and results["time_steps"] % 100 == 0:
                progress = (self.dynamics_engine.current_time - results["start_time"]) / duration_seconds
                progress_callback(progress, step_metrics)
            
            # Check stability
            stability = self.dynamics_engine.check_stability()
            if not stability["is_stable"]:
                logger.error("BMP simulation became unstable, stopping")
                results["error"] = "Simulation instability"
                break
        
        results["final_metrics"] = self.get_comprehensive_status()
        logger.info(f"Completed BMP simulation: {results['time_steps']} time steps")
        
        return results
    
    def get_gene_expression_map(self, gene_name: str, 
                               expression_mode: str = 'binary') -> np.ndarray:
        """Get gene expression map based on current BMP concentrations.
        
        Args:
            gene_name: Target gene name
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Expression map array (0-1 values)
        """
        return self.gene_expression.get_gene_expression_map(gene_name, expression_mode)
    
    def get_dorsal_ventral_profile(self, x_pos: Optional[int] = None, 
                                  z_pos: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get dorsal-ventral BMP concentration profile.
        
        Args:
            x_pos: A-P position (uses center if None)
            z_pos: L-R position (uses center if None)
            
        Returns:
            Tuple of (y_coordinates, concentrations)
        """
        if x_pos is None:
            x_pos = self.grid.dimensions.x_size // 2
        if z_pos is None:
            z_pos = self.grid.dimensions.z_size // 2
        
        concentrations = self.grid.concentrations['BMP'][x_pos, :, z_pos]
        y_coords = np.arange(len(concentrations)) * self.grid.resolution
        
        return y_coords, concentrations
    
    def validate_gradient_formation(self) -> Dict[str, Any]:
        """Validate BMP gradient formation and properties.
        
        Returns:
            Dictionary of validation metrics
        """
        bmp_conc = self.grid.concentrations['BMP']
        
        # Basic statistics
        max_conc = np.max(bmp_conc)
        min_conc = np.min(bmp_conc)
        mean_conc = np.mean(bmp_conc)
        
        # Gradient properties
        grad_x, grad_y, grad_z = self.grid.get_gradient('BMP')
        max_gradient = np.max(np.sqrt(grad_x**2 + grad_y**2 + grad_z**2))
        
        # Check for proper D-V gradient (should be strongest in Y direction, opposite to SHH)
        mean_grad_y = np.mean(np.abs(grad_y))
        mean_grad_x = np.mean(np.abs(grad_x))
        mean_grad_z = np.mean(np.abs(grad_z))
        
        # Component validations
        source_validation = self.source_manager.validate_sources()
        stability_check = self.dynamics_engine.check_stability()
        expression_validation = self.gene_expression.validate_expression_patterns()
        
        return {
            "concentration_stats": {
                "max_concentration_nM": float(max_conc),
                "min_concentration_nM": float(min_conc), 
                "mean_concentration_nM": float(mean_conc),
                "concentration_range_nM": float(max_conc - min_conc)
            },
            "gradient_properties": {
                "max_gradient_magnitude": float(max_gradient),
                "mean_dorsal_ventral_gradient": float(mean_grad_y),
                "mean_anterior_posterior_gradient": float(mean_grad_x),
                "mean_left_right_gradient": float(mean_grad_z),
                "dv_gradient_dominance": float(mean_grad_y / (mean_grad_x + mean_grad_z + 1e-10))
            },
            "component_validations": {
                "sources": source_validation,
                "dynamics": stability_check,
                "gene_expression": expression_validation
            },
            "simulation_time_s": self.dynamics_engine.current_time
        }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive BMP system status.
        
        Returns:
            Dictionary of complete system status
        """
        return {
            "system_info": {
                "morphogen": "BMP",
                "status": "active",
                "simulation_time_s": self.dynamics_engine.current_time
            },
            "components": {
                "source_manager": self.source_manager.get_source_statistics(),
                "dynamics_engine": self.dynamics_engine.get_dynamics_summary(),
                "gene_expression": {
                    "available_genes": self.gene_expression.get_available_genes(),
                    "gene_count": len(self.gene_expression.get_available_genes())
                }
            },
            "validation": self.validate_gradient_formation(),
            "parameters": {
                "diffusion_coefficient": self.diffusion_params.diffusion_coefficient,
                "degradation_rate": self.diffusion_params.degradation_rate,
                "half_life_min": self.diffusion_params.half_life / 60.0,
                "source_intensity": self.source_params.intensity
            }
        }
    
    def reset_system(self) -> None:
        """Reset entire BMP gradient system."""
        self.source_manager.reset_sources()
        self.dynamics_engine.reset_simulation()
        self.gene_expression.clear_expression_cache()
        
        logger.info("Reset BMP gradient system")
    
    def get_available_genes(self) -> List[str]:
        """Get list of available genes for expression analysis."""
        return self.gene_expression.get_available_genes()
    
    @property
    def current_time_hours(self) -> float:
        """Get current simulation time in hours."""
        return self.dynamics_engine.current_time / 3600.0
