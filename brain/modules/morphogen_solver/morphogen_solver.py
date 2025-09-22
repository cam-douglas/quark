#!/usr/bin/env python3
"""Main Morphogen Solver - Foundation Layer Implementation.

Orchestrates the complete morphogen gradient solver system for neural tube
patterning during embryonic development (weeks 3-8).

Integration: Main entry point for foundation layer morphogen systems
Rationale: Coordinates all morphogen systems and provides unified interface
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .biological_parameters import BiologicalParameters
from .shh_gradient_system import SHHGradientSystem
from .bmp_gradient_system import BMPGradientSystem
from .wnt_gradient_system import WNTGradientSystem
from .fgf_gradient_system import FGFGradientSystem
from .parameter_types import DiffusionParameters, SourceParameters

logger = logging.getLogger(__name__)

class MorphogenSolver:
    """Main morphogen gradient solver for neural tube patterning.
    
    Coordinates multiple morphogen systems (SHH, BMP, WNT, FGF) during
    embryonic neural tube development with biologically-accurate parameters.
    
    Key Features:
    - Multi-morphogen gradient simulation
    - Cross-regulation interactions
    - Biologically-accurate parameters
    - Real-time visualization support
    - Validation against experimental data
    """
    
    def __init__(self, grid_dimensions: GridDimensions, 
                 species: str = "mouse", stage: str = "E8.5-E10.5"):
        """Initialize morphogen solver.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
            species: Model organism
            stage: Developmental stage
        """
        self.grid_dimensions = grid_dimensions
        self.species = species
        self.stage = stage
        
        # Initialize core components
        self.spatial_grid = SpatialGrid(grid_dimensions)
        self.bio_params = BiologicalParameters(species, stage)
        
        # Morphogen systems
        self.morphogen_systems: Dict[str, Any] = {}
        
        # Initialize morphogen systems
        self._initialize_shh_system()
        self._initialize_bmp_system()
        self._initialize_wnt_system()
        self._initialize_fgf_system()
        
        # Simulation state
        self.current_time = 0.0  # seconds
        self.is_configured = False
        
        logger.info("Initialized MorphogenSolver")
        logger.info(f"Grid: {grid_dimensions.x_size}x{grid_dimensions.y_size}x{grid_dimensions.z_size}")
        logger.info(f"Resolution: {grid_dimensions.resolution} µm")
        logger.info(f"Species: {species}, Stage: {stage}")
    
    def _initialize_shh_system(self) -> None:
        """Initialize SHH gradient system."""
        # Get SHH parameters
        shh_diffusion = self.bio_params.get_diffusion_parameters('SHH')
        shh_source = self.bio_params.get_source_parameters('SHH')
        shh_interactions = self.bio_params.get_interaction_parameters('SHH')
        
        # Create SHH system
        self.morphogen_systems['SHH'] = SHHGradientSystem(
            self.spatial_grid,
            shh_diffusion,
            shh_source,
            shh_interactions
        )
        
        logger.info("Initialized SHH gradient system")
    
    def _initialize_bmp_system(self) -> None:
        """Initialize BMP gradient system."""
        # Get BMP parameters
        bmp_diffusion = self.bio_params.get_diffusion_parameters('BMP')
        bmp_source = self.bio_params.get_source_parameters('BMP')
        bmp_interactions = self.bio_params.get_interaction_parameters('BMP')
        
        # Create BMP system
        self.morphogen_systems['BMP'] = BMPGradientSystem(
            self.spatial_grid,
            bmp_diffusion,
            bmp_source,
            bmp_interactions
        )
        
        logger.info("Initialized BMP gradient system")

    def _initialize_wnt_system(self) -> None:
        """Initialize WNT gradient system."""
        if 'WNT' in self.bio_params.get_all_morphogens():
            wnt_diffusion = self.bio_params.get_diffusion_parameters('WNT')
            wnt_source = self.bio_params.get_source_parameters('WNT')
            wnt_interactions = self.bio_params.get_interaction_parameters('WNT')
            
            self.morphogen_systems['WNT'] = WNTGradientSystem(
                self.spatial_grid, wnt_diffusion, wnt_source, wnt_interactions
            )
            logger.info("Initialized WNT gradient system")

    def _initialize_fgf_system(self) -> None:
        """Initialize FGF gradient system."""
        if 'FGF' in self.bio_params.get_all_morphogens():
            fgf_diffusion = self.bio_params.get_diffusion_parameters('FGF')
            fgf_source = self.bio_params.get_source_parameters('FGF')
            fgf_interactions = self.bio_params.get_interaction_parameters('FGF')
            
            self.morphogen_systems['FGF'] = FGFGradientSystem(
                self.spatial_grid, fgf_diffusion, fgf_source, fgf_interactions
            )
            logger.info("Initialized FGF gradient system")
    
    def configure_neural_tube(self, neural_tube_length: float = 500.0,
                            neural_tube_height: float = 200.0,
                            neural_tube_width: float = 300.0) -> None:
        """Configure neural tube geometry and source regions.
        
        Args:
            neural_tube_length: A-P length (µm)
            neural_tube_height: D-V height (µm) 
            neural_tube_width: L-R width (µm)
        """
        # Calculate grid dimensions based on neural tube size
        x_size = int(neural_tube_length / self.grid_dimensions.resolution)
        y_size = int(neural_tube_height / self.grid_dimensions.resolution)
        z_size = int(neural_tube_width / self.grid_dimensions.resolution)
        
        # Update grid dimensions
        neural_tube_dims = GridDimensions(
            x_size=x_size,
            y_size=y_size,
            z_size=z_size,
            resolution=self.grid_dimensions.resolution
        )
        
        # Configure morphogen sources
        if 'SHH' in self.morphogen_systems:
            self.morphogen_systems['SHH'].configure_sources(neural_tube_dims)
        if 'BMP' in self.morphogen_systems:
            self.morphogen_systems['BMP'].configure_sources(neural_tube_dims)
        if 'WNT' in self.morphogen_systems:
            self.morphogen_systems['WNT'].configure_sources(neural_tube_dims)
        if 'FGF' in self.morphogen_systems:
            self.morphogen_systems['FGF'].configure_sources(neural_tube_dims)
        
        self.is_configured = True
        
        logger.info(f"Configured neural tube: {neural_tube_length}x{neural_tube_height}x{neural_tube_width} µm")
        logger.info(f"Grid size: {x_size}x{y_size}x{z_size} voxels")
    
    def generate_shh_gradient(self, simulation_hours: float = 24.0,
                            progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Generate SHH dorsal-ventral concentration gradient.
        
        Args:
            simulation_hours: Simulation duration in hours
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary of simulation results
        """
        if not self.is_configured:
            raise ValueError("Neural tube not configured. Call configure_neural_tube() first.")
        
        if 'SHH' not in self.morphogen_systems:
            raise ValueError("SHH system not initialized")
        
        logger.info(f"Starting SHH gradient generation for {simulation_hours} hours")
        
        # Run SHH simulation
        shh_system = self.morphogen_systems['SHH']
        results = shh_system.run_simulation(simulation_hours, progress_callback)
        
        # Get final gradient properties
        gradient_validation = shh_system.validate_gradient_formation()
        
        # Calculate gradient metrics
        y_coords, shh_profile = shh_system.get_dorsal_ventral_profile()
        
        results.update({
            "gradient_validation": gradient_validation,
            "dorsal_ventral_profile": {
                "y_coordinates_um": y_coords.tolist(),
                "shh_concentrations_nM": shh_profile.tolist()
            },
            "gradient_properties": {
                "max_concentration": float(np.max(shh_profile)),
                "min_concentration": float(np.min(shh_profile)),
                "gradient_range": float(np.max(shh_profile) - np.min(shh_profile)),
                "ventral_concentration": float(shh_profile[-1]),  # Bottom of neural tube
                "dorsal_concentration": float(shh_profile[0])     # Top of neural tube
            }
        })
        
        logger.info("Completed SHH gradient generation")
        logger.info(f"Gradient range: {results['gradient_properties']['gradient_range']:.2f} nM")
        
        return results
    
    def get_shh_concentration_map(self) -> np.ndarray:
        """Get current SHH concentration map.
        
        Returns:
            3D SHH concentration array (nM)
        """
        if 'SHH' not in self.spatial_grid.concentrations:
            raise ValueError("SHH concentrations not available")
        
        return self.spatial_grid.concentrations['SHH'].copy()
    
    def get_dorsal_ventral_gradient(self, morphogen: str = 'SHH',
                                  x_pos: Optional[int] = None,
                                  z_pos: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get dorsal-ventral morphogen gradient profile.
        
        Args:
            morphogen: Morphogen name
            x_pos: A-P position (center if None)
            z_pos: L-R position (center if None)
            
        Returns:
            Tuple of (y_coordinates, concentrations)
        """
        if morphogen not in self.spatial_grid.concentrations:
            raise ValueError(f"{morphogen} concentrations not available")
        
        if x_pos is None:
            x_pos = self.grid_dimensions.x_size // 2
        if z_pos is None:
            z_pos = self.grid_dimensions.z_size // 2
        
        concentrations = self.spatial_grid.concentrations[morphogen][x_pos, :, z_pos]
        y_coords = np.arange(len(concentrations)) * self.grid_dimensions.resolution
        
        return y_coords, concentrations

    def override_diffusion_parameters(self, morphogen: str, new_diffusion_params: DiffusionParameters):
        """Overrides the diffusion parameters for a specific morphogen and re-initializes its system."""
        if morphogen not in self.bio_params.get_all_morphogens():
            raise ValueError(f"Unknown morphogen: {morphogen}")

        # Update the parameters in the biological parameters database
        param_set = self.bio_params.morphogen_db.get_parameter_set(morphogen)
        param_set.diffusion = new_diffusion_params
        self.bio_params.morphogen_db.add_parameter_set(param_set)

        # Re-initialize the specific morphogen system to use the new parameters
        if morphogen == 'SHH':
            self._initialize_shh_system()
        elif morphogen == 'BMP':
            self._initialize_bmp_system()
        elif morphogen == 'WNT':
            self._initialize_wnt_system()
        elif morphogen == 'FGF':
            self._initialize_fgf_system()
            
        logger.info(f"Overrode diffusion parameters and re-initialized system for {morphogen}")
    
    def simulate_morphogen_dynamics(self, total_time: float, dt: float):
        """Simulate all morphogen dynamics for a total duration."""
        num_steps = int(total_time / dt)
        for i in range(num_steps):
            self.simulate_time_step(dt)
            if (i * dt) % 300 == 0: # Log every 5 minutes
                logger.debug(f"Simulation progress: {self.current_time_hours:.2f} hours")
    
    def simulate_time_step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Simulate one time step for all morphogen systems.
        
        Args:
            dt: Time step size (seconds)
            
        Returns:
            Dictionary of simulation metrics
        """
        metrics = {}
        
        # Simulate each morphogen system
        for morphogen_name, system in self.morphogen_systems.items():
            try:
                step_metrics = system.simulate_time_step(dt)
                metrics[morphogen_name] = step_metrics
            except Exception as e:
                logger.error(f"Simulation failed for {morphogen_name}: {e}")
                metrics[morphogen_name] = {"error": str(e)}
        
        # Update global time
        if metrics and 'SHH' in metrics and 'simulation_time_s' in metrics['SHH']:
            self.current_time = metrics['SHH']['simulation_time_s']
        
        return metrics
    
    def validate_gradients(self) -> Dict[str, Any]:
        """Validate all morphogen gradients.
        
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        for morphogen_name, system in self.morphogen_systems.items():
            try:
                if hasattr(system, 'validate_gradient_formation'):
                    validation = system.validate_gradient_formation()
                    validation_results[morphogen_name] = validation
                else:
                    validation_results[morphogen_name] = {"status": "validation_not_available"}
            except Exception as e:
                logger.error(f"Validation failed for {morphogen_name}: {e}")
                validation_results[morphogen_name] = {"error": str(e)}
        
        return validation_results
    
    def get_solver_status(self) -> Dict[str, Any]:
        """Get comprehensive solver status.
        
        Returns:
            Dictionary of solver status
        """
        return {
            "solver_info": {
                "species": self.species,
                "developmental_stage": self.stage,
                "current_time_s": self.current_time,
                "current_time_hours": self.current_time / 3600.0,
                "is_configured": self.is_configured
            },
            "grid_info": self.spatial_grid.get_grid_info(),
            "morphogen_systems": {
                name: system.get_comprehensive_status() if hasattr(system, 'get_comprehensive_status')
                else {"status": "active"}
                for name, system in self.morphogen_systems.items()
            },
            "biological_parameters": self.bio_params.get_parameter_summary(),
            "validation": self.validate_gradients()
        }
    
    def reset_solver(self) -> None:
        """Reset solver to initial state."""
        # Reset all morphogen systems
        for system in self.morphogen_systems.values():
            if hasattr(system, 'reset_system'):
                system.reset_system()
        
        # Reset time
        self.current_time = 0.0
        
        # Reset configuration
        self.is_configured = False
        
        logger.info("Reset morphogen solver to initial state")
    
    @property
    def current_time_hours(self) -> float:
        """Get current simulation time in hours."""
        return self.current_time / 3600.0
