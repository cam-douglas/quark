#!/usr/bin/env python3
"""CSF Flow Dynamics Framework.

Main coordinator for cerebrospinal fluid (CSF) flow dynamics with pressure
gradient mappings and velocity field computation for embryonic ventricular
system modeling.

Integration: Fluid dynamics component of spatial structure system
Rationale: Main flow dynamics coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .ventricular_topology import VentricularTopology
from .voxel_excavation import VoxelExcavationAlgorithm
from .csf_flow_types import FlowParameters, FlowBoundaryCondition
from .csf_boundary_manager import CSFBoundaryManager
from .csf_pressure_solver import CSFPressureSolver

logger = logging.getLogger(__name__)

class CSFFlowDynamics:
    """CSF flow dynamics framework for ventricular system.
    
    Main coordinator for cerebrospinal fluid flow dynamics including
    pressure field computation, velocity field calculation, and
    flow streamline analysis for embryonic brain development.
    """
    
    def __init__(self, spatial_grid: SpatialGrid,
                 ventricular_topology: VentricularTopology,
                 excavation_algorithm: VoxelExcavationAlgorithm,
                 flow_params: Optional[FlowParameters] = None):
        """Initialize CSF flow dynamics framework.
        
        Args:
            spatial_grid: 3D spatial grid for flow computation
            ventricular_topology: Ventricular topology system
            excavation_algorithm: Voxel excavation algorithm
            flow_params: Flow dynamics parameters
        """
        self.grid = spatial_grid
        self.topology = ventricular_topology
        self.excavation = excavation_algorithm
        self.params = flow_params or FlowParameters()
        
        # Initialize component systems
        self.boundary_manager = CSFBoundaryManager(spatial_grid, ventricular_topology, self.params)
        self.pressure_solver = CSFPressureSolver(spatial_grid.dimensions, self.params)
        
        # Flow computation state
        self.cavity_mask: Optional[np.ndarray] = None
        self.boundary_conditions: List[FlowBoundaryCondition] = []
        self.pressure_field: Optional[np.ndarray] = None
        self.velocity_field: Optional[np.ndarray] = None
        
        logger.info("Initialized CSFFlowDynamics framework")
        logger.info(f"Flow parameters: viscosity={self.params.viscosity_pa_s} Pa·s, "
                   f"density={self.params.density_kg_m3} kg/m³")
    
    def initialize_flow_domain(self, excavation_result) -> None:
        """Initialize flow domain from excavated ventricular cavities."""
        self.cavity_mask = excavation_result.excavated_mask
        
        # Initialize boundary conditions
        self.boundary_conditions = self.boundary_manager.setup_boundary_conditions(self.cavity_mask)
        
        # Initialize pressure and velocity fields
        dims = self.grid.dimensions
        self.pressure_field = np.zeros((dims.x_size, dims.y_size, dims.z_size))
        self.velocity_field = np.zeros((dims.x_size, dims.y_size, dims.z_size, 3))
        
        cavity_volume = np.sum(self.cavity_mask) * (dims.resolution ** 3) / 1e9  # mm³
        logger.info(f"Initialized flow domain: {cavity_volume:.3f} mm³ cavity volume")
        logger.info(f"Boundary conditions: {len(self.boundary_conditions)} conditions")
    
    def compute_pressure_field(self) -> np.ndarray:
        """Compute pressure field using finite difference method."""
        if self.cavity_mask is None:
            raise ValueError("Flow domain not initialized")
        
        self.pressure_field = self.pressure_solver.compute_pressure_field(
            self.boundary_conditions, self.cavity_mask)
        
        return self.pressure_field
    
    def compute_velocity_field(self) -> np.ndarray:
        """Compute velocity field from pressure gradients."""
        if self.pressure_field is None:
            self.compute_pressure_field()
        
        logger.info("Computing velocity field from pressure gradients")
        
        dims = self.grid.dimensions
        dx = dy = dz = dims.resolution * 1e-6  # Convert µm to m
        
        # Compute pressure gradients using central differences
        grad_p_x = np.zeros_like(self.pressure_field)
        grad_p_y = np.zeros_like(self.pressure_field)
        grad_p_z = np.zeros_like(self.pressure_field)
        
        # Central differences for interior points
        grad_p_x[1:-1, :, :] = (self.pressure_field[2:, :, :] - 
                               self.pressure_field[:-2, :, :]) / (2 * dx)
        grad_p_y[:, 1:-1, :] = (self.pressure_field[:, 2:, :] - 
                               self.pressure_field[:, :-2, :]) / (2 * dy)
        grad_p_z[:, :, 1:-1] = (self.pressure_field[:, :, 2:] - 
                               self.pressure_field[:, :, :-2]) / (2 * dz)
        
        # Darcy's law: v = -k/μ * ∇p (simplified for CSF flow)
        permeability = 1e-12  # m² (simplified permeability for CSF spaces)
        
        self.velocity_field = np.zeros((dims.x_size, dims.y_size, dims.z_size, 3))
        self.velocity_field[:, :, :, 0] = -permeability / self.params.viscosity_pa_s * grad_p_x
        self.velocity_field[:, :, :, 1] = -permeability / self.params.viscosity_pa_s * grad_p_y
        self.velocity_field[:, :, :, 2] = -permeability / self.params.viscosity_pa_s * grad_p_z
        
        # Mask velocity field to cavity regions only
        if self.cavity_mask is not None:
            for i in range(3):
                self.velocity_field[:, :, :, i][~self.cavity_mask] = 0.0
        
        # Calculate velocity magnitude statistics
        velocity_magnitude = np.sqrt(np.sum(self.velocity_field**2, axis=3))
        if self.cavity_mask is not None:
            cavity_velocities = velocity_magnitude[self.cavity_mask]
            if len(cavity_velocities) > 0:
                v_stats = (np.min(cavity_velocities), np.mean(cavity_velocities), 
                          np.max(cavity_velocities))
                logger.info(f"Velocity field: min={v_stats[0]:.2e}, "
                           f"mean={v_stats[1]:.2e}, max={v_stats[2]:.2e} m/s")
        
        return self.velocity_field
    
    def compute_flow_streamlines(self, n_streamlines: int = 10) -> List[np.ndarray]:
        """Compute CSF flow streamlines."""
        if self.velocity_field is None:
            self.compute_velocity_field()
        
        logger.info(f"Computing {n_streamlines} CSF flow streamlines")
        
        streamlines = []
        
        # Find starting points for streamlines (production sites)
        start_points = []
        for bc in self.boundary_conditions:
            if bc.boundary_type.value == "production":
                start_points.append(bc.location)
        
        # If not enough production sites, add random points in cavities
        if len(start_points) < n_streamlines and self.cavity_mask is not None:
            cavity_indices = np.where(self.cavity_mask)
            n_random = min(n_streamlines - len(start_points), len(cavity_indices[0]))
            
            random_indices = np.random.choice(len(cavity_indices[0]), n_random, replace=False)
            for idx in random_indices:
                start_points.append((cavity_indices[0][idx], cavity_indices[1][idx], 
                                   cavity_indices[2][idx]))
        
        # Integrate streamlines
        for i, start_point in enumerate(start_points[:n_streamlines]):
            streamline = self._integrate_streamline(start_point)
            if len(streamline) > 1:
                streamlines.append(streamline)
                logger.debug(f"Streamline {i+1}: {len(streamline)} points")
        
        logger.info(f"Computed {len(streamlines)} valid streamlines")
        return streamlines
    
    def _integrate_streamline(self, start_point: Tuple[int, int, int], 
                            max_steps: int = 1000) -> np.ndarray:
        """Integrate single streamline from starting point."""
        if self.velocity_field is None:
            return np.array([])
        
        dims = self.grid.dimensions
        streamline_points = [start_point]
        
        current_pos = np.array(start_point, dtype=float)
        dt = 0.1  # Integration time step (arbitrary units)
        
        for step in range(max_steps):
            # Get current position (with bounds checking)
            x, y, z = [int(np.clip(pos, 0, size-1)) for pos, size in 
                      zip(current_pos, [dims.x_size, dims.y_size, dims.z_size])]
            
            # Check if still in cavity
            if self.cavity_mask is not None and not self.cavity_mask[x, y, z]:
                break
            
            # Get velocity at current position
            velocity = self.velocity_field[x, y, z, :]
            velocity_magnitude = np.linalg.norm(velocity)
            
            # Stop if velocity is too small
            if velocity_magnitude < 1e-12:
                break
            
            # Update position using Euler integration
            current_pos += velocity * dt
            
            # Check bounds
            if (current_pos[0] < 0 or current_pos[0] >= dims.x_size or
                current_pos[1] < 0 or current_pos[1] >= dims.y_size or
                current_pos[2] < 0 or current_pos[2] >= dims.z_size):
                break
            
            streamline_points.append(tuple(current_pos))
        
        return np.array(streamline_points)
    
    def validate_mass_conservation(self) -> float:
        """Validate mass conservation in flow field."""
        if self.velocity_field is None:
            return float('inf')
        
        logger.info("Validating mass conservation")
        
        dims = self.grid.dimensions
        dx = dy = dz = dims.resolution * 1e-6  # Convert µm to m
        
        # Compute divergence of velocity field
        div_v = np.zeros((dims.x_size, dims.y_size, dims.z_size))
        
        # Central differences for interior points
        div_v[1:-1, :, :] += (self.velocity_field[2:, :, :, 0] - 
                             self.velocity_field[:-2, :, :, 0]) / (2 * dx)
        div_v[:, 1:-1, :] += (self.velocity_field[:, 2:, :, 1] - 
                             self.velocity_field[:, :-2, :, 1]) / (2 * dy)
        div_v[:, :, 1:-1] += (self.velocity_field[:, :, 2:, 2] - 
                             self.velocity_field[:, :, :-2, 2]) / (2 * dz)
        
        # Calculate mass conservation error (∇·v should be 0 for incompressible flow)
        if self.cavity_mask is not None:
            cavity_divergence = div_v[self.cavity_mask]
            mass_error = np.sqrt(np.mean(cavity_divergence**2))
        else:
            mass_error = np.sqrt(np.mean(div_v**2))
        
        logger.info(f"Mass conservation error: {mass_error:.2e} s⁻¹")
        
        return mass_error
    
    def export_flow_analysis(self) -> Dict[str, Any]:
        """Export comprehensive flow analysis."""
        if self.cavity_mask is None:
            logger.warning("Flow domain not initialized")
            return {}
        
        # Ensure all fields are computed
        if self.pressure_field is None:
            self.compute_pressure_field()
        if self.velocity_field is None:
            self.compute_velocity_field()
        
        streamlines = self.compute_flow_streamlines()
        mass_error = self.validate_mass_conservation()
        
        # Calculate flow statistics
        velocity_magnitude = np.sqrt(np.sum(self.velocity_field**2, axis=3))
        cavity_velocities = velocity_magnitude[self.cavity_mask]
        cavity_pressures = self.pressure_field[self.cavity_mask]
        
        flow_analysis = {
            "flow_parameters": {
                "viscosity_pa_s": self.params.viscosity_pa_s,
                "density_kg_m3": self.params.density_kg_m3,
                "production_rate_ml_min": self.params.production_rate_ml_min,
                "absorption_rate_ml_min": self.params.absorption_rate_ml_min
            },
            "pressure_statistics": {
                "min_pa": float(np.min(cavity_pressures)) if len(cavity_pressures) > 0 else 0.0,
                "max_pa": float(np.max(cavity_pressures)) if len(cavity_pressures) > 0 else 0.0,
                "mean_pa": float(np.mean(cavity_pressures)) if len(cavity_pressures) > 0 else 0.0,
                "std_pa": float(np.std(cavity_pressures)) if len(cavity_pressures) > 0 else 0.0
            },
            "velocity_statistics": {
                "min_m_s": float(np.min(cavity_velocities)) if len(cavity_velocities) > 0 else 0.0,
                "max_m_s": float(np.max(cavity_velocities)) if len(cavity_velocities) > 0 else 0.0,
                "mean_m_s": float(np.mean(cavity_velocities)) if len(cavity_velocities) > 0 else 0.0,
                "std_m_s": float(np.std(cavity_velocities)) if len(cavity_velocities) > 0 else 0.0
            },
            "streamlines": {
                "count": len(streamlines),
                "average_length": float(np.mean([len(s) for s in streamlines])) if streamlines else 0.0,
                "total_points": sum(len(s) for s in streamlines)
            },
            "validation": {
                "mass_conservation_error": float(mass_error),
                "boundary_conditions_count": len(self.boundary_conditions),
                "cavity_volume_mm3": float(np.sum(self.cavity_mask) * 
                                          (self.grid.dimensions.resolution ** 3) / 1e9)
            }
        }
        
        return flow_analysis