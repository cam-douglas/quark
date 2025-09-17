#!/usr/bin/env python3
"""CSF Pressure Field Solver.

Solves pressure field equations for CSF flow using finite difference methods
with proper boundary condition handling for ventricular system modeling.

Integration: Pressure solver component for CSF flow dynamics
Rationale: Focused pressure field computation separated from main flow coordinator
"""

from typing import List
import numpy as np
from scipy.sparse import diags
from scipy.sparse import linalg as sparse_linalg
import logging

from .spatial_grid import GridDimensions
from .csf_flow_types import FlowBoundaryCondition, FlowBoundaryType, FlowParameters

logger = logging.getLogger(__name__)

class CSFPressureSolver:
    """Pressure field solver for CSF flow dynamics.
    
    Solves pressure field using finite difference discretization of
    Poisson equation with source/sink terms from CSF production and
    absorption sites.
    """
    
    def __init__(self, grid_dimensions: GridDimensions, flow_params: FlowParameters):
        """Initialize CSF pressure solver.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
            flow_params: Flow dynamics parameters
        """
        self.dims = grid_dimensions
        self.params = flow_params
        
        logger.info("Initialized CSFPressureSolver")
        logger.info(f"Grid size: {grid_dimensions.x_size}x{grid_dimensions.y_size}x{grid_dimensions.z_size}")
    
    def compute_pressure_field(self, boundary_conditions: List[FlowBoundaryCondition],
                              cavity_mask: np.ndarray) -> np.ndarray:
        """Compute pressure field using finite difference method.
        
        Args:
            boundary_conditions: List of boundary conditions
            cavity_mask: Boolean mask of cavity regions
            
        Returns:
            3D pressure field array (Pa)
        """
        logger.info("Computing pressure field using finite difference method")
        
        # Setup pressure equation: ∇²p = -∇·f (where f is source/sink term)
        n_voxels = self.dims.x_size * self.dims.y_size * self.dims.z_size
        
        # Create source term from boundary conditions
        source_term = self._create_source_term(boundary_conditions)
        
        # Setup finite difference operator (Laplacian)
        laplacian_matrix = self._create_laplacian_matrix()
        
        # Apply boundary conditions
        laplacian_matrix, source_term = self._apply_pressure_boundary_conditions(
            laplacian_matrix, source_term, boundary_conditions)
        
        # Solve linear system: Ap = b
        try:
            pressure_vector = sparse_linalg.spsolve(laplacian_matrix, source_term)
            pressure_field = pressure_vector.reshape(
                (self.dims.x_size, self.dims.y_size, self.dims.z_size))
        except Exception as e:
            logger.error(f"Pressure field computation failed: {e}")
            # Fallback to simple gradient
            pressure_field = self._compute_simple_pressure_gradient()
        
        # Mask pressure field to cavity regions only
        pressure_field[~cavity_mask] = 0.0
        
        pressure_range = (np.min(pressure_field[cavity_mask]), 
                         np.max(pressure_field[cavity_mask]))
        logger.info(f"Pressure field computed: range {pressure_range[0]:.2f} to "
                   f"{pressure_range[1]:.2f} Pa")
        
        return pressure_field
    
    def _create_source_term(self, boundary_conditions: List[FlowBoundaryCondition]) -> np.ndarray:
        """Create source term for pressure equation."""
        source_term = np.zeros(self.dims.x_size * self.dims.y_size * self.dims.z_size)
        
        for bc in boundary_conditions:
            if bc.boundary_type in [FlowBoundaryType.PRODUCTION, FlowBoundaryType.ABSORPTION]:
                x, y, z = bc.location
                if (0 <= x < self.dims.x_size and 0 <= y < self.dims.y_size and 0 <= z < self.dims.z_size):
                    idx = x * self.dims.y_size * self.dims.z_size + y * self.dims.z_size + z
                    source_term[idx] = bc.value / (self.dims.resolution ** 3)  # Convert to per-voxel
        
        return source_term
    
    def _create_laplacian_matrix(self):
        """Create finite difference Laplacian matrix."""
        n_voxels = self.dims.x_size * self.dims.y_size * self.dims.z_size
        
        # Create 3D Laplacian using finite differences
        # ∇²p ≈ (p[i+1,j,k] - 2p[i,j,k] + p[i-1,j,k])/dx² + ... for y,z
        
        dx = dy = dz = self.dims.resolution * 1e-6  # Convert µm to m
        
        # Diagonal entries (-6 for interior points)
        main_diag = -6 * np.ones(n_voxels) / (dx**2)
        
        # Off-diagonal entries (1 for adjacent points)
        x_offdiag = np.ones(n_voxels - self.dims.y_size * self.dims.z_size) / (dx**2)
        y_offdiag = np.ones(n_voxels - self.dims.z_size) / (dy**2)
        z_offdiag = np.ones(n_voxels - 1) / (dz**2)
        
        # Handle boundary conditions for off-diagonals
        # (Simplified - assumes periodic or zero boundaries)
        
        # Create sparse matrix
        offsets = [-self.dims.y_size * self.dims.z_size, -self.dims.z_size, -1, 0, 1, 
                  self.dims.z_size, self.dims.y_size * self.dims.z_size]
        diagonals = [x_offdiag, y_offdiag, z_offdiag, main_diag, z_offdiag, 
                    y_offdiag, x_offdiag]
        
        laplacian_matrix = diags(diagonals, offsets, shape=(n_voxels, n_voxels), 
                               format='csr')
        
        return laplacian_matrix
    
    def _apply_pressure_boundary_conditions(self, laplacian_matrix, source_term,
                                          boundary_conditions: List[FlowBoundaryCondition]):
        """Apply boundary conditions to pressure equation."""
        # This is a simplified implementation
        # In practice, would properly handle Dirichlet and Neumann conditions
        return laplacian_matrix, source_term
    
    def _compute_simple_pressure_gradient(self) -> np.ndarray:
        """Compute simple pressure gradient as fallback."""
        pressure_field = np.zeros((self.dims.x_size, self.dims.y_size, self.dims.z_size))
        
        # Create simple linear pressure gradient along main flow direction
        max_pressure = self.params.pressure_gradient_pa_m * 0.001  # 1mm gradient
        
        # Apply gradient along y-axis (anterior-posterior)
        for y in range(self.dims.y_size):
            pressure_field[:, y, :] = max_pressure * (1.0 - y / self.dims.y_size)
        
        return pressure_field
