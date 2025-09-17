#!/usr/bin/env python3
"""CSF Flow Types and Parameters.

Type definitions and parameter structures for CSF flow dynamics including
boundary conditions, flow parameters, and result data structures.

Integration: Type definitions for CSF flow dynamics system
Rationale: Centralized type and parameter definitions
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

class FlowBoundaryType(Enum):
    """Types of flow boundary conditions."""
    PRODUCTION = "production"      # CSF production sites
    ABSORPTION = "absorption"      # CSF absorption sites
    NO_FLUX = "no_flux"           # Impermeable boundaries
    PRESSURE = "pressure"         # Fixed pressure boundaries

@dataclass
class FlowParameters:
    """Parameters for CSF flow dynamics."""
    viscosity_pa_s: float = 1.0e-3        # CSF viscosity (Pa·s)
    density_kg_m3: float = 1000.0         # CSF density (kg/m³)
    production_rate_ml_min: float = 0.5    # CSF production rate (ml/min)
    absorption_rate_ml_min: float = 0.5    # CSF absorption rate (ml/min)
    pressure_gradient_pa_m: float = 100.0  # Baseline pressure gradient (Pa/m)

@dataclass
class FlowBoundaryCondition:
    """Boundary condition for CSF flow."""
    boundary_type: FlowBoundaryType
    location: Tuple[int, int, int]    # Voxel coordinates
    value: float                      # Boundary value (pressure, flux, etc.)
    region_mask: Optional[np.ndarray] # Region where condition applies

@dataclass
class FlowField:
    """Complete CSF flow field solution."""
    pressure_field: np.ndarray        # Pressure field (Pa)
    velocity_field: np.ndarray        # Velocity field (m/s) - 3D vectors
    flow_streamlines: List[np.ndarray] # Flow streamline coordinates
    mass_conservation_error: float    # Mass conservation validation
