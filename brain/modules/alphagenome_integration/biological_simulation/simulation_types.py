#!/usr/bin/env python3
"""Simulation Types Module - Core data structures for biological simulation.

Defines enums, data classes, and type definitions for biological simulation.

Integration: Core types used by BiologicalSimulator and related analyses.
Rationale: Centralized type definitions with clean separation from business logic.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import uuid

class SimulationMode(Enum):
    """Simulation execution modes"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    DEBUG = "debug"
    BENCHMARK = "benchmark"

class BiologicalProcess(Enum):
    """Types of biological processes that can be simulated"""
    NEURAL_DEVELOPMENT = "neural_development"
    CELL_DIVISION = "cell_division"
    DIFFERENTIATION = "differentiation"
    MORPHOGENESIS = "morphogenesis"
    GENE_EXPRESSION = "gene_expression"
    PROTEIN_SYNTHESIS = "protein_synthesis"
    SYNAPTIC_FORMATION = "synaptic_formation"
    MYELINATION = "myelination"
    NEURONAL_MIGRATION = "neuronal_migration"
    APOPTOSIS = "apoptosis"

@dataclass
class MorphogenGradient:
    """Represents a morphogen concentration gradient"""
    name: str
    concentration: float
    position: Tuple[float, float, float]
    diffusion_rate: float
    decay_rate: float
    source_strength: float

@dataclass
class DevelopmentalEvent:
    """Represents a discrete developmental event"""
    event_id: str
    event_type: BiologicalProcess
    timestamp: float
    location: Tuple[float, float, float]
    parameters: Dict[str, Any]
    success: bool = False

@dataclass
class SimulationParameters:
    """Configuration parameters for biological simulation"""
    simulation_id: str = None
    mode: SimulationMode = SimulationMode.DEVELOPMENT
    time_steps: int = 1000
    spatial_resolution: float = 10.0  # Micrometers
    temporal_resolution: float = 0.01  # hours
    spatial_dimensions: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)  # um
    
    # Biological constraints
    max_cell_density: float = 1000.0  # cells per cubic mm
    morphogen_diffusion_enabled: bool = True
    cell_migration_enabled: bool = True
    cell_division_enabled: bool = True
    apoptosis_enabled: bool = True
    
    # Output settings
    save_frequency: float = 1.0  # Hours
    enable_visualization: bool = False
    enable_logging: bool = True
    detailed_logging: bool = True
    
    def __post_init__(self):
        if self.simulation_id is None:
            self.simulation_id = str(uuid.uuid4())
