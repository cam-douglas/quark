#!/usr/bin/env python3
"""Cell Types Module - Core cell type definitions and developmental stages.

Defines cell types, developmental stages, and cellular parameters for cell construction.

Integration: Core types used by CellConstructor and biological simulation workflows.
Rationale: Centralized cell type definitions separate from construction logic.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple

class CellType(Enum):
    """Types of cells that can be constructed"""
    STEM_CELL = "stem_cell"
    NEURAL_PROGENITOR = "neural_progenitor"
    NEURON = "neuron"
    ASTROCYTE = "astrocyte"
    OLIGODENDROCYTE = "oligodendrocyte"
    MICROGLIA = "microglia"
    ENDOTHELIAL = "endothelial"
    EPENDYMAL = "ependymal"
    RADIAL_GLIA = "radial_glia"
    INTERNEURON = "interneuron"
    PROJECTION_NEURON = "projection_neuron"

class DevelopmentalStage(Enum):
    """Developmental stages for neural development"""
    NEURAL_INDUCTION = "neural_induction"
    NEURAL_PLATE = "neural_plate"
    NEURAL_TUBE_CLOSURE = "neural_tube_closure"
    NEURAL_PROLIFERATION = "neural_proliferation"
    NEURONAL_MIGRATION = "neuronal_migration"
    DIFFERENTIATION = "differentiation"
    SYNAPTOGENESIS = "synaptogenesis"
    CIRCUIT_REFINEMENT = "circuit_refinement"
    MYELINATION = "myelination"
    MATURATION = "maturation"

@dataclass
class CellularParameters:
    """Parameters defining cellular characteristics"""
    # Basic cellular properties
    cell_id: str
    cell_type: CellType
    developmental_stage: DevelopmentalStage
    position: Tuple[float, float, float]

    # Biological properties
    diameter: float = 10.0  # micrometers
    volume: float = 523.6  # cubic micrometers (sphere)
    membrane_potential: float = -70.0  # mV

    # Genetic properties
    gene_expression_profile: Dict[str, float] = None
    epigenetic_marks: Dict[str, float] = None

    # Metabolic properties
    energy_level: float = 1.0
    growth_rate: float = 0.1
    division_probability: float = 0.01

    # Connectivity
    synaptic_connections: List[str] = None
    gap_junctions: List[str] = None

    # Developmental properties
    migration_direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    differentiation_potential: Dict[str, float] = None

    def __post_init__(self):
        if self.gene_expression_profile is None:
            self.gene_expression_profile = {}
        if self.epigenetic_marks is None:
            self.epigenetic_marks = {}
        if self.synaptic_connections is None:
            self.synaptic_connections = []
        if self.gap_junctions is None:
            self.gap_junctions = []
        if self.differentiation_potential is None:
            self.differentiation_potential = {}

@dataclass
class TissueParameters:
    """Parameters defining tissue-level characteristics"""
    tissue_id: str
    tissue_type: str
    region: str

    # Spatial organization
    dimensions: Tuple[float, float, float]
    cell_density: float = 1000.0  # cells per cubic mm

    # Tissue properties
    stiffness: float = 1.0  # kPa
    permeability: float = 0.5
    vascularization: float = 0.1

    # Developmental properties
    growth_factors: Dict[str, float] = None
    morphogen_concentrations: Dict[str, float] = None
    mechanical_forces: Dict[str, float] = None

    # Cellular composition
    cell_type_ratios: Dict[str, float] = None

    def __post_init__(self):
        if self.growth_factors is None:
            self.growth_factors = {}
        if self.morphogen_concentrations is None:
            self.morphogen_concentrations = {}
        if self.mechanical_forces is None:
            self.mechanical_forces = {}
        if self.cell_type_ratios is None:
            self.cell_type_ratios = {}
