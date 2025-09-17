#!/usr/bin/env python3
"""Cell Constructor Core - Main cell construction engine and coordination.

Contains the core CellConstructor class and main construction orchestration.

Integration: Core cell construction engine for AlphaGenome biological workflows.
Rationale: Centralized cell construction logic separate from data types.
"""

import sys
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

# Import cell types
from .cell_types import CellType, DevelopmentalStage, CellularParameters

# Import API configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from api_config import get_alphagenome_config

logger = logging.getLogger(__name__)

class CellConstructor:
    """
    Comprehensive cell constructor for biological simulation.
    Constructs cells, manages cellular properties, and handles differentiation.
    """

    def __init__(self, dna_controller=None):
        self.dna_controller = dna_controller

        # Initialize construction components
        self.constructed_cells = {}
        self.cell_lineages = {}
        self.tissue_parameters = {}
        self.construction_history = []

        # Load API configuration
        self.config = get_alphagenome_config()

        # Initialize cell construction systems
        self._initialize_cell_templates()
        self._setup_differentiation_pathways()

        logger.info(f"CellConstructor initialized in {self.config['simulation_mode']} mode")

        # Handle AlphaGenome initialization
        if not self.config["api_key_available"]:
            print("No AlphaGenome API key provided - using simulation mode")

    def _initialize_cell_templates(self):
        """Initialize standard cell templates for construction."""
        self.cell_templates = {
            CellType.STEM_CELL: CellularParameters(
                cell_id="template_stem",
                cell_type=CellType.STEM_CELL,
                developmental_stage=DevelopmentalStage.NEURAL_INDUCTION,
                position=(0.0, 0.0, 0.0),
                diameter=15.0,
                division_probability=0.8,
                differentiation_potential={
                    "neural_progenitor": 0.7,
                    "astrocyte": 0.2,
                    "oligodendrocyte": 0.1
                }
            ),
            CellType.NEURAL_PROGENITOR: CellularParameters(
                cell_id="template_progenitor",
                cell_type=CellType.NEURAL_PROGENITOR,
                developmental_stage=DevelopmentalStage.NEURAL_PROLIFERATION,
                position=(0.0, 0.0, 0.0),
                diameter=12.0,
                division_probability=0.6,
                differentiation_potential={
                    "neuron": 0.8,
                    "astrocyte": 0.15,
                    "oligodendrocyte": 0.05
                }
            ),
            CellType.NEURON: CellularParameters(
                cell_id="template_neuron",
                cell_type=CellType.NEURON,
                developmental_stage=DevelopmentalStage.DIFFERENTIATION,
                position=(0.0, 0.0, 0.0),
                diameter=20.0,
                membrane_potential=-70.0,
                division_probability=0.0,  # Neurons don't divide
                gene_expression_profile={
                    "TUBB3": 0.9,
                    "MAP2": 0.8,
                    "NEFL": 0.7
                }
            )
        }

    def _setup_differentiation_pathways(self):
        """Set up cellular differentiation pathways."""
        self.differentiation_pathways = {
            "neurogenesis": {
                "start": CellType.NEURAL_PROGENITOR,
                "end": CellType.NEURON,
                "duration_hours": 24.0,
                "required_factors": ["NEUROG2", "NEUROD1"],
                "inhibiting_factors": ["NOTCH1"]
            },
            "gliogenesis": {
                "start": CellType.NEURAL_PROGENITOR,
                "end": CellType.ASTROCYTE,
                "duration_hours": 48.0,
                "required_factors": ["GFAP", "S100B"],
                "inhibiting_factors": ["NEUROG2"]
            },
            "oligodendrogenesis": {
                "start": CellType.NEURAL_PROGENITOR,
                "end": CellType.OLIGODENDROCYTE,
                "duration_hours": 72.0,
                "required_factors": ["OLIG2", "MBP"],
                "inhibiting_factors": ["GFAP"]
            }
        }

    def construct_cell(self, cell_type: CellType, position: Tuple[float, float, float],
                      custom_parameters: Optional[Dict[str, Any]] = None) -> str:
        """Construct a new cell at the specified position."""

        # Get template for cell type
        if cell_type in self.cell_templates:
            template = self.cell_templates[cell_type]

            # Create new cell parameters based on template
            cell_params = CellularParameters(
                cell_id=str(uuid.uuid4()),
                cell_type=cell_type,
                developmental_stage=template.developmental_stage,
                position=position,
                diameter=template.diameter,
                membrane_potential=template.membrane_potential,
                division_probability=template.division_probability,
                gene_expression_profile=template.gene_expression_profile.copy(),
                differentiation_potential=template.differentiation_potential.copy()
            )

            # Apply custom parameters if provided
            if custom_parameters:
                for param, value in custom_parameters.items():
                    if hasattr(cell_params, param):
                        setattr(cell_params, param, value)

            # Register constructed cell
            self.constructed_cells[cell_params.cell_id] = cell_params

            # Record construction event
            self.construction_history.append({
                "timestamp": 0.0,  # Would be actual simulation time
                "event": "cell_construction",
                "cell_id": cell_params.cell_id,
                "cell_type": cell_type.value,
                "position": position
            })

            logger.info(f"Constructed {cell_type.value} cell at {position}")
            return cell_params.cell_id

        else:
            logger.error(f"No template available for cell type: {cell_type}")
            return ""

    def differentiate_cell(self, cell_id: str, target_type: CellType) -> bool:
        """Differentiate a cell to a new type."""
        if cell_id not in self.constructed_cells:
            return False

        cell = self.constructed_cells[cell_id]
        current_type = cell.cell_type

        # Check if differentiation is possible
        pathway_name = f"{current_type.value}_to_{target_type.value}"

        # Find applicable pathway
        applicable_pathway = None
        for pathway_name, pathway in self.differentiation_pathways.items():
            if pathway["start"] == current_type and pathway["end"] == target_type:
                applicable_pathway = pathway
                break

        if applicable_pathway:
            # Update cell properties
            cell.cell_type = target_type
            cell.developmental_stage = DevelopmentalStage.DIFFERENTIATION

            # Update gene expression for new type
            if target_type == CellType.NEURON:
                cell.gene_expression_profile.update({
                    "TUBB3": 0.9,
                    "MAP2": 0.8,
                    "NEFL": 0.7
                })
            elif target_type == CellType.ASTROCYTE:
                cell.gene_expression_profile.update({
                    "GFAP": 0.9,
                    "S100B": 0.8,
                    "AQP4": 0.7
                })

            # Record differentiation event
            self.construction_history.append({
                "timestamp": 0.0,
                "event": "cell_differentiation",
                "cell_id": cell_id,
                "from_type": current_type.value,
                "to_type": target_type.value
            })

            logger.info(f"Differentiated cell {cell_id} from {current_type.value} to {target_type.value}")
            return True

        return False

    def get_cell_count_by_type(self) -> Dict[str, int]:
        """Get count of cells by type."""
        counts = {}
        for cell in self.constructed_cells.values():
            cell_type = cell.cell_type.value
            counts[cell_type] = counts.get(cell_type, 0) + 1
        return counts

    def get_construction_summary(self) -> Dict[str, Any]:
        """Get summary of cell construction state."""
        return {
            "total_cells": len(self.constructed_cells),
            "cell_types": self.get_cell_count_by_type(),
            "construction_events": len(self.construction_history),
            "api_mode": self.config["simulation_mode"],
            "available_templates": list(self.cell_templates.keys())
        }
