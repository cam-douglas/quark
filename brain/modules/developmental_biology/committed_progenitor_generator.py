"""
Committed Progenitor Generator

This module generates committed neural progenitors from neuroepithelial cells,
implementing restricted lineage potency and creating progenitor pools.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType
from .lineage_barcode_types import LineageTag
from .ventricular_zone_organizer import ZoneType
from .progenitor_type_classifier import (
    ProgenitorTypeClassifier, NeuronType, CommitmentLevel, LineagePotency
)
from .progenitor_pool_manager import ProgenitorPoolManager


from .committed_progenitor_types import CommittedProgenitor


class CommittedProgenitorGenerator:
    """
    Generates committed neural progenitors from neuroepithelial cells,
    implementing restricted lineage potency and creating progenitor pools.
    """
    
    def __init__(self):
        """Initialize committed progenitor generator"""
        self.type_classifier = ProgenitorTypeClassifier()
        self.pool_manager = ProgenitorPoolManager()
        self.commitment_parameters: Dict[str, Dict[str, float]] = {}
        self._setup_commitment_parameters()
    
    
    def _setup_commitment_parameters(self) -> None:
        """Setup commitment parameters for different zones"""
        self.commitment_parameters = {
            "ventricular_zone": {
                "commitment_probability": 0.3,
                "commitment_threshold": 0.7,
                "potency_restriction_rate": 0.2,
                "molecular_marker_strength": 0.8
            },
            "subventricular_zone": {
                "commitment_probability": 0.5,
                "commitment_threshold": 0.6,
                "potency_restriction_rate": 0.3,
                "molecular_marker_strength": 0.9
            },
            "intermediate_zone": {
                "commitment_probability": 0.7,
                "commitment_threshold": 0.5,
                "potency_restriction_rate": 0.4,
                "molecular_marker_strength": 1.0
            }
        }
    
    def generate_committed_progenitors(self, neuroepithelial_cells: Dict[str, NeuroepithelialCell],
                                     developmental_time: float,
                                     morphogen_concentrations: Dict[str, float]) -> Dict[str, CommittedProgenitor]:
        """
        Generate committed progenitors from neuroepithelial cells
        
        Args:
            neuroepithelial_cells: Dictionary of neuroepithelial cells
            developmental_time: Current developmental time in weeks
            morphogen_concentrations: Current morphogen concentrations
            
        Returns:
            Dictionary of generated committed progenitors
        """
        committed_progenitors = {}
        
        for cell_id, cell in neuroepithelial_cells.items():
            # Check if cell is ready for commitment
            if self._should_commit_cell(cell, developmental_time, morphogen_concentrations):
                # Determine progenitor type based on morphogen exposure
                progenitor_type = self.type_classifier.determine_progenitor_type(cell, morphogen_concentrations)
                
                # Generate committed progenitor
                progenitor = self._create_committed_progenitor(
                    cell, progenitor_type, developmental_time, morphogen_concentrations
                )
                
                if progenitor:
                    committed_progenitors[progenitor.cell_id] = progenitor
                    
                    # Add to appropriate progenitor pool
                    self.pool_manager.add_to_progenitor_pool(progenitor)
        
        return committed_progenitors
    
    def _should_commit_cell(self, cell: NeuroepithelialCell, developmental_time: float,
                          morphogen_concentrations: Dict[str, float]) -> bool:
        """Determine if a neuroepithelial cell should commit to a specific fate"""
        # Only multipotent cells can commit
        if cell.cell_type not in [NeuroepithelialCellType.EARLY_MULTIPOTENT, 
                                NeuroepithelialCellType.LATE_MULTIPOTENT]:
            return False
        
        # Get zone-specific parameters
        zone_name = cell.zone_type.value if hasattr(cell, 'zone_type') else "ventricular_zone"
        params = self.commitment_parameters.get(zone_name, 
                                              self.commitment_parameters["ventricular_zone"])
        
        # Calculate commitment probability based on morphogen exposure
        morphogen_signal = sum(morphogen_concentrations.values()) / len(morphogen_concentrations)
        commitment_probability = params["commitment_probability"] * morphogen_signal
        
        # Check if commitment threshold is met
        if morphogen_signal >= params["commitment_threshold"]:
            return np.random.random() < commitment_probability
        
        return False
    
    
    def _create_committed_progenitor(self, parent_cell: NeuroepithelialCell, 
                                   progenitor_type: str, developmental_time: float,
                                   morphogen_concentrations: Dict[str, float]) -> Optional[CommittedProgenitor]:
        """Create a committed progenitor from a neuroepithelial cell"""
        # Get lineage potency for this progenitor type
        lineage_potency = self.type_classifier.get_lineage_potency(progenitor_type)
        if not lineage_potency:
            return None
        
        # Check if within competency window
        if not self.type_classifier.validate_competency_window(progenitor_type, developmental_time):
            return None
        
        # Generate new cell ID for committed progenitor
        progenitor_id = f"cp_{parent_cell.cell_id}_{progenitor_type}"
        
        # Create lineage barcode (inherit from parent if available)
        lineage_barcode = self._create_lineage_barcode(parent_cell, progenitor_type)
        
        # Generate molecular markers
        molecular_markers = self.type_classifier.generate_molecular_markers(
            progenitor_type, morphogen_concentrations
        )
        
        # Create committed progenitor
        committed_progenitor = CommittedProgenitor(
            cell_id=progenitor_id,
            parent_cell=parent_cell,
            lineage_potency=lineage_potency,
            lineage_barcode=lineage_barcode,
            commitment_time=developmental_time,
            zone_type=parent_cell.zone_type if hasattr(parent_cell, 'zone_type') else ZoneType.VENTRICULAR_ZONE,
            position=parent_cell.position,
            molecular_markers=molecular_markers
        )
        
        return committed_progenitor
    
    def _create_lineage_barcode(self, parent_cell: NeuroepithelialCell, 
                              progenitor_type: str) -> LineageTag:
        """Create lineage tag for committed progenitor"""
        # Create simple lineage tag
        cell_id = f"cp_{parent_cell.cell_id}_{progenitor_type}"
        generation = getattr(parent_cell, 'generation', 0) + 1
        
        return LineageTag(
            cell_id=cell_id,
            generation=generation,
            lineage_barcodes={},  # Would be populated by barcode generator
            lineage_history=[parent_cell.cell_id],
            creation_time=0.0,  # Would be set by calling system
            last_update_time=0.0
        )
    
    # Public interface methods - delegate to managers
    def get_progenitor_pool(self, neuron_type: NeuronType) -> List[CommittedProgenitor]:
        """Get progenitor pool for specific neuron type"""
        return self.pool_manager.get_progenitor_pool(neuron_type)
    
    def get_progenitor_pool_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all progenitor pools"""
        return self.pool_manager.get_progenitor_pool_statistics()
    
    def validate_lineage_potency(self, progenitor: CommittedProgenitor, 
                               target_fate: NeuronType) -> bool:
        """Validate if progenitor can differentiate into target fate"""
        return self.pool_manager.validate_lineage_potency(progenitor, target_fate)
    
    def restrict_lineage_potency(self, progenitor: CommittedProgenitor, 
                               developmental_time: float) -> LineagePotency:
        """Restrict lineage potency over time"""
        time_since_commitment = developmental_time - progenitor.commitment_time
        return self.type_classifier.restrict_lineage_potency(
            progenitor.lineage_potency, time_since_commitment
        )
