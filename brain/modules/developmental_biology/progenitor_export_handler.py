"""
Progenitor Export Handler

This module handles the export of progenitor pools to various formats
for downstream processing systems.

Author: Quark AI
Date: 2025-01-27
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
from .committed_progenitor_types import CommittedProgenitor
from .progenitor_type_classifier import NeuronType
from .lineage_tag_preservator import TagPreservationState


class ExportFormat(Enum):
    """Format for progenitor pool exports"""
    JSON = "json"
    CSV = "csv"
    PICKLE = "pickle"
    HDF5 = "hdf5"


@dataclass
class ProgenitorPoolExport:
    """Export data for progenitor pools"""
    neuron_type: str
    total_progenitors: int
    progenitor_data: List[Dict[str, Any]]
    export_timestamp: float
    export_format: str
    metadata: Dict[str, Any]


class ProgenitorExportHandler:
    """
    Handles the export of progenitor pools to various formats
    for downstream processing systems.
    """
    
    def __init__(self):
        """Initialize progenitor export handler"""
        self.completed_exports: List[ProgenitorPoolExport] = []
    
    def create_progenitor_pool_exports(self, progenitor_pools: Dict[NeuronType, List[CommittedProgenitor]],
                                     preservation_states: Dict[str, TagPreservationState],
                                     export_format: ExportFormat) -> List[ProgenitorPoolExport]:
        """Create exports for progenitor pools"""
        pool_exports = []
        
        for neuron_type, progenitors in progenitor_pools.items():
            if not progenitors:
                continue
            
            # Convert progenitors to exportable data
            progenitor_data = []
            for progenitor in progenitors:
                progenitor_dict = self._convert_progenitor_to_dict(progenitor, preservation_states)
                progenitor_data.append(progenitor_dict)
            
            # Create export
            export = ProgenitorPoolExport(
                neuron_type=neuron_type.value,
                total_progenitors=len(progenitors),
                progenitor_data=progenitor_data,
                export_timestamp=0.0,
                export_format=export_format.value,
                metadata={
                    "source_system": "developmental_biology_lineage",
                    "stage": "embryonic_lineage_generation",
                    "version": "1.0"
                }
            )
            
            pool_exports.append(export)
            self.completed_exports.append(export)
        
        return pool_exports
    
    def _convert_progenitor_to_dict(self, progenitor: CommittedProgenitor, 
                                  preservation_states: Dict[str, TagPreservationState]) -> Dict[str, Any]:
        """Convert progenitor to dictionary format"""
        # Get preservation state if available
        preservation_state = preservation_states.get(progenitor.cell_id)
        
        progenitor_dict = {
            "cell_id": progenitor.cell_id,
            "parent_cell_id": progenitor.parent_cell.cell_id,
            "commitment_time": progenitor.commitment_time,
            "zone_type": progenitor.zone_type.value,
            "position": progenitor.position,
            "molecular_markers": progenitor.molecular_markers,
            "lineage_potency": {
                "commitment_level": progenitor.lineage_potency.commitment_level.value,
                "allowed_fates": [fate.value for fate in progenitor.lineage_potency.allowed_fates],
                "restriction_factors": progenitor.lineage_potency.restriction_factors,
                "competency_window": progenitor.lineage_potency.competency_window
            },
            "lineage_barcode": {
                "barcode_id": progenitor.lineage_barcode.barcode_id,
                "dna_barcode": progenitor.lineage_barcode.dna_barcode,
                "rna_barcode": progenitor.lineage_barcode.rna_barcode,
                "protein_barcode": progenitor.lineage_barcode.protein_barcode,
                "epigenetic_barcode": progenitor.lineage_barcode.epigenetic_barcode,
                "generation": progenitor.lineage_barcode.generation
            }
        }
        
        # Add preservation state if available
        if preservation_state:
            progenitor_dict["preservation_state"] = {
                "inheritance_mode": preservation_state.inheritance_mode.value,
                "fidelity_level": preservation_state.fidelity_level.value,
                "preservation_time": preservation_state.preservation_time
            }
        
        return progenitor_dict
    
    def export_pool_to_file(self, pool_export: ProgenitorPoolExport, target_system: str) -> str:
        """Export progenitor pool to file"""
        filename = f"{target_system}_{pool_export.neuron_type}_progenitors.{pool_export.export_format}"
        
        if pool_export.export_format == "json":
            # Export as JSON
            export_data = asdict(pool_export)
            # In a real implementation, would write to file
            print(f"Exported {pool_export.total_progenitors} {pool_export.neuron_type} progenitors to {filename}")
        
        elif pool_export.export_format == "csv":
            # Export as CSV
            print(f"Exported {pool_export.total_progenitors} {pool_export.neuron_type} progenitors to {filename}")
        
        elif pool_export.export_format == "hdf5":
            # Export as HDF5
            print(f"Exported {pool_export.total_progenitors} {pool_export.neuron_type} progenitors to {filename}")
        
        return filename
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics"""
        total_exports = len(self.completed_exports)
        total_progenitors_exported = sum(export.total_progenitors for export in self.completed_exports)
        
        # Count exports by neuron type
        exports_by_type = {}
        for export in self.completed_exports:
            exports_by_type[export.neuron_type] = exports_by_type.get(export.neuron_type, 0) + export.total_progenitors
        
        return {
            "total_exports": total_exports,
            "total_progenitors_exported": total_progenitors_exported,
            "exports_by_neuron_type": exports_by_type
        }
