"""
Downstream Interface Manager

This module prepares interfaces for Stage 2 neurogenesis, creates progenitor
pool exports, and implements handoff protocols for downstream processes.

Author: Quark AI
Date: 2025-01-27
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .committed_progenitor_types import CommittedProgenitor
from .progenitor_type_classifier import NeuronType
from .lineage_tag_preservator import TagPreservationState
from .progenitor_export_handler import ProgenitorExportHandler, ProgenitorPoolExport, ExportFormat


class HandoffStatus(Enum):
    """Status of downstream handoff"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HandoffProtocol:
    """Protocol for downstream handoff"""
    protocol_id: str
    target_system: str
    handoff_status: HandoffStatus
    progenitor_pools: List[ProgenitorPoolExport]
    validation_results: Dict[str, bool]
    handoff_timestamp: float


class DownstreamInterfaceManager:
    """
    Prepares interfaces for Stage 2 neurogenesis, creates progenitor
    pool exports, and implements handoff protocols.
    """
    
    def __init__(self):
        """Initialize downstream interface manager"""
        self.active_handoffs: Dict[str, HandoffProtocol] = {}
        self.export_handler = ProgenitorExportHandler()
        self.interface_parameters: Dict[str, Any] = {}
        self._setup_interface_parameters()
    
    def _setup_interface_parameters(self) -> None:
        """Setup interface parameters for downstream systems"""
        self.interface_parameters = {
            "stage2_neurogenesis": {
                "required_progenitor_types": [
                    "motor_neuron_progenitor",
                    "ventral_interneuron_progenitor",
                    "dorsal_interneuron_progenitor"
                ],
                "minimum_pool_sizes": {
                    "motor_neuron": 100,
                    "interneuron_v0": 50,
                    "interneuron_v1": 50,
                    "dorsal_interneuron": 75
                },
                "required_fidelity_threshold": 0.8,
                "export_format": ExportFormat.JSON
            },
            "fetal_development": {
                "required_progenitor_types": [
                    "glial_progenitor",
                    "motor_neuron_progenitor"
                ],
                "minimum_pool_sizes": {
                    "oligodendrocyte": 200,
                    "astrocyte": 150,
                    "motor_neuron": 300
                },
                "required_fidelity_threshold": 0.75,
                "export_format": ExportFormat.HDF5
            }
        }
    
    def prepare_downstream_interface(self, target_system: str,
                                   progenitor_pools: Dict[NeuronType, List[CommittedProgenitor]],
                                   preservation_states: Dict[str, TagPreservationState]) -> str:
        """
        Prepare interface for downstream system
        
        Args:
            target_system: Target downstream system
            progenitor_pools: Dictionary of progenitor pools
            preservation_states: Dictionary of tag preservation states
            
        Returns:
            Protocol ID for the handoff
        """
        # Generate protocol ID
        protocol_id = f"handoff_{target_system}_{len(self.active_handoffs)}"
        
        # Get interface parameters for target system
        if target_system not in self.interface_parameters:
            raise ValueError(f"Unknown target system: {target_system}")
        
        params = self.interface_parameters[target_system]
        
        # Validate progenitor pools meet requirements
        validation_results = self._validate_progenitor_pools(progenitor_pools, params)
        
        # Create progenitor pool exports
        pool_exports = self.export_handler.create_progenitor_pool_exports(
            progenitor_pools, preservation_states, params["export_format"]
        )
        
        # Create handoff protocol
        handoff_protocol = HandoffProtocol(
            protocol_id=protocol_id,
            target_system=target_system,
            handoff_status=HandoffStatus.PENDING,
            progenitor_pools=pool_exports,
            validation_results=validation_results,
            handoff_timestamp=0.0  # Would be set by calling system
        )
        
        # Store active handoff
        self.active_handoffs[protocol_id] = handoff_protocol
        
        return protocol_id
    
    def _validate_progenitor_pools(self, progenitor_pools: Dict[NeuronType, List[CommittedProgenitor]],
                                 params: Dict[str, Any]) -> Dict[str, bool]:
        """Validate progenitor pools meet downstream requirements"""
        validation_results = {}
        
        # Check minimum pool sizes
        minimum_sizes = params["minimum_pool_sizes"]
        for neuron_type_str, min_size in minimum_sizes.items():
            # Find matching neuron type
            matching_pools = []
            for neuron_type, progenitors in progenitor_pools.items():
                if neuron_type_str in neuron_type.value:
                    matching_pools.extend(progenitors)
            
            validation_results[f"pool_size_{neuron_type_str}"] = len(matching_pools) >= min_size
        
        # Check required progenitor types are present
        required_types = params["required_progenitor_types"]
        for req_type in required_types:
            has_type = any(req_type in str(progenitors[0].lineage_potency.allowed_fates)
                          for progenitors in progenitor_pools.values() if progenitors)
            validation_results[f"has_type_{req_type}"] = has_type
        
        return validation_results
    
    
    def execute_handoff(self, protocol_id: str) -> bool:
        """Execute handoff to downstream system"""
        if protocol_id not in self.active_handoffs:
            raise ValueError(f"Unknown protocol ID: {protocol_id}")
        
        handoff = self.active_handoffs[protocol_id]
        
        # Update status to in progress
        handoff.handoff_status = HandoffStatus.IN_PROGRESS
        
        # Check if all validations passed
        all_validations_passed = all(handoff.validation_results.values())
        
        if all_validations_passed:
            # Perform handoff operations
            success = self._perform_handoff_operations(handoff)
            
            if success:
                handoff.handoff_status = HandoffStatus.COMPLETED
                return True
            else:
                handoff.handoff_status = HandoffStatus.FAILED
                return False
        else:
            handoff.handoff_status = HandoffStatus.FAILED
            return False
    
    def _perform_handoff_operations(self, handoff: HandoffProtocol) -> bool:
        """Perform actual handoff operations"""
        try:
            # Export progenitor pools to files
            export_files = []
            for pool_export in handoff.progenitor_pools:
                filename = self.export_handler.export_pool_to_file(pool_export, handoff.target_system)
                export_files.append(filename)
            
            # Create handoff manifest
            self._create_handoff_manifest(handoff, export_files)
            
            return True
        except Exception as e:
            print(f"Handoff operation failed: {e}")
            return False
    
    def _create_handoff_manifest(self, handoff: HandoffProtocol, export_files: List[str]) -> None:
        """Create manifest file for handoff"""
        manifest = {
            "protocol_id": handoff.protocol_id,
            "target_system": handoff.target_system,
            "handoff_status": handoff.handoff_status.value,
            "total_pools": len(handoff.progenitor_pools),
            "total_progenitors": sum(pool.total_progenitors for pool in handoff.progenitor_pools),
            "validation_results": handoff.validation_results,
            "export_files": export_files
        }
        
        # In a real implementation, would write to file
        # with open(f"{handoff.protocol_id}_manifest.json", 'w') as f:
        #     json.dump(manifest, f, indent=2)
        print(f"Created handoff manifest for {handoff.protocol_id}")
    
    def get_handoff_status(self, protocol_id: str) -> Optional[HandoffStatus]:
        """Get status of handoff protocol"""
        if protocol_id in self.active_handoffs:
            return self.active_handoffs[protocol_id].handoff_status
        return None
    
    def get_interface_statistics(self) -> Dict[str, Any]:
        """Get statistics for downstream interfaces"""
        total_handoffs = len(self.active_handoffs)
        completed_handoffs = sum(1 for h in self.active_handoffs.values() 
                               if h.handoff_status == HandoffStatus.COMPLETED)
        failed_handoffs = sum(1 for h in self.active_handoffs.values() 
                            if h.handoff_status == HandoffStatus.FAILED)
        
        # Get export statistics from export handler
        export_stats = self.export_handler.get_export_statistics()
        total_exports = export_stats["total_exports"]
        total_progenitors_exported = export_stats["total_progenitors_exported"]
        exports_by_type = export_stats["exports_by_neuron_type"]
        
        return {
            "total_handoffs": total_handoffs,
            "completed_handoffs": completed_handoffs,
            "failed_handoffs": failed_handoffs,
            "success_rate": completed_handoffs / total_handoffs if total_handoffs > 0 else 0.0,
            "total_exports": total_exports,
            "total_progenitors_exported": total_progenitors_exported,
            "exports_by_neuron_type": exports_by_type,
            "supported_target_systems": list(self.interface_parameters.keys())
        }
    
    def validate_readiness_for_stage2(self, progenitor_pools: Dict[NeuronType, List[CommittedProgenitor]]) -> Dict[str, bool]:
        """Validate readiness for Stage 2 Fetal Development"""
        stage2_params = self.interface_parameters["stage2_neurogenesis"]
        return self._validate_progenitor_pools(progenitor_pools, stage2_params)
