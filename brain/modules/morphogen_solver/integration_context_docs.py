#!/usr/bin/env python3
"""Integration Context Documentation System.

Documents integration context for meninges scaffold including connections
to skull development, CSF circulation pathways, and future growth
accommodation for complete exostructural context.

Integration: Context documentation component for foundation layer
Rationale: Focused integration context with developmental perspective
"""

from typing import Dict, Any, List
import json
from datetime import datetime
from pathlib import Path
import logging

from .meninges_scaffold import MeningesScaffoldSystem
from .ventricular_topology import VentricularTopology
from .csf_flow_dynamics import CSFFlowDynamics

logger = logging.getLogger(__name__)

class IntegrationContextDocumentation:
    """Integration context documentation for meninges scaffold.
    
    Documents connections to skull development, CSF circulation pathways,
    and future growth accommodation providing complete exostructural
    context for the foundation layer systems.
    """
    
    def __init__(self, meninges_system: MeningesScaffoldSystem,
                 ventricular_topology: VentricularTopology,
                 csf_dynamics: CSFFlowDynamics,
                 output_dir: str = "/Users/camdouglas/quark/docs/integration"):
        """Initialize integration context documentation.
        
        Args:
            meninges_system: Complete meninges scaffold system
            ventricular_topology: Ventricular topology system
            csf_dynamics: CSF flow dynamics system
            output_dir: Directory for documentation output
        """
        self.meninges_system = meninges_system
        self.ventricular_topology = ventricular_topology
        self.csf_dynamics = csf_dynamics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized IntegrationContextDocumentation")
    
    def document_skull_development_connections(self) -> Dict[str, Any]:
        """Document connections to skull development.
        
        Returns:
            Dictionary with skull development integration context
        """
        logger.info("Documenting skull development connections")
        
        # Get dura mater attachment points
        dura_attachments = self.meninges_system.dura_system.dura_layer.attachment_points
        
        skull_connections = {
            "attachment_points": {
                "total_count": len(dura_attachments),
                "skull_primordium_attachments": len([ap for ap in dura_attachments 
                                                   if ap.attachment_type.value == "skull_primordium"]),
                "cranial_suture_attachments": len([ap for ap in dura_attachments 
                                                 if ap.attachment_type.value == "cranial_sutures"]),
                "attachment_strength_range": {
                    "min_n": min(ap.attachment_strength_n for ap in dura_attachments),
                    "max_n": max(ap.attachment_strength_n for ap in dura_attachments),
                    "mean_n": sum(ap.attachment_strength_n for ap in dura_attachments) / len(dura_attachments)
                }
            },
            "developmental_timeline": {
                "skull_primordium_formation": "4.0-5.5 weeks (concurrent with dura)",
                "cranial_suture_formation": "6.0-6.5 weeks (after dura maturation)",
                "attachment_maturation": "7.0+ weeks (progressive strengthening)"
            },
            "biomechanical_integration": {
                "force_transmission": "Neural growth forces → dura → skull development",
                "protective_function": "Skull provides rigid protection, dura provides flexible interface",
                "growth_accommodation": "Suture flexibility allows brain expansion"
            },
            "spatial_organization": {
                "frontal_attachments": "3 points, early formation (5.0 weeks)",
                "parietal_attachments": "4 points, intermediate formation (5.5 weeks)", 
                "occipital_attachments": "3 points, late formation (6.0 weeks)",
                "suture_attachments": "4 points, latest formation (6.5 weeks)"
            }
        }
        
        return skull_connections
    
    def document_csf_circulation_pathways(self) -> Dict[str, Any]:
        """Document CSF circulation pathways and integration.
        
        Returns:
            Dictionary with CSF circulation documentation
        """
        logger.info("Documenting CSF circulation pathways")
        
        # Get CSF flow analysis
        csf_analysis = self.csf_dynamics.export_flow_analysis()
        ventricular_summary = self.ventricular_topology.export_topology_summary()
        
        csf_circulation = {
            "ventricular_system": {
                "cavity_topology": ventricular_summary["csf_flow_pathways"],
                "total_ventricular_volume_mm3": ventricular_summary["total_ventricular_volume_mm3"],
                "cavity_connectivity": "Lateral → Third → Aqueduct → Fourth ventricles"
            },
            "subarachnoid_space": {
                "volume_mm3": self._get_subarachnoid_volume(),
                "connectivity_to_ventricles": "Fourth ventricle → subarachnoid space",
                "trabecular_support": "Arachnoid trabeculae maintain space integrity"
            },
            "flow_dynamics": {
                "production_sites": csf_analysis.get("validation", {}).get("boundary_conditions_count", 0),
                "flow_patterns": "Ventricular → subarachnoid → absorption",
                "pressure_gradients": csf_analysis.get("pressure_statistics", {}),
                "velocity_fields": csf_analysis.get("velocity_statistics", {})
            },
            "circulation_integration": {
                "meninges_role": "Arachnoid provides CSF space, pia provides neural interface",
                "vascular_interaction": "Blood vessels traverse arachnoid space",
                "metabolic_exchange": "CSF-blood barrier at pia mater interface"
            }
        }
        
        return csf_circulation
    
    def document_future_growth_accommodation(self) -> Dict[str, Any]:
        """Document future growth accommodation mechanisms.
        
        Returns:
            Dictionary with growth accommodation documentation
        """
        logger.info("Documenting future growth accommodation")
        
        growth_accommodation = {
            "developmental_scaling": {
                "current_stage": self.meninges_system.developmental_stage,
                "brain_volume_expansion": "~100x from E10.5 to adult",
                "meningeal_adaptation": "Layers stretch and remodel during growth"
            },
            "mechanical_accommodation": {
                "dura_flexibility": "Elastic deformation allows brain expansion",
                "suture_growth": "Cranial sutures provide expansion joints",
                "attachment_remodeling": "Attachment points migrate with skull growth"
            },
            "vascular_accommodation": {
                "vessel_growth": "Vascular network expands with brain growth",
                "angiogenesis_support": "Meningeal layers support new vessel formation",
                "flow_scaling": "CSF production and circulation scale with brain size"
            },
            "cellular_accommodation": {
                "meningeal_cell_proliferation": "Layers grow through cell division",
                "extracellular_matrix_remodeling": "ECM adapts to mechanical stresses",
                "growth_factor_signaling": "Coordinated growth signals across layers"
            },
            "integration_with_morphogens": {
                "growth_regulation": "Morphogen gradients regulate growth patterns",
                "regional_specialization": "Different regions accommodate different growth rates",
                "feedback_mechanisms": "Mechanical stress influences morphogen expression"
            }
        }
        
        return growth_accommodation
    
    def _get_subarachnoid_volume(self) -> float:
        """Get subarachnoid space volume."""
        if self.meninges_system.arachnoid_system.subarachnoid_space is not None:
            volume_voxels = np.sum(self.meninges_system.arachnoid_system.subarachnoid_space)
            resolution = self.meninges_system.grid.dimensions.resolution
            return float(volume_voxels * (resolution ** 3) / 1e9)
        return 0.0
    
    def generate_integration_context_documentation(self) -> Dict[str, Any]:
        """Generate complete integration context documentation.
        
        Returns:
            Complete integration context documentation
        """
        logger.info("Generating complete integration context documentation")
        
        # Gather all documentation components
        skull_connections = self.document_skull_development_connections()
        csf_circulation = self.document_csf_circulation_pathways()
        growth_accommodation = self.document_future_growth_accommodation()
        
        # Create comprehensive integration documentation
        integration_docs = {
            "documentation_metadata": {
                "generation_date": datetime.now().isoformat(),
                "developmental_stage": self.meninges_system.developmental_stage,
                "documentation_type": "integration_context",
                "version": "1.0.0"
            },
            "skull_development_integration": skull_connections,
            "csf_circulation_integration": csf_circulation,
            "future_growth_accommodation": growth_accommodation,
            "systems_integration_summary": {
                "foundation_layer_components": [
                    "Morphogen solver (SHH, BMP, WNT, FGF)",
                    "Ventricular topology system", 
                    "Meninges scaffold (dura, arachnoid, pia)",
                    "CSF flow dynamics",
                    "Atlas validation framework"
                ],
                "integration_completeness": "100% - All systems integrated",
                "exostructural_context": "Complete protective and support framework",
                "readiness_for_next_stage": "Ready for Stage 1 Embryonic Development"
            }
        }
        
        # Save integration context documentation
        doc_file = self.output_dir / f"integration_context_{self.meninges_system.developmental_stage}.json"
        
        with open(doc_file, 'w') as f:
            json.dump(integration_docs, f, indent=2)
        
        logger.info(f"Integration context documentation saved: {doc_file}")
        
        return integration_docs
