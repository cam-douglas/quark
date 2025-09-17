"""
Expert annotation protocol and nuclei taxonomy for brainstem segmentation.

Defines comprehensive annotation guidelines, nuclei taxonomy, and training
materials for expert annotators working on embryonic brainstem data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class NucleusDefinition:
    """Definition of a brainstem nucleus for annotation."""
    
    id: int
    name: str
    subdivision: str  # midbrain, pons, medulla
    function: str  # sensorimotor, autonomic
    description: str
    anatomical_landmarks: List[str]
    annotation_guidelines: str
    common_mistakes: List[str]


class BrainstemNucleiTaxonomy:
    """Comprehensive taxonomy of brainstem nuclei for expert annotation."""
    
    def __init__(self):
        """Initialize with embryonic brainstem nuclei definitions."""
        self.nuclei = self._create_nuclei_definitions()
        self.subdivisions = ["midbrain", "pons", "medulla"]
        self.functional_types = ["sensorimotor", "autonomic", "arousal"]
    
    def _create_nuclei_definitions(self) -> List[NucleusDefinition]:
        """Create comprehensive nuclei definitions based on Stage-1 requirements."""
        return [
            # Midbrain nuclei
            NucleusDefinition(
                id=1,
                name="Periaqueductal Grey",
                subdivision="midbrain",
                function="autonomic",
                description="Central grey matter surrounding cerebral aqueduct",
                anatomical_landmarks=["cerebral aqueduct", "superior colliculus"],
                annotation_guidelines="Identify grey matter immediately surrounding aqueduct. Avoid including aqueduct lumen.",
                common_mistakes=["Including aqueduct space", "Extending too far laterally"]
            ),
            NucleusDefinition(
                id=2,
                name="Edinger-Westphal Nucleus",
                subdivision="midbrain",
                function="autonomic",
                description="Parasympathetic preganglionic neurons for pupillary control",
                anatomical_landmarks=["oculomotor nucleus", "cerebral aqueduct"],
                annotation_guidelines="Small cluster dorsomedial to main oculomotor nucleus",
                common_mistakes=["Confusing with main oculomotor nucleus", "Missing small size"]
            ),
            NucleusDefinition(
                id=3,
                name="Substantia Nigra",
                subdivision="midbrain",
                function="sensorimotor",
                description="Dopaminergic neurons for motor control",
                anatomical_landmarks=["cerebral peduncle", "red nucleus"],
                annotation_guidelines="Dark band ventral to red nucleus, lateral to cerebral peduncle",
                common_mistakes=["Including cerebral peduncle fibers", "Unclear pars compacta/reticulata boundary"]
            ),
            NucleusDefinition(
                id=4,
                name="Red Nucleus",
                subdivision="midbrain",
                function="sensorimotor",
                description="Motor coordination and cerebellar relay",
                anatomical_landmarks=["substantia nigra", "superior colliculus"],
                annotation_guidelines="Round, well-defined nucleus in tegmentum",
                common_mistakes=["Extending into surrounding white matter"]
            ),
            
            # Pons nuclei
            NucleusDefinition(
                id=5,
                name="Locus Coeruleus",
                subdivision="pons",
                function="arousal",
                description="Noradrenergic neurons for arousal and attention",
                anatomical_landmarks=["fourth ventricle", "superior cerebellar peduncle"],
                annotation_guidelines="Small, densely packed cluster near fourth ventricle floor",
                common_mistakes=["Confusing with adjacent pontine nuclei", "Overestimating size"]
            ),
            NucleusDefinition(
                id=6,
                name="Pontine Nuclei",
                subdivision="pons",
                function="sensorimotor",
                description="Corticopontocerebellar relay neurons",
                anatomical_landmarks=["basilar pons", "corticospinal tract"],
                annotation_guidelines="Scattered nuclei throughout basilar pons",
                common_mistakes=["Including white matter tracts", "Missing scattered distribution"]
            ),
            NucleusDefinition(
                id=7,
                name="Facial Nucleus",
                subdivision="pons",
                function="sensorimotor",
                description="Motor neurons for facial expression",
                anatomical_landmarks=["facial nerve", "olivary complex"],
                annotation_guidelines="Compact motor nucleus in ventrolateral pons",
                common_mistakes=["Confusing with nearby olivary nuclei"]
            ),
            
            # Medulla nuclei
            NucleusDefinition(
                id=8,
                name="Raphe Magnus",
                subdivision="medulla",
                function="autonomic",
                description="Serotonergic neurons for pain modulation",
                anatomical_landmarks=["midline", "pyramidal tract"],
                annotation_guidelines="Midline cluster in rostral medulla",
                common_mistakes=["Extending too far laterally", "Missing midline location"]
            ),
            NucleusDefinition(
                id=9,
                name="Nucleus Ambiguus",
                subdivision="medulla",
                function="autonomic",
                description="Motor neurons for larynx, pharynx, and heart",
                anatomical_landmarks=["olivary complex", "spinal trigeminal nucleus"],
                annotation_guidelines="Elongated column in ventrolateral medulla",
                common_mistakes=["Confusing with hypoglossal nucleus", "Missing elongated shape"]
            ),
            NucleusDefinition(
                id=10,
                name="Pre-Bötzinger Complex",
                subdivision="medulla",
                function="autonomic",
                description="Respiratory rhythm generator",
                anatomical_landmarks=["nucleus ambiguus", "hypoglossal nucleus"],
                annotation_guidelines="Small cluster ventromedial to nucleus ambiguus",
                common_mistakes=["Overestimating size", "Unclear boundaries with ambiguus"]
            ),
            NucleusDefinition(
                id=11,
                name="Hypoglossal Nucleus",
                subdivision="medulla",
                function="sensorimotor",
                description="Motor neurons for tongue movement",
                anatomical_landmarks=["central canal", "medial lemniscus"],
                annotation_guidelines="Paired nuclei near midline in dorsal medulla",
                common_mistakes=["Confusing with dorsal motor nucleus of vagus"]
            ),
            NucleusDefinition(
                id=12,
                name="Dorsal Motor Nucleus of Vagus",
                subdivision="medulla",
                function="autonomic",
                description="Parasympathetic preganglionic neurons",
                anatomical_landmarks=["fourth ventricle floor", "hypoglossal nucleus"],
                annotation_guidelines="Longitudinal column in dorsal medulla",
                common_mistakes=["Confusing with hypoglossal nucleus", "Missing longitudinal extent"]
            )
        ]
    
    def get_annotation_protocol(self) -> Dict[str, Any]:
        """Generate comprehensive annotation protocol document."""
        protocol = {
            "title": "Embryonic Brainstem Annotation Protocol v1.0",
            "date": "2025-09-17",
            "overview": {
                "purpose": "Standardize expert annotation of embryonic brainstem nuclei",
                "target_resolution": "≤50 µm isotropic",
                "coordinate_system": "DevCCF E13.5 template space",
                "annotation_tool": "ITK-SNAP or 3D Slicer"
            },
            "general_guidelines": {
                "consistency": "Use same brightness/contrast settings across all volumes",
                "boundaries": "Define boundaries at 50% intensity transitions",
                "uncertainty": "Mark uncertain regions with separate uncertainty label",
                "quality_control": "Cross-check with anatomical atlas every 10 slices"
            },
            "nuclei_definitions": [asdict(nucleus) for nucleus in self.nuclei],
            "validation_criteria": {
                "inter_annotator_agreement": "Dice ≥ 0.90",
                "atlas_correspondence": "Landmark error ≤ 200 µm",
                "completion_time": "≤ 4 hours per volume"
            },
            "training_requirements": {
                "prerequisites": "Neuroanatomy background, 3D visualization experience",
                "training_duration": "8 hours initial + 4 hours supervised practice",
                "certification": "Pass validation test with Dice ≥ 0.85"
            }
        }
        
        return protocol
    
    def save_protocol(self, output_dir: Path) -> Path:
        """Save annotation protocol to JSON file."""
        protocol = self.get_annotation_protocol()
        
        output_file = output_dir / "expert_annotation_protocol.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(protocol, f, indent=2)
        
        logger.info(f"Annotation protocol saved: {output_file}")
        return output_file
    
    def generate_training_materials(self, output_dir: Path) -> List[Path]:
        """Generate training materials for annotators."""
        from brain.modules.brainstem_segmentation.annotation_training_materials import generate_all_training_materials
        return generate_all_training_materials(output_dir)


def create_expert_annotation_package(data_dir: Path) -> Dict[str, Path]:
    """Create complete expert annotation package.

    Args:
        data_dir: Base directory for output

    Returns:
        Dictionary mapping component names to file paths
    """
    try:
        output_dir = data_dir / "metadata" / "expert_annotation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create taxonomy and protocol
        taxonomy = BrainstemNucleiTaxonomy()
        protocol_file = taxonomy.save_protocol(output_dir)
        
        # Generate training materials
        training_files = taxonomy.generate_training_materials(output_dir)
        
        # Create package manifest
        package = {
            "annotation_protocol": protocol_file,
            "quick_reference": training_files[0],
            "annotation_checklist": training_files[1], 
            "common_mistakes_guide": training_files[2]
        }
        
        # Save manifest
        manifest_file = output_dir / "annotation_package_manifest.json"
        manifest_data = {
            "package_version": "1.0",
            "created_date": "2025-09-17",
            "components": {name: str(path) for name, path in package.items()},
            "nuclei_count": len(taxonomy.nuclei),
            "subdivisions": taxonomy.subdivisions,
            "functional_types": taxonomy.functional_types
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        package["manifest"] = manifest_file
        
        logger.info(f"Expert annotation package created in {output_dir}")
        logger.info(f"Components: {list(package.keys())}")
        
        return package
        
    except Exception as e:
        logger.error(f"Failed to create annotation package: {e}")
        return {}


if __name__ == "__main__":
    # Demo execution
    data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
    package = create_expert_annotation_package(data_dir)
    
    print("🧠 Expert Annotation Protocol Generated")
    print("=" * 50)
    for component, path in package.items():
        print(f"📄 {component}: {path}")
