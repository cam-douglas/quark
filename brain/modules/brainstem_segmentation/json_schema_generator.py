#!/usr/bin/env python3
"""
JSON Schema Generator - Step 3.F4

Creates hierarchical JSON label schema for brainstem segmentation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def create_brainstem_label_schema() -> Dict[str, Any]:
    """Create hierarchical JSON schema for brainstem labels."""
    
    schema = {
        "schema_version": "1.0",
        "generated": datetime.now().isoformat(),
        "description": "Hierarchical brainstem segmentation schema for human brain",
        "coordinate_system": "MNI152",
        "resolution_mm": 0.2,
        "data_source": "NextBrain Atlas + Human Literature",
        
        "hierarchy": {
            "brainstem": {
                "label_id": 9,
                "name": "Brain-Stem",
                "subdivisions": {
                    "midbrain": {
                        "label_range": [100, 119],
                        "name": "Midbrain",
                        "nuclei": {
                            "periaqueductal_gray": {
                                "label_id": 100,
                                "name": "Periaqueductal Gray",
                                "abbreviation": "PAG",
                                "functional_class": "autonomic",
                                "neurotransmitter": "GABA/glutamate",
                                "clinical_relevance": "pain modulation, arousal"
                            },
                            "edinger_westphal": {
                                "label_id": 101,
                                "name": "Edinger-Westphal Nucleus",
                                "abbreviation": "EW",
                                "functional_class": "autonomic",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "pupillary constriction"
                            },
                            "substantia_nigra_compacta": {
                                "label_id": 102,
                                "name": "Substantia Nigra pars compacta",
                                "abbreviation": "SNc",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "dopamine",
                                "clinical_relevance": "Parkinson's disease"
                            },
                            "substantia_nigra_reticulata": {
                                "label_id": 103,
                                "name": "Substantia Nigra pars reticulata",
                                "abbreviation": "SNr",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "GABA",
                                "clinical_relevance": "basal ganglia output"
                            },
                            "red_nucleus": {
                                "label_id": 4,
                                "name": "Red Nucleus",
                                "abbreviation": "RN",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "glutamate",
                                "clinical_relevance": "motor coordination"
                            },
                            "oculomotor_nucleus": {
                                "label_id": 105,
                                "name": "Oculomotor Nucleus",
                                "abbreviation": "III",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "eye movements"
                            },
                            "trochlear_nucleus": {
                                "label_id": 106,
                                "name": "Trochlear Nucleus",
                                "abbreviation": "IV",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "superior oblique control"
                            },
                            "superior_colliculus": {
                                "label_id": 107,
                                "name": "Superior Colliculus",
                                "abbreviation": "SC",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "glutamate/GABA",
                                "clinical_relevance": "visual orientation"
                            },
                            "inferior_colliculus": {
                                "label_id": 85,
                                "name": "Inferior Colliculus",
                                "abbreviation": "IC",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "GABA/glycine",
                                "clinical_relevance": "auditory processing"
                            },
                            "ventral_tegmental_area": {
                                "label_id": 109,
                                "name": "Ventral Tegmental Area",
                                "abbreviation": "VTA",
                                "functional_class": "consciousness",
                                "neurotransmitter": "dopamine",
                                "clinical_relevance": "reward and motivation"
                            },
                            "reticular_formation_midbrain": {
                                "label_id": 110,
                                "name": "Reticular Formation (midbrain)",
                                "abbreviation": "RF-MB",
                                "functional_class": "consciousness",
                                "neurotransmitter": "mixed",
                                "clinical_relevance": "arousal and consciousness"
                            },
                            "interpeduncular_nucleus": {
                                "label_id": 111,
                                "name": "Interpeduncular Nucleus",
                                "abbreviation": "IPN",
                                "functional_class": "autonomic",
                                "neurotransmitter": "GABA/glutamate",
                                "clinical_relevance": "habenulo-interpeduncular pathway"
                            }
                        }
                    },
                    "pons": {
                        "label_range": [120, 139],
                        "name": "Pons",
                        "nuclei": {
                            "pontine_nuclei": {
                                "label_id": 29,
                                "name": "Pontine Nuclei",
                                "abbreviation": "PN",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "glutamate",
                                "clinical_relevance": "cerebellar relay for motor control"
                            },
                            "locus_coeruleus": {
                                "label_id": 121,
                                "name": "Locus Coeruleus",
                                "abbreviation": "LC",
                                "functional_class": "consciousness",
                                "neurotransmitter": "norepinephrine",
                                "clinical_relevance": "noradrenergic arousal system"
                            },
                            "abducens_nucleus": {
                                "label_id": 122,
                                "name": "Abducens Nucleus",
                                "abbreviation": "VI",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "lateral rectus eye movement"
                            },
                            "facial_motor_nucleus": {
                                "label_id": 123,
                                "name": "Facial Motor Nucleus",
                                "abbreviation": "VII",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "facial expression control"
                            },
                            "superior_olivary_complex": {
                                "label_id": 124,
                                "name": "Superior Olivary Complex",
                                "abbreviation": "SOC",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "GABA/glycine",
                                "clinical_relevance": "binaural hearing"
                            },
                            "trigeminal_motor_nucleus": {
                                "label_id": 125,
                                "name": "Trigeminal Motor Nucleus",
                                "abbreviation": "Vmo",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "mastication control"
                            },
                            "trigeminal_sensory_nuclei": {
                                "label_id": 126,
                                "name": "Trigeminal Sensory Nuclei",
                                "abbreviation": "Vsen",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "glutamate",
                                "clinical_relevance": "facial sensation"
                            },
                            "vestibular_nuclei_pontine": {
                                "label_id": 127,
                                "name": "Vestibular Nuclei (pontine portion)",
                                "abbreviation": "VN",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "GABA/glutamate",
                                "clinical_relevance": "balance and spatial orientation"
                            },
                            "parabrachial_nuclei": {
                                "label_id": 128,
                                "name": "Parabrachial Nuclei",
                                "abbreviation": "PBN",
                                "functional_class": "autonomic",
                                "neurotransmitter": "glutamate/GABA",
                                "clinical_relevance": "autonomic and respiratory control"
                            },
                            "raphe_pontis": {
                                "label_id": 129,
                                "name": "Raphe Pontis",
                                "abbreviation": "RPo",
                                "functional_class": "consciousness",
                                "neurotransmitter": "serotonin",
                                "clinical_relevance": "serotonergic modulation"
                            },
                            "reticular_formation_pontine": {
                                "label_id": 130,
                                "name": "Reticular Formation (pontine)",
                                "abbreviation": "RF-PN",
                                "functional_class": "consciousness",
                                "neurotransmitter": "mixed",
                                "clinical_relevance": "sleep-wake regulation"
                            },
                            "kolliker_fuse_nucleus": {
                                "label_id": 131,
                                "name": "Kölliker-Fuse Nucleus",
                                "abbreviation": "KF",
                                "functional_class": "autonomic",
                                "neurotransmitter": "glutamate",
                                "clinical_relevance": "respiratory control"
                            }
                        }
                    },
                    "medulla": {
                        "label_range": [140, 159],
                        "name": "Medulla Oblongata",
                        "nuclei": {
                            "nucleus_ambiguus": {
                                "label_id": 140,
                                "name": "Nucleus Ambiguus",
                                "abbreviation": "NA",
                                "functional_class": "autonomic",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "pharyngeal/laryngeal motor control"
                            },
                            "hypoglossal_nucleus": {
                                "label_id": 141,
                                "name": "Hypoglossal Nucleus",
                                "abbreviation": "XII",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "tongue movement"
                            },
                            "dorsal_motor_nucleus_vagus": {
                                "label_id": 142,
                                "name": "Dorsal Motor Nucleus of Vagus",
                                "abbreviation": "DMNX",
                                "functional_class": "autonomic",
                                "neurotransmitter": "acetylcholine",
                                "clinical_relevance": "parasympathetic visceral control"
                            },
                            "nucleus_tractus_solitarius": {
                                "label_id": 143,
                                "name": "Nucleus Tractus Solitarius",
                                "abbreviation": "NTS",
                                "functional_class": "autonomic",
                                "neurotransmitter": "glutamate/GABA",
                                "clinical_relevance": "visceral sensory processing"
                            },
                            "inferior_olivary_complex": {
                                "label_id": 144,
                                "name": "Inferior Olivary Complex",
                                "abbreviation": "IO",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "glutamate",
                                "clinical_relevance": "cerebellar motor learning"
                            },
                            "raphe_magnus": {
                                "label_id": 145,
                                "name": "Raphe Magnus",
                                "abbreviation": "RMg",
                                "functional_class": "autonomic",
                                "neurotransmitter": "serotonin",
                                "clinical_relevance": "descending pain modulation"
                            },
                            "raphe_pallidus": {
                                "label_id": 146,
                                "name": "Raphe Pallidus",
                                "abbreviation": "RPa",
                                "functional_class": "autonomic",
                                "neurotransmitter": "serotonin",
                                "clinical_relevance": "thermoregulation"
                            },
                            "pre_botzinger_complex": {
                                "label_id": 147,
                                "name": "Pre-Bötzinger Complex",
                                "abbreviation": "preBötC",
                                "functional_class": "autonomic",
                                "neurotransmitter": "glutamate/GABA",
                                "clinical_relevance": "respiratory rhythm generation"
                            },
                            "botzinger_complex": {
                                "label_id": 148,
                                "name": "Bötzinger Complex",
                                "abbreviation": "BötC",
                                "functional_class": "autonomic",
                                "neurotransmitter": "GABA/glycine",
                                "clinical_relevance": "expiratory control"
                            },
                            "rostral_ventrolateral_medulla": {
                                "label_id": 149,
                                "name": "Rostral Ventrolateral Medulla",
                                "abbreviation": "RVLM",
                                "functional_class": "autonomic",
                                "neurotransmitter": "glutamate",
                                "clinical_relevance": "cardiovascular control"
                            },
                            "gracile_cuneate_nuclei": {
                                "label_id": 150,
                                "name": "Gracile and Cuneate Nuclei",
                                "abbreviation": "GC",
                                "functional_class": "sensorimotor",
                                "neurotransmitter": "glutamate",
                                "clinical_relevance": "proprioceptive relay"
                            },
                            "reticular_formation_medullary": {
                                "label_id": 151,
                                "name": "Reticular Formation (medullary)",
                                "abbreviation": "RF-MD",
                                "functional_class": "autonomic",
                                "neurotransmitter": "mixed",
                                "clinical_relevance": "autonomic integration"
                            }
                        }
                    }
                }
            }
        },
        
        "color_mapping": {
            "sensorimotor": [255, 100, 100],  # Red
            "autonomic": [100, 255, 100],     # Green  
            "consciousness": [100, 100, 255], # Blue
            "mixed": [255, 255, 100]          # Yellow
        },
        
        "validation_criteria": {
            "anatomical_accuracy": "±200μm from literature coordinates",
            "dice_threshold": 0.87,
            "minimum_volume_voxels": 100,
            "connectivity_validation": "Required for consciousness nuclei"
        },
        
        "nextbrain_mapping": {
            "existing_labels": {
                4: "Left-Red-Nucleus",
                9: "Brain-Stem", 
                19: "Pons-corticopontine-pontocerebellar-fibers",
                29: "Pons-pontine-nuclei",
                85: "Central-nucleus-of-inferior-colliculus",
                99: "Medulla_oblongata"
            },
            "new_labels_needed": [
                "PAG", "VTA", "LC", "NTS", "DRN", "RMg", "IO", "SC", "SN", "PBN", "SOC"
            ]
        }
    }
    
    return schema


def save_schema(schema: Dict[str, Any], output_dir: Path) -> None:
    """Save JSON schema to file."""
    
    schema_file = output_dir / "brainstem_labels_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"✅ Step 3.F4 Complete: JSON schema saved to {schema_file}")


def main():
    """Execute Step 3.F4: Draft hierarchical JSON label schema."""
    
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏷️ STEP 3.F4 - JSON LABEL SCHEMA")
    print("=" * 40)
    
    # Create schema
    schema = create_brainstem_label_schema()
    
    # Save schema
    save_schema(schema, output_dir)
    
    # Summary
    total_nuclei = len(schema["hierarchy"]["brainstem"]["subdivisions"]["midbrain"]["nuclei"]) + \
                   len(schema["hierarchy"]["brainstem"]["subdivisions"]["pons"]["nuclei"]) + \
                   len(schema["hierarchy"]["brainstem"]["subdivisions"]["medulla"]["nuclei"])
    
    print(f"📊 Schema includes {total_nuclei} detailed nuclei")
    print(f"🎯 Functional classes: 4 (sensorimotor, autonomic, consciousness, mixed)")
    print(f"🏷️ Label ranges assigned for future expansion")
    print(f"🔗 NextBrain integration mapping included")
    
    print(f"\n✅ STEPS 1-3 COMPLETE!")
    print("   Step 1.F2: ✅ Data collected and registered")
    print("   Step 2.F2: ✅ Literature review and nucleus catalog")  
    print("   Step 3.F4: ✅ JSON label schema created")


if __name__ == "__main__":
    main()
