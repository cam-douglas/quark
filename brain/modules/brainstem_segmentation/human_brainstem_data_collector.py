#!/usr/bin/env python3
"""
Human Brainstem Data Collector

Pivots to human brain data for brainstem segmentation project.
Collects human-specific datasets and literature.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HumanBrainstemDataset:
    """Metadata for human brainstem dataset."""
    name: str
    authors: List[str]
    year: str
    modality: str  # MRI, histology, etc.
    resolution_um: Optional[float]
    structures: List[str]  # midbrain, pons, medulla, nuclei
    availability: str  # public, freesurfer, request, etc.
    url: Optional[str]
    pmid: Optional[str]
    doi: Optional[str]
    notes: str


def collect_human_brainstem_datasets() -> List[HumanBrainstemDataset]:
    """Collect human brainstem datasets from literature search."""
    
    datasets = []
    
    # FreeSurfer Brainstem Segmentation (Iglesias et al. 2015)
    datasets.append(HumanBrainstemDataset(
        name="FreeSurfer Brainstem Segmentation",
        authors=["Iglesias", "Van Leemput", "Bhatt", "Fischl"],
        year="2015",
        modality="MRI",
        resolution_um=1000,  # 1mm
        structures=["midbrain", "pons", "medulla", "superior cerebellar peduncle"],
        availability="public",
        url="https://freesurfer.net/",
        pmid="25776214",
        doi="10.1016/j.neuroimage.2015.03.012",
        notes="Bayesian segmentation, 4 brainstem structures, validated on 383 scans"
    ))
    
    # NextBrain Atlas (Casamitjana et al. 2024)
    datasets.append(HumanBrainstemDataset(
        name="NextBrain Histological Atlas",
        authors=["Casamitjana", "Mancini", "Robinson", "Iglesias"],
        year="2024",
        modality="histology+MRI",
        resolution_um=100,  # 100μm
        structures=["333 anatomical ROIs including brainstem nuclei"],
        availability="public",
        url="https://doi.org/10.1101/2024.02.05.579016",
        pmid=None,
        doi="10.1101/2024.02.05.579016",
        notes="Next-generation probabilistic atlas from 5 whole brain hemispheres"
    ))
    
    # Human Brainstem and Cerebellum Atlas (Agostinelli et al. 2022)
    datasets.append(HumanBrainstemDataset(
        name="Human Brainstem Chemoarchitecture Atlas",
        authors=["Agostinelli", "Seaman", "Saper"],
        year="2022",
        modality="7T MRI + histology",
        resolution_um=200,  # 200μm MRI
        structures=["cholinergic", "serotonergic", "catecholaminergic neurons", "cerebellum"],
        availability="public",
        url="https://www.jneurosci.org/content/43/2/221",
        pmid=None,
        doi="10.1523/JNEUROSCI.0587-22.2022",
        notes="High-resolution 7T MRI paired with detailed immunohistochemistry"
    ))
    
    # Brainstem Arousal Nuclei Atlas (Olchanyi et al. 2024)
    datasets.append(HumanBrainstemDataset(
        name="Brainstem Arousal Nuclei Atlas",
        authors=["Olchanyi", "Augustinack", "Lewis", "Iglesias", "Edlow"],
        year="2024",
        modality="diffusion MRI + histology",
        resolution_um=750,  # 750μm
        structures=["arousal nuclei", "consciousness networks"],
        availability="public",
        url="https://doi.org/10.1101/2024.09.26.24314117",
        pmid=None,
        doi="10.1101/2024.09.26.24314117",
        notes="Probabilistic atlas of brainstem arousal nuclei, Bayesian segmentation tool"
    ))
    
    # High-Resolution Interactive Brainstem Atlas (Adil et al. 2021)
    datasets.append(HumanBrainstemDataset(
        name="MR Histology Brainstem Atlas",
        authors=["Adil", "Calabrese", "Johnson", "Lad", "White"],
        year="2021",
        modality="7T MRI histology",
        resolution_um=50,  # 50μm anatomic, 200μm diffusion
        structures=["90 structures", "11 fiber bundles"],
        availability="request",
        url=None,
        pmid=None,
        doi="10.1093/NEUROS/NYAA447_644",
        notes="Unprecedented 50μm resolution, interactive 3D atlas"
    ))
    
    # Structural Connectivity of Brainstem Nuclei (Singh et al. 2022)
    datasets.append(HumanBrainstemDataset(
        name="Brainstem Nuclei Connectivity Atlas",
        authors=["Singh", "García-Gomar", "Cauzzo", "Bianciardi"],
        year="2022",
        modality="7T + 3T diffusion MRI",
        resolution_um=1700,  # 1.7mm at 7T
        structures=["15 autonomic/pain/limbic/sensory nuclei"],
        availability="public",
        url="https://doi.org/10.1002/hbm.25836",
        pmid=None,
        doi="10.1002/hbm.25836",
        notes="Structural connectome, probabilistic tractography, clinical translatability"
    ))
    
    # Arousal and Motor Brainstem Nuclei (García-Gomar et al. 2022)
    datasets.append(HumanBrainstemDataset(
        name="Arousal Motor Brainstem Connectome",
        authors=["García-Gomar", "Singh", "Cauzzo", "Bianciardi"],
        year="2022",
        modality="7T + 3T diffusion MRI",
        resolution_um=1700,
        structures=["18 arousal and motor brainstem nuclei"],
        availability="public",
        url="https://doi.org/10.1002/hbm.25962",
        pmid=None,
        doi="10.1002/hbm.25962",
        notes="Living humans, 7T/3T comparison, consciousness and motor networks"
    ))
    
    # Novel True-Color Brainstem Map (You & Park 2023)
    datasets.append(HumanBrainstemDataset(
        name="True-Color Sectioned Brainstem Atlas",
        authors=["You", "Park"],
        year="2023",
        modality="sectioned images",
        resolution_um=60,  # 0.06mm pixel size
        structures=["212 structures including nuclei and tracts"],
        availability="public",
        url="https://doi.org/10.3346/jkms.2023.38.e76",
        pmid=None,
        doi="10.3346/jkms.2023.38.e76",
        notes="True color sectioned images, 0.2mm intervals, volume model"
    ))
    
    # Limbic Brainstem Nuclei Atlas (Levinson et al. 2022)
    datasets.append(HumanBrainstemDataset(
        name="Limbic Brainstem Structural Connectivity Atlas",
        authors=["Levinson", "Miller", "Iftekhar", "Bari"],
        year="2022",
        modality="diffusion MRI",
        resolution_um=None,
        structures=["locus coeruleus", "VTA", "PAG", "dorsal raphe", "NTS"],
        availability="public",
        url="https://doi.org/10.3389/fnimg.2022.1009399",
        pmid=None,
        doi="10.3389/fnimg.2022.1009399",
        notes="197 HCP subjects, MNI atlas, monoaminergic nuclei"
    ))
    
    # Human Thalamic Nuclei Atlas (Saranathan et al. 2021)
    datasets.append(HumanBrainstemDataset(
        name="High-Resolution Thalamic Nuclei Atlas",
        authors=["Saranathan", "Iglehart", "Monti", "Rutt"],
        year="2021",
        modality="high-resolution MRI",
        resolution_um=700,  # 0.7x0.7x0.5mm
        structures=["thalamic nuclei", "adjacent brainstem"],
        availability="public",
        url="https://doi.org/10.1038/s41597-021-01062-y",
        pmid=None,
        doi="10.1038/s41597-021-01062-y",
        notes="9 healthy subjects, Morel atlas guide, 3T and 7T"
    ))
    
    return datasets


def identify_priority_datasets() -> List[str]:
    """Identify priority datasets for immediate acquisition."""
    
    priorities = [
        "FreeSurfer Brainstem Segmentation",  # Readily available, widely used
        "NextBrain Histological Atlas",       # Most comprehensive, publicly available
        "Human Brainstem Chemoarchitecture Atlas",  # 7T MRI + histology
        "Brainstem Arousal Nuclei Atlas",     # Recent, consciousness focus
        "Brainstem Nuclei Connectivity Atlas" # Structural connectivity
    ]
    
    return priorities


def generate_download_plan() -> Dict[str, str]:
    """Generate specific download plan for human datasets."""
    
    plan = {
        "freesurfer": {
            "method": "Direct installation",
            "url": "https://freesurfer.net/fswiki/DownloadAndInstall",
            "command": "Download FreeSurfer 7.4+, includes brainstem segmentation",
            "data_location": "FreeSurfer installation/subjects/fsaverage/mri/"
        },
        "nextbrain": {
            "method": "Public repository",
            "url": "https://doi.org/10.1101/2024.02.05.579016",
            "command": "Download from bioRxiv supplementary materials",
            "data_location": "Supplementary files and online visualization tool"
        },
        "agostinelli_atlas": {
            "method": "Journal supplementary",
            "url": "https://www.jneurosci.org/content/43/2/221",
            "command": "Download from Journal of Neuroscience supplementary data",
            "data_location": "Supplementary materials section"
        },
        "arousal_nuclei": {
            "method": "medRxiv preprint",
            "url": "https://doi.org/10.1101/2024.09.26.24314117",
            "command": "Download atlas and segmentation tool from preprint",
            "data_location": "Supplementary files"
        },
        "bianciardi_connectivity": {
            "method": "Human Brain Mapping data",
            "url": "https://doi.org/10.1002/hbm.25836",
            "command": "Download connectivity matrices and atlas",
            "data_location": "Journal supplementary data"
        }
    }
    
    return plan


def create_human_data_report(output_dir: Path) -> None:
    """Create comprehensive report for human brainstem data."""
    
    datasets = collect_human_brainstem_datasets()
    priorities = identify_priority_datasets()
    download_plan = generate_download_plan()
    
    report_file = output_dir / "human_brainstem_data_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Human Brainstem Segmentation Data Report\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        f.write("## 🧠 Pivot to Human Brain Data\n\n")
        f.write("**Rationale**: Focus on human brainstem anatomy for clinical relevance\n")
        f.write("**Advantage**: Rich literature, validated atlases, clinical applicability\n")
        f.write("**Challenge**: Higher complexity, fewer developmental stages\n\n")
        
        f.write(f"## 📊 Available Human Brainstem Datasets ({len(datasets)} total)\n\n")
        
        for i, dataset in enumerate(datasets, 1):
            f.write(f"### {i}. {dataset.name}\n")
            f.write(f"- **Authors**: {', '.join(dataset.authors)}\n")
            f.write(f"- **Year**: {dataset.year}\n")
            f.write(f"- **Modality**: {dataset.modality}\n")
            if dataset.resolution_um:
                f.write(f"- **Resolution**: {dataset.resolution_um} μm\n")
            f.write(f"- **Structures**: {', '.join(dataset.structures) if isinstance(dataset.structures, list) else dataset.structures}\n")
            f.write(f"- **Availability**: {dataset.availability}\n")
            if dataset.url:
                f.write(f"- **URL**: {dataset.url}\n")
            if dataset.pmid:
                f.write(f"- **PMID**: {dataset.pmid}\n")
            f.write(f"- **Notes**: {dataset.notes}\n\n")
        
        f.write("## 🎯 Priority Datasets\n\n")
        for i, priority in enumerate(priorities, 1):
            f.write(f"{i}. **{priority}**\n")
            # Find matching dataset
            matching = [d for d in datasets if d.name == priority][0]
            f.write(f"   - Availability: {matching.availability}\n")
            f.write(f"   - Modality: {matching.modality}\n\n")
        
        f.write("## 📋 Download Plan\n\n")
        for name, plan in download_plan.items():
            f.write(f"### {name.replace('_', ' ').title()}\n")
            f.write(f"- **Method**: {plan['method']}\n")
            f.write(f"- **URL**: {plan['url']}\n")
            f.write(f"- **Action**: {plan['command']}\n")
            f.write(f"- **Data Location**: {plan['data_location']}\n\n")
        
        f.write("## 🔬 Human Brainstem Nuclei (Updated Catalog)\n\n")
        
        # Human-specific nuclei catalog
        human_nuclei = {
            "midbrain": [
                "Periaqueductal gray (PAG)",
                "Superior colliculus",
                "Inferior colliculus", 
                "Red nucleus",
                "Substantia nigra",
                "Ventral tegmental area (VTA)",
                "Oculomotor nucleus (CN III)",
                "Trochlear nucleus (CN IV)",
                "Edinger-Westphal nucleus"
            ],
            "pons": [
                "Locus coeruleus",
                "Pontine nuclei",
                "Trigeminal motor nucleus (CN V)",
                "Abducens nucleus (CN VI)",
                "Facial nucleus (CN VII)",
                "Superior olivary complex",
                "Parabrachial nucleus",
                "Pedunculopontine nucleus"
            ],
            "medulla": [
                "Raphe nuclei (magnus, pallidus, obscurus)",
                "Nucleus tractus solitarius (NTS)",
                "Dorsal motor nucleus of vagus (CN X)",
                "Hypoglossal nucleus (CN XII)",
                "Nucleus ambiguus",
                "Inferior olivary complex",
                "Lateral reticular nucleus",
                "Pre-Bötzinger complex",
                "Area postrema"
            ]
        }
        
        for region, nuclei in human_nuclei.items():
            f.write(f"### {region.title()}\n")
            for nucleus in nuclei:
                f.write(f"- {nucleus}\n")
            f.write("\n")
        
        f.write("## 🚀 Implementation Strategy\n\n")
        f.write("### Phase 1: Immediate Actions\n")
        f.write("1. Install FreeSurfer 7.4+ for baseline brainstem segmentation\n")
        f.write("2. Download NextBrain atlas for detailed anatomical reference\n")
        f.write("3. Acquire Agostinelli 7T MRI + histology atlas\n")
        f.write("4. Set up human-specific registration pipeline\n\n")
        
        f.write("### Phase 2: Advanced Integration\n")
        f.write("1. Integrate arousal nuclei atlas for consciousness networks\n")
        f.write("2. Add structural connectivity data from Bianciardi lab\n")
        f.write("3. Incorporate true-color sectioned atlas for validation\n")
        f.write("4. Develop multi-modal segmentation pipeline\n\n")
        
        f.write("### Phase 3: Validation & Clinical Application\n")
        f.write("1. Validate against multiple atlas sources\n")
        f.write("2. Test on clinical datasets (AD, PD, PSP, MSA)\n")
        f.write("3. Develop automated segmentation tools\n")
        f.write("4. Create interactive visualization interface\n\n")
        
        f.write("## 🎯 Expected Outcomes\n\n")
        f.write("- **Comprehensive human brainstem atlas** with 25+ nuclei\n")
        f.write("- **Multi-resolution pipeline** (50μm to 1mm)\n")
        f.write("- **Clinical validation** on neurological disorders\n")
        f.write("- **Open-source tools** for brainstem segmentation\n")
        f.write("- **Educational resources** for neuroanatomy\n\n")
        
    # Save JSON version
    json_file = output_dir / "human_brainstem_datasets.json"
    with open(json_file, 'w') as f:
        json.dump({
            "datasets": [asdict(d) for d in datasets],
            "priorities": priorities,
            "download_plan": download_plan
        }, f, indent=2)
    
    logger.info(f"Generated human brainstem data report at {report_file}")
    logger.info(f"Saved machine-readable data at {json_file}")


def main():
    """Execute human brainstem data collection and planning."""
    
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧠 HUMAN BRAINSTEM DATA COLLECTION")
    print("=" * 50)
    
    # Generate comprehensive report
    create_human_data_report(output_dir)
    
    # Show summary
    datasets = collect_human_brainstem_datasets()
    priorities = identify_priority_datasets()
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Identified {len(datasets)} human brainstem datasets")
    print(f"✅ Prioritized {len(priorities)} datasets for immediate acquisition")
    print(f"✅ Generated download plan with specific URLs and methods")
    print(f"✅ Updated nucleus catalog for human anatomy")
    
    print(f"\n🎯 TOP PRIORITY DATASETS:")
    for i, priority in enumerate(priorities[:3], 1):
        print(f"  {i}. {priority}")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Install FreeSurfer for immediate brainstem segmentation capability")
    print("2. Download NextBrain atlas for comprehensive anatomical reference")
    print("3. Begin human-specific nucleus catalog compilation")
    print("4. Adapt registration pipeline for human brain anatomy")


if __name__ == "__main__":
    main()
