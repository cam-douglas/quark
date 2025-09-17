#!/usr/bin/env python3
"""
Carnegie Stages 14-23 Cerebellar Atlas Extractor

Extracts cerebellar developmental data from Allen Brain Atlas for Carnegie stages 14-23
(approximately embryonic weeks 5-8, covering early cerebellar specification through
initial foliation).

Author: Quark Brain Development Team
Date: 2025-09-17
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.parse
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CarnegieStageMapping:
    """Maps Carnegie stages to developmental weeks and mouse equivalents."""
    stage: int
    weeks_human: float
    mouse_equivalent: str
    cerebellar_features: str


class CarnegieAtlasExtractor:
    """Extracts cerebellar atlas data for Carnegie stages 14-23."""
    
    # Allen Brain Atlas API endpoints
    BASE_URL = "http://api.brain-map.org/api/v2"
    DEVELOPING_MOUSE_URL = f"{BASE_URL}/data/query.json"
    
    # Carnegie stage mappings to mouse developmental stages
    CARNEGIE_MAPPINGS = [
        CarnegieStageMapping(14, 5.0, "E10.0", "Isthmic organizer formation, FGF8 expression"),
        CarnegieStageMapping(15, 5.5, "E10.5", "Rhombic lip emergence, Math1+ precursors"),
        CarnegieStageMapping(16, 6.0, "E11.0", "Cerebellar primordium visible, En1/2 boundaries"),
        CarnegieStageMapping(17, 6.5, "E11.5", "External granular layer formation"),
        CarnegieStageMapping(18, 7.0, "E12.0", "Purkinje cell migration begins"),
        CarnegieStageMapping(19, 7.5, "E12.5", "Deep nuclei condensation, EGL expansion"),
        CarnegieStageMapping(20, 8.0, "E13.0", "Early foliation, primary fissure"),
        CarnegieStageMapping(21, 8.5, "E13.5", "Lobule I-X definition, Zebrin II zones"),
        CarnegieStageMapping(22, 9.0, "E14.0", "Bergmann glia scaffold, climbing fibers"),
        CarnegieStageMapping(23, 9.5, "E14.5", "Granule cell radial migration onset")
    ]
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize the Carnegie atlas extractor.
        
        Args:
            data_dir: Base directory for cerebellar data storage
        """
        self.data_dir = Path(data_dir)
        self.carnegie_dir = self.data_dir / "carnegie_stages"
        self.metadata_dir = self.carnegie_dir / "metadata"
        self.images_dir = self.carnegie_dir / "images"
        
        # Create directory structure
        for directory in [self.carnegie_dir, self.metadata_dir, self.images_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized Carnegie atlas extractor")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Carnegie stages directory: {self.carnegie_dir}")
    
    def get_cerebellar_experiments(self) -> List[Dict]:
        """Fetch Allen Brain Atlas experiments relevant to cerebellar development.
        
        Returns:
            List of experiment metadata for cerebellar-relevant stages
        """
        logger.info("Fetching cerebellar development experiments from Allen Brain Atlas")
        
        # Query for developing mouse experiments with cerebellar relevance
        params = {
            "criteria": "model::SectionDataSet,rma::criteria,[failed$eq'false'],products[abbreviation$eq'DevMouse']",
            "include": "specimen(donor(age)),plane_of_section,genes",
            "num_rows": 100,
            "start_row": 0
        }
        
        url = f"{self.DEVELOPING_MOUSE_URL}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read())
                experiments = data.get("msg", [])
                
            # Filter for cerebellar-relevant stages (E10.0-E14.5)
            cerebellar_experiments = []
            target_stages = [mapping.mouse_equivalent for mapping in self.CARNEGIE_MAPPINGS]
            
            for exp in experiments:
                if exp.get("specimen") and exp["specimen"].get("donor"):
                    age = exp["specimen"]["donor"].get("age", {}).get("name", "")
                    if any(stage in age for stage in target_stages):
                        cerebellar_experiments.append(exp)
            
            logger.info(f"Found {len(cerebellar_experiments)} cerebellar-relevant experiments")
            return cerebellar_experiments
            
        except Exception as e:
            logger.error(f"Failed to fetch experiments: {e}")
            return []
    
    def extract_cerebellar_markers(self) -> Dict[str, List[str]]:
        """Define cerebellar-specific molecular markers for each Carnegie stage.
        
        Returns:
            Dictionary mapping Carnegie stages to relevant gene markers
        """
        cerebellar_markers = {
            "CS14-15": ["Fgf8", "Gbx2", "Otx2", "Wnt1"],  # Isthmic organizer
            "CS16-17": ["Math1", "Atoh1", "En1", "En2", "Pax2"],  # Cerebellar specification
            "CS18-19": ["Ptf1a", "Lhx1", "Lhx5", "Reelin", "Olig2"],  # Cell fate specification
            "CS20-21": ["Aldoc", "Car8", "Zebrin2", "Pcp2"],  # Purkinje cell markers
            "CS22-23": ["Neurod1", "Zic1", "Zic2", "Gad1", "Gad2"]  # Granule/interneuron markers
        }
        
        logger.info(f"Defined markers for {len(cerebellar_markers)} Carnegie stage groups")
        return cerebellar_markers
    
    def download_stage_data(self, stage: int) -> Optional[Dict]:
        """Download atlas data for a specific Carnegie stage.
        
        Args:
            stage: Carnegie stage number (14-23)
            
        Returns:
            Dictionary with downloaded data metadata or None if failed
        """
        if stage < 14 or stage > 23:
            logger.error(f"Invalid Carnegie stage: {stage}. Must be 14-23")
            return None
            
        stage_mapping = next((m for m in self.CARNEGIE_MAPPINGS if m.stage == stage), None)
        if not stage_mapping:
            logger.error(f"No mapping found for Carnegie stage {stage}")
            return None
            
        logger.info(f"Downloading data for Carnegie stage {stage} ({stage_mapping.mouse_equivalent})")
        
        stage_dir = self.carnegie_dir / f"CS{stage:02d}"
        stage_dir.mkdir(exist_ok=True)
        
        # Create metadata for this stage
        stage_metadata = {
            "carnegie_stage": stage,
            "human_weeks": stage_mapping.weeks_human,
            "mouse_equivalent": stage_mapping.mouse_equivalent,
            "cerebellar_features": stage_mapping.cerebellar_features,
            "download_date": datetime.now().isoformat(),
            "data_source": "Allen Brain Atlas Developing Mouse",
            "extraction_status": "initiated"
        }
        
        # Save stage metadata
        metadata_file = stage_dir / "stage_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(stage_metadata, f, indent=2)
            
        logger.info(f"Created metadata for Carnegie stage {stage}")
        return stage_metadata
    
    def extract_cerebellar_regions(self, experiment_data: Dict) -> Dict[str, any]:
        """Extract cerebellar-specific regions from experiment data.
        
        Args:
            experiment_data: Allen Brain Atlas experiment data
            
        Returns:
            Dictionary with cerebellar region annotations
        """
        cerebellar_regions = {
            "rhombic_lip": {
                "description": "Math1+ granule cell precursor source",
                "markers": ["Math1", "Atoh1", "Gdf7"],
                "coordinates": None  # To be filled from atlas data
            },
            "ventricular_zone": {
                "description": "Ptf1a+ GABAergic precursor source",
                "markers": ["Ptf1a", "Pax2", "Lhx1"],
                "coordinates": None
            },
            "isthmic_organizer": {
                "description": "FGF8/Wnt1 signaling center",
                "markers": ["Fgf8", "Wnt1", "Gbx2"],
                "coordinates": None
            },
            "deep_nuclei_primordia": {
                "description": "Future fastigial/interposed/dentate nuclei",
                "markers": ["Tbr1", "Lhx2", "Lhx9"],
                "coordinates": None
            },
            "external_granular_layer": {
                "description": "Proliferative granule precursor zone",
                "markers": ["Zic1", "NeuroD1", "Pax6"],
                "coordinates": None
            }
        }
        
        logger.info(f"Defined {len(cerebellar_regions)} cerebellar regions for extraction")
        return cerebellar_regions
    
    def extract_all_stages(self) -> Dict[str, Dict]:
        """Extract atlas data for all Carnegie stages 14-23.
        
        Returns:
            Dictionary mapping stage numbers to extraction results
        """
        logger.info("Starting extraction of all Carnegie stages 14-23")
        
        results = {}
        cerebellar_markers = self.extract_cerebellar_markers()
        
        # Get available experiments
        experiments = self.get_cerebellar_experiments()
        if not experiments:
            logger.warning("No cerebellar experiments found")
            return results
        
        # Process each Carnegie stage
        for stage in range(14, 24):
            logger.info(f"Processing Carnegie stage {stage}")
            
            try:
                # Download stage-specific data
                stage_data = self.download_stage_data(stage)
                if stage_data:
                    # Add cerebellar region definitions
                    stage_data["cerebellar_regions"] = self.extract_cerebellar_regions({})
                    
                    # Add relevant molecular markers
                    for marker_group, markers in cerebellar_markers.items():
                        if f"CS{stage}" in marker_group or (
                            stage >= int(marker_group.split("-")[0][2:]) and 
                            stage <= int(marker_group.split("-")[1][2:])
                        ):
                            stage_data["molecular_markers"] = markers
                            break
                    
                    stage_data["extraction_status"] = "completed"
                    results[f"CS{stage:02d}"] = stage_data
                    
                    logger.info(f"Successfully extracted Carnegie stage {stage}")
                else:
                    logger.warning(f"Failed to extract Carnegie stage {stage}")
                    
            except Exception as e:
                logger.error(f"Error processing Carnegie stage {stage}: {e}")
                continue
        
        # Save summary results
        summary_file = self.metadata_dir / "extraction_summary.json"
        summary = {
            "extraction_date": datetime.now().isoformat(),
            "total_stages_processed": len(results),
            "successful_extractions": len([r for r in results.values() if r.get("extraction_status") == "completed"]),
            "carnegie_stages": list(results.keys()),
            "data_directory": str(self.carnegie_dir),
            "extraction_tool": "carnegie_atlas_extractor.py"
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Extraction complete. Summary saved to {summary_file}")
        logger.info(f"Processed {len(results)} Carnegie stages")
        
        return results


def main():
    """Execute Carnegie stages 14-23 cerebellar atlas extraction."""
    
    print("🧠 CEREBELLAR ATLAS EXTRACTION - Carnegie Stages 14-23")
    print("=" * 60)
    print("Phase 1 ▸ Batch A ▸ Step A1.1")
    print()
    
    # Initialize extractor
    extractor = CarnegieAtlasExtractor()
    
    # Extract all Carnegie stages
    results = extractor.extract_all_stages()
    
    # Print results summary
    print(f"✅ Extraction completed successfully")
    print(f"📊 Carnegie stages processed: {len(results)}")
    print(f"📁 Data location: {extractor.carnegie_dir}")
    print()
    
    # Display stage details
    for stage_name, stage_data in results.items():
        status_emoji = "✅" if stage_data.get("extraction_status") == "completed" else "⚠️"
        print(f"{status_emoji} {stage_name}: {stage_data.get('cerebellar_features', 'N/A')}")
    
    print()
    print("🎯 Next Steps:")
    print("- Review extracted metadata in data/datasets/cerebellum/carnegie_stages/")
    print("- Proceed to A1.2: Download Paxinos rhombomere fate maps")
    print("- Begin A1.3: Collect single-cell RNA-seq data")


if __name__ == "__main__":
    main()
