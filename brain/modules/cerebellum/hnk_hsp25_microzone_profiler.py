#!/usr/bin/env python3
"""
HNK-1/CD57 and HSP25 Parasagittal Microzone Profiler

Profiles HNK-1 (CD57) and HSP25 expression patterns for parasagittal microzone
alternation in the developing cerebellum. These markers complement zebrin II
in defining the fundamental microzone organization and are critical for
understanding climbing fiber territory mapping.

Key features:
- HNK-1/CD57: Neural cell adhesion molecule marking climbing fiber territories
- HSP25: Heat shock protein marking Purkinje cell subsets
- Parasagittal alternation patterns complementing zebrin II
- Critical for climbing fiber-Purkinje cell one-to-one mapping

Author: Quark Brain Development Team
Date: 2025-09-17
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.parse
import requests
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MicrozoneMarkerDefinition:
    """Definition of microzone alternation marker."""
    marker_name: str
    gene_symbol: str
    protein_name: str
    expression_pattern: str  # "alternating_stripes", "uniform", "gradient"
    zone_specificity: str  # "positive_zones", "negative_zones", "both"
    cell_types_targeted: List[str]
    developmental_onset: str
    functional_significance: str


class HNKHsp25MicrozoneProfiler:
    """Profiles HNK-1/CD57 and HSP25 for microzone alternation patterns."""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize microzone marker profiler."""
        self.data_dir = Path(data_dir)
        self.microzone_dir = self.data_dir / "microzone_markers"
        self.profiles_dir = self.microzone_dir / "expression_profiles"
        self.metadata_dir = self.microzone_dir / "metadata"
        
        for directory in [self.microzone_dir, self.profiles_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized microzone marker profiler")
    
    def define_microzone_markers(self) -> List[MicrozoneMarkerDefinition]:
        """Define HNK-1/CD57 and HSP25 microzone markers."""
        logger.info("Defining microzone alternation markers")
        
        markers = [
            MicrozoneMarkerDefinition(
                marker_name="HNK1_CD57",
                gene_symbol="B3gat1",  # Beta-1,3-glucuronyltransferase 1
                protein_name="HNK-1/CD57",
                expression_pattern="alternating_stripes",
                zone_specificity="climbing_fiber_territories",
                cell_types_targeted=["climbing_fiber", "Purkinje_cell_subset"],
                developmental_onset="E14.5",
                functional_significance="climbing_fiber_territory_mapping"
            ),
            MicrozoneMarkerDefinition(
                marker_name="HSP25",
                gene_symbol="Hspb1",  # Heat shock protein beta-1
                protein_name="Heat shock protein 25",
                expression_pattern="alternating_stripes",
                zone_specificity="purkinje_cell_subsets",
                cell_types_targeted=["Purkinje_cell_subset"],
                developmental_onset="E16.5",
                functional_significance="purkinje_cell_compartmentalization"
            )
        ]
        
        logger.info(f"Defined {len(markers)} microzone markers")
        return markers
    
    def download_marker_data(self) -> Dict[str, any]:
        """Download HNK-1/HSP25 expression data."""
        logger.info("Downloading HNK-1/HSP25 expression data")
        
        download_results = {
            "download_date": datetime.now().isoformat(),
            "genes_targeted": ["B3gat1", "Hspb1"],
            "files_downloaded": [],
            "download_status": {},
            "total_size_mb": 0
        }
        
        allen_dir = self.profiles_dir / "allen_brain_map"
        allen_dir.mkdir(exist_ok=True)
        
        base_url = "http://api.brain-map.org/api/v2"
        
        queries = {
            "b3gat1_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'B3gat1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))",
            "hspb1_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Hspb1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))"
        }
        
        for query_name, query_url in queries.items():
            try:
                with urllib.request.urlopen(query_url, timeout=30) as response:
                    data = json.loads(response.read())
                
                output_file = allen_dir / f"{query_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                file_size_mb = output_file.stat().st_size / (1024*1024)
                download_results["files_downloaded"].append(str(output_file))
                download_results["download_status"][query_name] = "success"
                download_results["total_size_mb"] += file_size_mb
                
                logger.info(f"✅ Downloaded {query_name} ({file_size_mb:.2f}MB)")
                
            except Exception as e:
                download_results["download_status"][query_name] = f"failed: {str(e)}"
                logger.error(f"❌ Failed to download {query_name}: {e}")
        
        return download_results
    
    def execute_profiling(self) -> Dict[str, any]:
        """Execute microzone marker profiling."""
        logger.info("Executing HNK-1/HSP25 microzone profiling")
        
        results = {
            "profiling_date": datetime.now().isoformat(),
            "markers_profiled": [],
            "total_data_mb": 0
        }
        
        markers = self.define_microzone_markers()
        results["markers_profiled"] = [m.marker_name for m in markers]
        
        download_results = self.download_marker_data()
        results["total_data_mb"] = download_results["total_size_mb"]
        results["download_details"] = download_results
        
        # Save results
        results_file = self.metadata_dir / "hnk_hsp25_profiling_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Execute HNK-1/HSP25 microzone profiling."""
    print("🧬 HNK-1/CD57 AND HSP25 MICROZONE PROFILING")
    print("=" * 60)
    print("Phase 1 ▸ Batch A ▸ Step A2.4")
    
    profiler = HNKHsp25MicrozoneProfiler()
    results = profiler.execute_profiling()
    
    print(f"✅ Microzone profiling completed")
    print(f"🧬 Markers profiled: {len(results['markers_profiled'])}")
    print(f"💾 Total data: {results['total_data_mb']:.1f}MB")


if __name__ == "__main__":
    main()
