#!/usr/bin/env python3
"""
Lhx1/5 and Pax2 GABAergic Interneuron Specification Tracer

Traces Lhx1, Lhx5, and Pax2 molecular markers for GABAergic interneuron
specification in the developing cerebellum. These transcription factors
are critical for specifying basket cells, stellate cells, and Golgi cells
from Ptf1a+ ventricular zone progenitors.

Key molecular pathways:
- Ptf1a → Lhx1/Lhx5 → GABAergic fate specification
- Pax2 → GABAergic interneuron differentiation
- Gad1/Gad2 → GABA synthesis pathway activation

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
class GABAergicMarkerDefinition:
    """Definition of GABAergic interneuron marker."""
    marker_name: str
    gene_symbol: str
    expression_domain: str
    cell_types_targeted: List[str]
    expression_onset: str
    expression_peak: str
    expression_maintenance: str
    downstream_targets: List[str]
    spatial_pattern: str


class LhxPax2GABAergicTracer:
    """Traces Lhx1/5 and Pax2 markers for GABAergic interneuron specification."""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize GABAergic marker tracer."""
        self.data_dir = Path(data_dir)
        self.gabaergic_dir = self.data_dir / "gabaergic_markers"
        self.expression_dir = self.gabaergic_dir / "expression_data"
        self.metadata_dir = self.gabaergic_dir / "metadata"
        
        for directory in [self.gabaergic_dir, self.expression_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized GABAergic marker tracer")
    
    def define_gabaergic_markers(self) -> List[GABAergicMarkerDefinition]:
        """Define GABAergic interneuron specification markers."""
        logger.info("Defining GABAergic interneuron specification markers")
        
        markers = [
            GABAergicMarkerDefinition(
                marker_name="Lhx1",
                gene_symbol="Lhx1",
                expression_domain="ventricular_zone_cerebellar",
                cell_types_targeted=["basket_cell_precursor", "stellate_cell_precursor"],
                expression_onset="E10.5",
                expression_peak="E12.5",
                expression_maintenance="E12.5-P7",
                downstream_targets=["Gad1", "Gad2", "Pvalb", "Sst"],
                spatial_pattern="ventricular_zone_gradient"
            ),
            GABAergicMarkerDefinition(
                marker_name="Lhx5",
                gene_symbol="Lhx5", 
                expression_domain="ventricular_zone_cerebellar",
                cell_types_targeted=["Golgi_cell_precursor", "basket_cell_precursor"],
                expression_onset="E11.0",
                expression_peak="E13.0",
                expression_maintenance="E13.0-P14",
                downstream_targets=["Gad1", "Gad2", "Neurod6"],
                spatial_pattern="ventricular_zone_patches"
            ),
            GABAergicMarkerDefinition(
                marker_name="Pax2",
                gene_symbol="Pax2",
                expression_domain="isthmus_and_cerebellar_vz",
                cell_types_targeted=["all_GABAergic_interneurons", "isthmic_organizer_cells"],
                expression_onset="E8.5",
                expression_peak="E11.0",
                expression_maintenance="E11.0-P0",
                downstream_targets=["Lhx1", "Lhx5", "Gad1", "Fgf8"],
                spatial_pattern="isthmus_plus_ventricular_zone"
            )
        ]
        
        logger.info(f"Defined {len(markers)} GABAergic specification markers")
        return markers
    
    def download_marker_expression_data(self) -> Dict[str, any]:
        """Download expression data for GABAergic markers."""
        logger.info("Downloading GABAergic marker expression data")
        
        download_results = {
            "download_date": datetime.now().isoformat(),
            "genes_targeted": ["Lhx1", "Lhx5", "Pax2", "Gad1", "Gad2"],
            "files_downloaded": [],
            "download_status": {},
            "total_size_mb": 0
        }
        
        allen_dir = self.expression_dir / "allen_brain_map"
        allen_dir.mkdir(exist_ok=True)
        
        base_url = "http://api.brain-map.org/api/v2"
        
        marker_queries = {
            "lhx1_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Lhx1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))",
            "lhx5_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Lhx5'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))",
            "pax2_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Pax2'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))",
            "gad1_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Gad1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))",
            "gad2_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Gad2'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))"
        }
        
        for query_name, query_url in marker_queries.items():
            try:
                logger.info(f"Downloading {query_name}")
                
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
    
    def execute_tracing(self) -> Dict[str, any]:
        """Execute GABAergic marker tracing."""
        logger.info("Executing Lhx1/5 and Pax2 GABAergic marker tracing")
        
        results = {
            "tracing_date": datetime.now().isoformat(),
            "markers_traced": [],
            "expression_data_downloaded": False,
            "total_data_mb": 0
        }
        
        # Define markers
        markers = self.define_gabaergic_markers()
        results["markers_traced"] = [m.marker_name for m in markers]
        
        # Download expression data
        download_results = self.download_marker_expression_data()
        results["expression_data_downloaded"] = len(download_results["files_downloaded"]) > 0
        results["total_data_mb"] = download_results["total_size_mb"]
        results["download_details"] = download_results
        
        # Save marker definitions
        markers_file = self.metadata_dir / "gabaergic_markers.json"
        markers_data = [
            {
                "marker_name": m.marker_name,
                "gene_symbol": m.gene_symbol,
                "expression_domain": m.expression_domain,
                "cell_types_targeted": m.cell_types_targeted,
                "temporal_expression": {
                    "onset": m.expression_onset,
                    "peak": m.expression_peak,
                    "maintenance": m.expression_maintenance
                },
                "downstream_targets": m.downstream_targets,
                "spatial_pattern": m.spatial_pattern
            } for m in markers
        ]
        
        with open(markers_file, 'w') as f:
            json.dump(markers_data, f, indent=2)
        
        # Save complete results
        results_file = self.metadata_dir / "lhx_pax2_tracing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("GABAergic marker tracing completed")
        return results


def main():
    """Execute Lhx1/5 and Pax2 GABAergic marker tracing."""
    print("🧬 LHX1/5 AND PAX2 GABAERGIC MARKER TRACING")
    print("=" * 60)
    print("Phase 1 ▸ Batch A ▸ Step A2.3")
    
    tracer = LhxPax2GABAergicTracer()
    results = tracer.execute_tracing()
    
    print(f"✅ GABAergic marker tracing completed")
    print(f"🧬 Markers traced: {len(results['markers_traced'])}")
    print(f"📊 Expression data downloaded: {results['expression_data_downloaded']}")
    print(f"💾 Total data: {results['total_data_mb']:.1f}MB")
    
    print("\n🎯 Markers Traced:")
    for marker in results['markers_traced']:
        print(f"  • {marker}")
    
    print(f"\n📁 Data location: {tracer.gabaergic_dir}")


if __name__ == "__main__":
    main()
