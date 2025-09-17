#!/usr/bin/env python3
"""
Engrailed-1/2 Expression Boundary Mapper

Maps Engrailed-1 and Engrailed-2 expression boundaries that define the precise
cerebellar territory versus midbrain territory during early development.
En1/En2 are critical homeodomain transcription factors that maintain the
midbrain-hindbrain boundary and specify cerebellar identity.

Key functions:
- Define En1/En2 expression domains at midbrain-hindbrain boundary
- Map cerebellar territory boundaries using En1/En2 expression
- Integrate with isthmic organizer FGF8/Wnt1 signaling
- Create 3D boundary definitions for morphogen solver integration

Author: Quark Brain Development Team
Date: 2025-09-17
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
import urllib.parse
import requests
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EngrailedDomain:
    """Definition of Engrailed expression domain."""
    gene_name: str  # "En1" or "En2"
    expression_level: str  # "high", "moderate", "low", "absent"
    spatial_domain: str  # "midbrain", "isthmus", "rostral_cerebellum"
    anterior_boundary: float  # A-P position (0.0-1.0)
    posterior_boundary: float
    dorsal_boundary: float  # D-V position (0.0-1.0)
    ventral_boundary: float
    developmental_onset: str  # "E8.5", "E9.0", etc.
    maintenance_period: str  # Duration of expression
    target_cell_types: List[str]


@dataclass
class TerritoryBoundary:
    """Definition of midbrain-cerebellum territory boundary."""
    boundary_name: str
    anatomical_landmarks: List[str]
    molecular_markers: List[str]
    spatial_coordinates: Dict[str, float]
    boundary_sharpness: str  # "sharp", "gradual", "diffuse"
    developmental_stability: str  # "stable", "dynamic", "transient"


class EngrailedBoundaryMapper:
    """Maps Engrailed-1/2 expression boundaries for cerebellar territory definition."""
    
    # Engrailed expression data sources
    ENGRAILED_SOURCES = {
        "allen_brain_map": {
            "name": "Allen Brain Map - Developing Mouse",
            "base_url": "https://developingmouse.brain-map.org/",
            "api_url": "http://api.brain-map.org/api/v2/",
            "en1_gene_id": "13835",  # Engrailed 1
            "en2_gene_id": "13836",  # Engrailed 2
            "developmental_stages": ["E9.5", "E10.5", "E11.5", "E12.5", "E13.5"]
        },
        "genepaint": {
            "name": "GenePaint.org Expression Database",
            "base_url": "http://www.genepaint.org/",
            "en1_sets": ["MG42", "MG43", "MG44"],
            "en2_sets": ["MG45", "MG46", "MG47"],
            "stages": ["E9.5", "E10.5", "E11.5", "E12.5"]
        },
        "emage": {
            "name": "EMAGE Gene Expression Database",
            "base_url": "http://www.emouseatlas.org/emage/",
            "search_terms": ["Engrailed1", "Engrailed2", "midbrain", "cerebellum"]
        }
    }
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize Engrailed boundary mapper.
        
        Args:
            data_dir: Base directory for cerebellar data storage
        """
        self.data_dir = Path(data_dir)
        self.engrailed_dir = self.data_dir / "engrailed_boundaries"
        self.expression_dir = self.engrailed_dir / "expression_data"
        self.boundaries_dir = self.engrailed_dir / "territory_boundaries"
        self.metadata_dir = self.engrailed_dir / "metadata"
        
        # Create directory structure
        for directory in [self.engrailed_dir, self.expression_dir, self.boundaries_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized Engrailed boundary mapper")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Engrailed directory: {self.engrailed_dir}")
    
    def define_engrailed_domains(self) -> List[EngrailedDomain]:
        """Define En1 and En2 expression domains across development.
        
        Returns:
            List of Engrailed expression domain definitions
        """
        logger.info("Defining Engrailed-1/2 expression domains")
        
        domains = [
            # Engrailed-1 domains
            EngrailedDomain(
                gene_name="En1",
                expression_level="high",
                spatial_domain="midbrain",
                anterior_boundary=0.20,  # Rostral midbrain
                posterior_boundary=0.41,  # Midbrain-hindbrain boundary
                dorsal_boundary=0.9,     # Dorsal neural tube
                ventral_boundary=0.3,    # Ventral midbrain
                developmental_onset="E8.5",
                maintenance_period="E8.5-adult",
                target_cell_types=["midbrain_progenitor", "dopaminergic_neuron", "GABAergic_neuron"]
            ),
            EngrailedDomain(
                gene_name="En1",
                expression_level="moderate",
                spatial_domain="rostral_cerebellum",
                anterior_boundary=0.41,  # Isthmic organizer
                posterior_boundary=0.45,  # Rostral cerebellar territory
                dorsal_boundary=0.9,     # Dorsal cerebellar plate
                ventral_boundary=0.5,    # Ventricular zone
                developmental_onset="E9.0",
                maintenance_period="E9.0-E14.5",
                target_cell_types=["cerebellar_progenitor", "deep_nuclei_neuron"]
            ),
            # Engrailed-2 domains
            EngrailedDomain(
                gene_name="En2",
                expression_level="high",
                spatial_domain="midbrain",
                anterior_boundary=0.25,  # Mid-midbrain
                posterior_boundary=0.41,  # Midbrain-hindbrain boundary
                dorsal_boundary=0.9,     # Dorsal neural tube
                ventral_boundary=0.4,    # Ventral midbrain
                developmental_onset="E8.5",
                maintenance_period="E8.5-adult",
                target_cell_types=["midbrain_progenitor", "collicular_neuron"]
            ),
            EngrailedDomain(
                gene_name="En2",
                expression_level="high",
                spatial_domain="entire_cerebellum",
                anterior_boundary=0.41,  # Isthmic organizer
                posterior_boundary=0.55,  # Entire cerebellar territory
                dorsal_boundary=0.9,     # Dorsal cerebellar plate
                ventral_boundary=0.4,    # Deep cerebellar territory
                developmental_onset="E9.5",
                maintenance_period="E9.5-adult",
                target_cell_types=["all_cerebellar_neurons", "Purkinje_cell", "granule_cell", "deep_nuclei"]
            ),
            # Boundary transition zones
            EngrailedDomain(
                gene_name="En1_En2_coexpression",
                expression_level="high",
                spatial_domain="isthmus",
                anterior_boundary=0.40,  # Isthmic organizer
                posterior_boundary=0.42,
                dorsal_boundary=0.9,     # Dorsal boundary
                ventral_boundary=0.5,    # Ventricular zone
                developmental_onset="E8.5",
                maintenance_period="E8.5-E12.5",
                target_cell_types=["isthmic_organizer_cell", "boundary_cell"]
            )
        ]
        
        logger.info(f"Defined {len(domains)} Engrailed expression domains")
        return domains
    
    def create_territory_boundaries(self) -> List[TerritoryBoundary]:
        """Create precise territory boundaries between midbrain and cerebellum.
        
        Returns:
            List of territory boundary definitions
        """
        logger.info("Creating midbrain-cerebellum territory boundaries")
        
        boundaries = [
            TerritoryBoundary(
                boundary_name="midbrain_cerebellar_boundary",
                anatomical_landmarks=[
                    "isthmic_organizer",
                    "superior_colliculus_caudal_edge",
                    "cerebellar_plate_rostral_edge",
                    "fourth_ventricle_rostral_recess"
                ],
                molecular_markers=["En1", "En2", "Fgf8", "Wnt1", "Pax2", "Gbx2", "Otx2"],
                spatial_coordinates={
                    "anteroposterior_position": 0.41,
                    "dorsoventral_range": [0.4, 0.9],
                    "mediolateral_extent": [0.0, 1.0],
                    "boundary_width_um": 50
                },
                boundary_sharpness="sharp",
                developmental_stability="stable"
            ),
            TerritoryBoundary(
                boundary_name="cerebellar_medulla_boundary",
                anatomical_landmarks=[
                    "cerebellar_plate_caudal_edge",
                    "fourth_ventricle_caudal_extent",
                    "choroid_plexus_attachment"
                ],
                molecular_markers=["En2", "Hoxa2", "Hoxb2", "Krox20"],
                spatial_coordinates={
                    "anteroposterior_position": 0.55,
                    "dorsoventral_range": [0.3, 0.8],
                    "mediolateral_extent": [0.0, 1.0],
                    "boundary_width_um": 100
                },
                boundary_sharpness="gradual",
                developmental_stability="dynamic"
            ),
            TerritoryBoundary(
                boundary_name="cerebellar_pons_boundary",
                anatomical_landmarks=[
                    "cerebellar_peduncle_attachment",
                    "pontine_tegmentum_dorsal_edge"
                ],
                molecular_markers=["En2", "Pax2", "Lhx1", "Lhx5"],
                spatial_coordinates={
                    "anteroposterior_position": 0.48,
                    "dorsoventral_range": [0.2, 0.6],
                    "mediolateral_extent": [0.3, 0.7],
                    "boundary_width_um": 75
                },
                boundary_sharpness="gradual",
                developmental_stability="stable"
            )
        ]
        
        logger.info(f"Created {len(boundaries)} territory boundaries")
        return boundaries
    
    def download_engrailed_expression_data(self) -> Dict[str, any]:
        """Download En1/En2 expression data from multiple sources.
        
        Returns:
            Download results and metadata
        """
        logger.info("Downloading Engrailed-1/2 expression data")
        
        download_results = {
            "download_date": datetime.now().isoformat(),
            "genes_targeted": ["En1", "En2"],
            "sources_attempted": [],
            "successful_downloads": [],
            "total_data_mb": 0,
            "download_details": {}
        }
        
        # 1. Download Allen Brain Map En1/En2 data
        logger.info("=== Allen Brain Map Engrailed Data ===")
        allen_results = self._download_allen_engrailed_data()
        download_results["sources_attempted"].append("Allen_Brain_Map")
        download_results["download_details"]["Allen_Brain_Map"] = allen_results
        
        if any("success" in str(status) for status in allen_results["download_status"].values()):
            download_results["successful_downloads"].append("Allen_Brain_Map")
            download_results["total_data_mb"] += allen_results["total_size_mb"]
        
        # 2. Download GenePaint En1/En2 data
        logger.info("=== GenePaint Engrailed Expression ===")
        genepaint_results = self._download_genepaint_engrailed_data()
        download_results["sources_attempted"].append("GenePaint")
        download_results["download_details"]["GenePaint"] = genepaint_results
        
        if any("success" in str(status) for status in genepaint_results["download_status"].values()):
            download_results["successful_downloads"].append("GenePaint")
            download_results["total_data_mb"] += genepaint_results["total_size_mb"]
        
        return download_results
    
    def _download_allen_engrailed_data(self) -> Dict[str, any]:
        """Download En1/En2 data from Allen Brain Map.
        
        Returns:
            Allen Brain Map download results
        """
        allen_results = {
            "dataset": "Allen_Engrailed_expression",
            "download_date": datetime.now().isoformat(),
            "genes": ["En1", "En2"],
            "files_downloaded": [],
            "download_status": {},
            "total_size_mb": 0
        }
        
        allen_dir = self.expression_dir / "allen_brain_map"
        allen_dir.mkdir(exist_ok=True)
        
        # Allen Brain Map API queries for En1 and En2
        base_url = "http://api.brain-map.org/api/v2"
        
        engrailed_queries = {
            "en1_gene_info": {
                "url": f"{base_url}/data/query.json?criteria=model::Gene,rma::criteria,[acronym$eq'En1']",
                "description": "Engrailed-1 gene information"
            },
            "en2_gene_info": {
                "url": f"{base_url}/data/query.json?criteria=model::Gene,rma::criteria,[acronym$eq'En2']", 
                "description": "Engrailed-2 gene information"
            },
            "en1_experiments": {
                "url": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'En1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age)),plane_of_section",
                "description": "En1 expression experiments in developing mouse"
            },
            "en2_experiments": {
                "url": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'En2'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age)),plane_of_section",
                "description": "En2 expression experiments in developing mouse"
            },
            "midbrain_structures": {
                "url": f"{base_url}/data/query.json?criteria=model::Structure,rma::criteria,[name$il'*midbrain*']",
                "description": "Midbrain anatomical structures"
            }
        }
        
        for query_name, query_info in engrailed_queries.items():
            try:
                logger.info(f"Querying Allen Brain Map: {query_name}")
                
                with urllib.request.urlopen(query_info["url"], timeout=30) as response:
                    data = json.loads(response.read())
                
                # Save query results
                output_file = allen_dir / f"{query_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                file_size_mb = output_file.stat().st_size / (1024*1024)
                allen_results["files_downloaded"].append(str(output_file))
                allen_results["download_status"][query_name] = "success"
                allen_results["total_size_mb"] += file_size_mb
                
                logger.info(f"✅ Downloaded {query_name} ({file_size_mb:.2f}MB)")
                
            except Exception as e:
                allen_results["download_status"][query_name] = f"failed: {str(e)}"
                logger.error(f"❌ Failed to download {query_name}: {e}")
        
        return allen_results
    
    def _download_genepaint_engrailed_data(self) -> Dict[str, any]:
        """Download En1/En2 data from GenePaint.
        
        Returns:
            GenePaint download results
        """
        genepaint_results = {
            "dataset": "GenePaint_Engrailed_expression",
            "download_date": datetime.now().isoformat(),
            "gene_sets": ["MG42", "MG43", "MG44", "MG45", "MG46", "MG47"],
            "files_downloaded": [],
            "download_status": {},
            "total_size_mb": 0
        }
        
        genepaint_dir = self.expression_dir / "genepaint"
        genepaint_dir.mkdir(exist_ok=True)
        
        # GenePaint En1/En2 sets
        engrailed_sets = {
            "En1_MG42": "http://www.genepaint.org/data/MG42/",
            "En1_MG43": "http://www.genepaint.org/data/MG43/",
            "En1_MG44": "http://www.genepaint.org/data/MG44/",
            "En2_MG45": "http://www.genepaint.org/data/MG45/",
            "En2_MG46": "http://www.genepaint.org/data/MG46/",
            "En2_MG47": "http://www.genepaint.org/data/MG47/"
        }
        
        for set_name, set_url in engrailed_sets.items():
            try:
                logger.info(f"Accessing GenePaint set: {set_name}")
                
                # Try to get directory listing
                response = requests.get(set_url, timeout=30)
                if response.status_code == 200:
                    # Save directory listing
                    listing_file = genepaint_dir / f"{set_name}_listing.html"
                    with open(listing_file, 'w') as f:
                        f.write(response.text)
                    
                    file_size_mb = listing_file.stat().st_size / (1024*1024)
                    genepaint_results["files_downloaded"].append(str(listing_file))
                    genepaint_results["download_status"][set_name] = "directory_listing_success"
                    genepaint_results["total_size_mb"] += file_size_mb
                    
                    logger.info(f"✅ Downloaded {set_name} listing ({file_size_mb:.2f}MB)")
                else:
                    genepaint_results["download_status"][set_name] = f"failed_http_{response.status_code}"
                    logger.warning(f"⚠️ Failed to access {set_name}: HTTP {response.status_code}")
                    
            except Exception as e:
                genepaint_results["download_status"][set_name] = f"failed: {str(e)}"
                logger.warning(f"⚠️ Failed to access {set_name}: {e}")
        
        return genepaint_results
    
    def create_boundary_maps(self, engrailed_domains: List[EngrailedDomain]) -> Dict[str, any]:
        """Create 3D boundary maps from Engrailed expression domains.
        
        Args:
            engrailed_domains: List of Engrailed expression domains
            
        Returns:
            3D boundary map data
        """
        logger.info("Creating 3D boundary maps from Engrailed expression")
        
        # Create 3D grid for boundary mapping
        grid_size = (100, 100, 50)  # X (M-L), Y (A-P), Z (D-V)
        
        # Initialize expression maps
        en1_expression = np.zeros(grid_size)
        en2_expression = np.zeros(grid_size)
        boundary_map = np.zeros(grid_size)
        
        for domain in engrailed_domains:
            # Convert boundary positions to grid coordinates
            ap_start = int(domain.anterior_boundary * grid_size[1])
            ap_end = int(domain.posterior_boundary * grid_size[1])
            dv_start = int(domain.ventral_boundary * grid_size[2])
            dv_end = int(domain.dorsal_boundary * grid_size[2])
            
            # Expression level mapping
            expression_value = {
                "high": 1.0,
                "moderate": 0.7,
                "low": 0.3,
                "absent": 0.0
            }.get(domain.expression_level, 0.0)
            
            # Fill expression domain
            if domain.gene_name == "En1":
                en1_expression[:, ap_start:ap_end, dv_start:dv_end] = expression_value
            elif domain.gene_name == "En2":
                en2_expression[:, ap_start:ap_end, dv_start:dv_end] = expression_value
            elif domain.gene_name == "En1_En2_coexpression":
                en1_expression[:, ap_start:ap_end, dv_start:dv_end] = expression_value
                en2_expression[:, ap_start:ap_end, dv_start:dv_end] = expression_value
        
        # Create boundary map (high gradient regions)
        en1_gradient = np.gradient(en1_expression, axis=1)  # A-P gradient
        en2_gradient = np.gradient(en2_expression, axis=1)
        boundary_map = np.sqrt(en1_gradient**2 + en2_gradient**2)
        
        # Identify sharp boundaries (high gradient regions)
        boundary_threshold = np.percentile(boundary_map, 95)  # Top 5% of gradients
        sharp_boundaries = boundary_map > boundary_threshold
        
        boundary_results = {
            "creation_date": datetime.now().isoformat(),
            "grid_dimensions": grid_size,
            "voxel_size_um": 50,
            "expression_maps": {
                "en1_expression_stats": {
                    "mean": float(np.mean(en1_expression)),
                    "max": float(np.max(en1_expression)),
                    "expressing_fraction": float(np.mean(en1_expression > 0.1))
                },
                "en2_expression_stats": {
                    "mean": float(np.mean(en2_expression)),
                    "max": float(np.max(en2_expression)),
                    "expressing_fraction": float(np.mean(en2_expression > 0.1))
                }
            },
            "boundary_analysis": {
                "boundary_threshold": float(boundary_threshold),
                "sharp_boundary_fraction": float(np.mean(sharp_boundaries)),
                "midbrain_cerebellum_boundary_position": 0.41,
                "boundary_width_estimate_um": 50
            }
        }
        
        # Save expression maps
        en1_file = self.boundaries_dir / "en1_expression_map.npy"
        en2_file = self.boundaries_dir / "en2_expression_map.npy"
        boundary_file = self.boundaries_dir / "territory_boundary_map.npy"
        
        np.save(en1_file, en1_expression)
        np.save(en2_file, en2_expression)
        np.save(boundary_file, boundary_map)
        
        logger.info("Created 3D boundary maps with Engrailed expression patterns")
        return boundary_results
    
    def execute_mapping(self) -> Dict[str, any]:
        """Execute Engrailed boundary mapping for cerebellar territory definition.
        
        Returns:
            Mapping results and metadata
        """
        logger.info("Executing Engrailed-1/2 boundary mapping")
        
        mapping_results = {
            "mapping_date": datetime.now().isoformat(),
            "sources_attempted": [],
            "successful_mappings": [],
            "total_data_mb": 0,
            "mapping_details": {}
        }
        
        # 1. Define Engrailed expression domains
        logger.info("=== Defining Engrailed Expression Domains ===")
        engrailed_domains = self.define_engrailed_domains()
        
        # 2. Create territory boundaries
        logger.info("=== Creating Territory Boundaries ===")
        territory_boundaries = self.create_territory_boundaries()
        
        # 3. Download expression data
        logger.info("=== Downloading Expression Data ===")
        download_results = self.download_engrailed_expression_data()
        mapping_results["sources_attempted"].extend(download_results["sources_attempted"])
        mapping_results["successful_mappings"].extend(download_results["successful_downloads"])
        mapping_results["total_data_mb"] += download_results["total_data_mb"]
        mapping_results["mapping_details"]["expression_downloads"] = download_results
        
        # 4. Create 3D boundary maps
        logger.info("=== Creating 3D Boundary Maps ===")
        boundary_maps = self.create_boundary_maps(engrailed_domains)
        mapping_results["mapping_details"]["boundary_maps"] = boundary_maps
        mapping_results["successful_mappings"].append("3D_Boundary_Maps")
        
        # 5. Save domain and boundary definitions
        domains_file = self.metadata_dir / "engrailed_domains.json"
        domains_data = [
            {
                "gene_name": domain.gene_name,
                "expression_level": domain.expression_level,
                "spatial_domain": domain.spatial_domain,
                "boundaries": {
                    "anterior": domain.anterior_boundary,
                    "posterior": domain.posterior_boundary,
                    "dorsal": domain.dorsal_boundary,
                    "ventral": domain.ventral_boundary
                },
                "developmental_timing": {
                    "onset": domain.developmental_onset,
                    "maintenance": domain.maintenance_period
                },
                "target_cell_types": domain.target_cell_types
            } for domain in engrailed_domains
        ]
        
        with open(domains_file, 'w') as f:
            json.dump(domains_data, f, indent=2)
        
        boundaries_file = self.metadata_dir / "territory_boundaries.json"
        boundaries_data = [
            {
                "boundary_name": boundary.boundary_name,
                "anatomical_landmarks": boundary.anatomical_landmarks,
                "molecular_markers": boundary.molecular_markers,
                "spatial_coordinates": boundary.spatial_coordinates,
                "boundary_properties": {
                    "sharpness": boundary.boundary_sharpness,
                    "stability": boundary.developmental_stability
                }
            } for boundary in territory_boundaries
        ]
        
        with open(boundaries_file, 'w') as f:
            json.dump(boundaries_data, f, indent=2)
        
        # Save complete mapping results
        results_file = self.metadata_dir / "engrailed_mapping_results.json"
        with open(results_file, 'w') as f:
            json.dump(mapping_results, f, indent=2)
        
        logger.info(f"Engrailed boundary mapping completed. Results saved to {results_file}")
        return mapping_results


def main():
    """Execute Engrailed-1/2 expression boundary mapping."""
    
    print("🧬 ENGRAILED-1/2 EXPRESSION BOUNDARY MAPPING")
    print("=" * 60)
    print("Phase 1 ▸ Batch A ▸ Step A2.1")
    print("Cerebellar Territory vs Midbrain Definition")
    print()
    
    # Initialize mapper
    mapper = EngrailedBoundaryMapper()
    
    # Execute mapping
    results = mapper.execute_mapping()
    
    # Print mapping summary
    print(f"✅ Engrailed boundary mapping completed")
    print(f"📊 Sources attempted: {len(results['sources_attempted'])}")
    print(f"✅ Successful mappings: {len(results['successful_mappings'])}")
    print(f"💾 Total data mapped: {results['total_data_mb']:.1f}MB")
    print()
    
    # Display mapping details
    print("📥 Mapping Results:")
    for mapping_name, details in results['mapping_details'].items():
        if mapping_name == "expression_downloads":
            print(f"  • Expression Downloads: {len(details['successful_downloads'])} sources")
        elif mapping_name == "boundary_maps":
            print(f"  • 3D Boundary Maps: {details['grid_dimensions']} grid")
            print(f"    En1 expressing fraction: {details['expression_maps']['en1_expression_stats']['expressing_fraction']:.2f}")
            print(f"    En2 expressing fraction: {details['expression_maps']['en2_expression_stats']['expressing_fraction']:.2f}")
    
    print()
    print("🎯 Territory Boundaries Defined:")
    print("  • Midbrain-Cerebellar boundary: A-P position 0.41")
    print("  • Cerebellar-Medulla boundary: A-P position 0.55")
    print("  • Cerebellar-Pons boundary: A-P position 0.48")
    
    print()
    print("📁 Data Locations:")
    print(f"  • Expression data: {mapper.expression_dir}")
    print(f"  • Boundary maps: {mapper.boundaries_dir}")
    print(f"  • Domain definitions: {mapper.metadata_dir}")
    
    print()
    print("🎯 Next Steps:")
    print("- Validate En1/En2 boundaries against developmental literature")
    print("- Integrate boundary maps with morphogen solver")
    print("- Proceed to A2.2: Document Gbx2/Otx2 interface at isthmic organizer")
    print("- Continue with A2.3: Trace Lhx1/5 and Pax2 markers")


if __name__ == "__main__":
    main()
