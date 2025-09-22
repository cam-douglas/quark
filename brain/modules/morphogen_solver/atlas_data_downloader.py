#!/usr/bin/env python3
"""Allen Atlas Data Downloader.

Downloads and manages Allen Brain Atlas embryonic reference data including
automated download, data validation, and local storage management for
morphogen solver validation.

Integration: Data management component for atlas validation system
Rationale: Focused data download and management separated from validation logic
"""

from typing import Dict, Any, Optional, List
import requests
import numpy as np
from pathlib import Path
import json
import logging
import hashlib
from urllib.parse import urljoin

from .atlas_validation_types import AtlasReference, CoordinateSystem

logger = logging.getLogger(__name__)

class AtlasDataDownloader:
    """Downloader and manager for Allen Brain Atlas data.
    
    Handles automated download of embryonic reference data from Allen
    Brain Atlas, validates data integrity, and manages local storage
    for morphogen solver validation tasks.
    """
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/allen_brain"):
        """Initialize atlas data downloader.
        
        Args:
            data_dir: Directory for storing atlas data (updated location)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Allen Brain Atlas API configuration
        self.allen_api_base = "https://api.brain-map.org/api/v2/"
        self.human_atlas_base = "https://human.brain-map.org/"
        self.brainspan_base = "https://www.brainspan.org/"
        
        self.atlas_endpoints = {
            "microarray": "api/v2/well_known_file_download/",
            "rna_seq": "api/v2/well_known_file_download/",
            "embryonic_query": "api/v2/data/query.json?criteria=model::AtlasImage,rma::criteria,[section_data_set$eq'Mouse Embryonic Development']",
            "brainspan_dev_transcriptome": "rnaseq/search/index.html",
            "brainspan_prenatal_lmd": "lcm/search/index.html",
            "brainspan_ish": "ish",
            "brainspan_reference_atlas": "static/atlas"
        }
        
        # Available dataset IDs from Allen Brain Atlas
        self.dataset_ids = {
            "H0351.2001": "178238387",  # Complete normalized microarray
            "H0351.2002": "178238373",  # Complete normalized microarray
            "H0351.1009": "178238359",  # Complete normalized microarray
            "H0351.1012": "178238316",  # Complete normalized microarray
            "H0351.1015": "178238266",  # Complete normalized microarray
            "H0351.1016": "178236545",  # Complete normalized microarray
            "RNA_H0351.2001": "278447594",  # RNA-Sequencing dataset
            "RNA_H0351.2002": "278448166"   # RNA-Sequencing dataset
        }
        
        # BrainSpan developmental data endpoints (from crawl4ai analysis)
        self.brainspan_datasets = {
            "rna_seq_exons": "https://www.brainspan.org/api/v2/well_known_file_download/267666525",  # RNA-Seq Gencode v10 exons
            "rna_seq_genes": "https://www.brainspan.org/api/v2/well_known_file_download/267666527",  # RNA-Seq Gencode v10 genes  
            "exon_microarray_probes": "https://www.brainspan.org/api/v2/well_known_file_download/267666529",  # Exon microarray probes
            "exon_microarray_genes": "https://www.brainspan.org/api/v2/well_known_file_download/267666531",  # Exon microarray genes
            "prenatal_15pcw_male": "https://www.brainspan.org/api/v2/well_known_file_download/267666533",  # H376.IIIA.02, 15 pcw male
            "prenatal_16pcw_female": "https://www.brainspan.org/api/v2/well_known_file_download/267666535",  # H376.IIIB.02, 16 pcw female
            "prenatal_21pcw_female1": "https://www.brainspan.org/api/v2/well_known_file_download/267666537",  # H376.IV.02, 21 pcw female
            "prenatal_21pcw_female2": "https://www.brainspan.org/api/v2/well_known_file_download/267666539"   # H376.IV.03, 21 pcw female
        }
        
        # Available embryonic stages
        self.available_stages = ["E8.5", "E9.5", "E10.5", "E11.5", "E12.5"]
        
        logger.info("Initialized AtlasDataDownloader")
        logger.info(f"Data directory: {self.data_dir}")
    
    def download_embryonic_reference(self, developmental_stage: str = "E10.5") -> Optional[AtlasReference]:
        """Download embryonic reference data for specified stage.
        
        Args:
            developmental_stage: Embryonic stage (E8.5, E9.5, etc.)
            
        Returns:
            AtlasReference object with downloaded data
        """
        if developmental_stage not in self.available_stages:
            logger.error(f"Stage {developmental_stage} not available. Available: {self.available_stages}")
            return None
        
        logger.info(f"Downloading Allen Atlas data for {developmental_stage}")
        
        # Check if already downloaded
        atlas_file = self.data_dir / f"allen_atlas_{developmental_stage}.npz"
        metadata_file = self.data_dir / f"allen_atlas_{developmental_stage}_metadata.json"
        
        if atlas_file.exists() and metadata_file.exists():
            logger.info("Atlas data already exists, loading from cache")
            return self._load_cached_atlas(atlas_file, metadata_file)
        
        # Download from Allen API
        atlas_data = self._download_from_allen_api(developmental_stage)
        
        if atlas_data is None:
            logger.error(f"Failed to download atlas data for {developmental_stage}")
            return None
        
        # Save to cache
        self._save_atlas_to_cache(atlas_data, atlas_file, metadata_file)
        
        return atlas_data
    
    def _download_from_allen_api(self, developmental_stage: str) -> Optional[AtlasReference]:
        """Download data from Allen Brain Atlas API using actual endpoints."""
        try:
            # Download microarray data for embryonic stages
            # Use H0351.2001 dataset as primary reference
            dataset_id = self.dataset_ids["H0351.2001"]
            download_url = f"{self.allen_api_base}{self.atlas_endpoints['microarray']}{dataset_id}"
            
            logger.info(f"Downloading Allen Atlas data: {download_url}")
            
            # Download the dataset
            response = requests.get(download_url, timeout=120, stream=True)
            response.raise_for_status()
            
            # Save raw data to temporary file
            temp_file = self.data_dir / f"allen_raw_{developmental_stage}.zip"
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded Allen Atlas data: {temp_file.stat().st_size / 1024**2:.1f} MB")
            
            # Process downloaded data (simplified - would need full Allen data parser)
            atlas_ref = self._process_downloaded_data(temp_file, developmental_stage)
            
            # Clean up temporary file
            temp_file.unlink()
            
            return atlas_ref
            
        except requests.RequestException as e:
            logger.error(f"Allen API download failed: {e}. Cannot proceed without real data.")
            return None
        except Exception as e:
            logger.error(f"Download processing failed: {e}. Cannot proceed without real data.")
            return None
    
    def _process_downloaded_data(self, data_file: Path, developmental_stage: str) -> AtlasReference:
        """Process downloaded Allen Atlas data file."""
        logger.info("Processing downloaded Allen Atlas data")
        
        # For now, create synthetic reference as Allen data format is complex
        # In production, would implement full Allen data parser
        logger.error("Data processing not implemented. Cannot create a valid AtlasReference from downloaded file.")
        
        return None
    
    def download_rna_seq_data(self, dataset_name: str = "H0351.2001") -> Optional[Dict[str, Any]]:
        """Download RNA-Sequencing data from Allen Atlas.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            RNA-seq data dictionary
        """
        if dataset_name not in ["H0351.2001", "H0351.2002"]:
            logger.error(f"Invalid RNA-seq dataset: {dataset_name}")
            return None
        
        try:
            rna_dataset_id = self.dataset_ids[f"RNA_{dataset_name}"]
            download_url = f"{self.allen_api_base}{self.atlas_endpoints['rna_seq']}{rna_dataset_id}"
            
            logger.info(f"Downloading RNA-seq data: {download_url}")
            
            response = requests.get(download_url, timeout=120, stream=True)
            response.raise_for_status()
            
            # Save RNA-seq data
            rna_file = self.data_dir / f"allen_rnaseq_{dataset_name}.zip"
            
            with open(rna_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded RNA-seq data: {rna_file.stat().st_size / 1024**2:.1f} MB")
            
            return {"file_path": str(rna_file), "dataset_name": dataset_name}
            
        except Exception as e:
            logger.error(f"RNA-seq download failed: {e}")
            return None
    
    def download_all_brainspan_data(self, developmental_stage: str) -> Optional[Dict[str, Any]]:
        """Download ALL available BrainSpan datasets comprehensively.
        
        Args:
            developmental_stage: Embryonic developmental stage
            
        Returns:
            Dictionary with ALL BrainSpan data information
        """
        logger.info(f"Downloading ALL BrainSpan developmental data for {developmental_stage}")
        
        downloaded_files = {}
        failed_downloads = []
        total_size_mb = 0.0
        
        # Download ALL BrainSpan datasets
        for dataset_name, dataset_url in self.brainspan_datasets.items():
            try:
                logger.info(f"Downloading {dataset_name}: {dataset_url}")
                
                response = requests.get(dataset_url, timeout=300, stream=True)
                response.raise_for_status()
                
                # Save file with descriptive name
                file_path = self.data_dir / f"brainspan_{dataset_name}_{developmental_stage}.zip"
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size_mb = file_path.stat().st_size / 1024**2
                total_size_mb += file_size_mb
                
                downloaded_files[dataset_name] = {
                    "file_path": str(file_path),
                    "size_mb": file_size_mb,
                    "download_url": dataset_url
                }
                
                logger.info(f"✅ Downloaded {dataset_name}: {file_size_mb:.1f} MB")
                
            except Exception as e:
                logger.warning(f"❌ Failed to download {dataset_name}: {e}")
                failed_downloads.append({"dataset": dataset_name, "error": str(e), "url": dataset_url})
                continue
        
        # Also download using crawl4ai to get the web interface data
        web_data = self._download_brainspan_web_interfaces(developmental_stage)
        if web_data:
            downloaded_files.update(web_data)
        
        success_count = len(downloaded_files) - len([f for f in downloaded_files.values() if isinstance(f, dict) and "web_interface" in f.get("type", "")])
        total_datasets = len(self.brainspan_datasets)
        
        logger.info(f"BrainSpan download summary: {success_count}/{total_datasets} datasets, {total_size_mb:.1f} MB total")
        
        if failed_downloads:
            logger.warning(f"Failed downloads: {len(failed_downloads)}")
            for failure in failed_downloads:
                logger.warning(f"  {failure['dataset']}: {failure['error']}")
        
        return {
            "downloaded_files": downloaded_files,
            "failed_downloads": failed_downloads,
            "developmental_stage": developmental_stage,
            "total_size_mb": total_size_mb,
            "success_rate": success_count / total_datasets,
            "download_successful": success_count > 0,
            "comprehensive_download": True
        }
    
    def _download_brainspan_web_interfaces(self, developmental_stage: str) -> Dict[str, Any]:
        """Download BrainSpan web interface data using crawl4ai."""
        web_data = {}
        
        try:
            # Import crawl4ai functions
            from mcp_crawl4ai import read_url
            
            # Download main BrainSpan page
            brainspan_main = read_url("https://www.brainspan.org/", format="markdown_with_citations")
            if brainspan_main:
                web_file = self.data_dir / f"brainspan_main_{developmental_stage}.md"
                with open(web_file, 'w') as f:
                    f.write(brainspan_main["result"])
                web_data["main_page"] = {"file_path": str(web_file), "type": "web_interface"}
            
            # Download developmental transcriptome interface
            dev_transcriptome = read_url("https://www.brainspan.org/rnaseq/search/index.html", format="markdown_with_citations")
            if dev_transcriptome:
                dev_file = self.data_dir / f"brainspan_dev_transcriptome_{developmental_stage}.md"
                with open(dev_file, 'w') as f:
                    f.write(dev_transcriptome["result"])
                web_data["dev_transcriptome"] = {"file_path": str(dev_file), "type": "web_interface"}
            
            # Download reference atlas interface
            ref_atlas = read_url("https://www.brainspan.org/static/atlas", format="markdown_with_citations")
            if ref_atlas:
                atlas_file = self.data_dir / f"brainspan_ref_atlas_{developmental_stage}.md"
                with open(atlas_file, 'w') as f:
                    f.write(ref_atlas["result"])
                web_data["reference_atlas"] = {"file_path": str(atlas_file), "type": "web_interface"}
            
            logger.info(f"Downloaded {len(web_data)} BrainSpan web interfaces")
            
        except Exception as e:
            logger.warning(f"Web interface download failed: {e}")
        
        return web_data
    
    def download_all_allen_atlas_data(self, developmental_stage: str) -> Optional[Dict[str, Any]]:
        """Download ALL available Allen Brain Atlas datasets.
        
        Args:
            developmental_stage: Embryonic developmental stage
            
        Returns:
            Dictionary with ALL Allen Atlas data
        """
        logger.info(f"Downloading ALL Allen Brain Atlas data for {developmental_stage}")
        
        downloaded_files = {}
        failed_downloads = []
        total_size_mb = 0.0
        
        # Download ALL Allen Atlas datasets
        for dataset_name, dataset_id in self.dataset_ids.items():
            try:
                if dataset_name.startswith("RNA_"):
                    download_url = f"{self.allen_api_base}{self.atlas_endpoints['rna_seq']}{dataset_id}"
                else:
                    download_url = f"{self.allen_api_base}{self.atlas_endpoints['microarray']}{dataset_id}"
                
                logger.info(f"Downloading Allen {dataset_name}: {download_url}")
                
                response = requests.get(download_url, timeout=300, stream=True)
                response.raise_for_status()
                
                file_path = self.data_dir / f"allen_{dataset_name}_{developmental_stage}.zip"
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size_mb = file_path.stat().st_size / 1024**2
                total_size_mb += file_size_mb
                
                downloaded_files[dataset_name] = {
                    "file_path": str(file_path),
                    "size_mb": file_size_mb,
                    "dataset_id": dataset_id,
                    "download_url": download_url
                }
                
                logger.info(f"✅ Downloaded Allen {dataset_name}: {file_size_mb:.1f} MB")
                
            except Exception as e:
                logger.warning(f"❌ Failed to download Allen {dataset_name}: {e}")
                failed_downloads.append({"dataset": dataset_name, "error": str(e), "url": download_url})
                continue
        
        success_count = len(downloaded_files)
        total_datasets = len(self.dataset_ids)
        
        logger.info(f"Allen Atlas download summary: {success_count}/{total_datasets} datasets, {total_size_mb:.1f} MB total")
        
        return {
            "downloaded_files": downloaded_files,
            "failed_downloads": failed_downloads,
            "developmental_stage": developmental_stage,
            "total_size_mb": total_size_mb,
            "success_rate": success_count / total_datasets,
            "download_successful": success_count > 0,
            "comprehensive_download": True
        }
    
    def download_brainspan_developmental_data(self, developmental_stage: str) -> Optional[Dict[str, Any]]:
        """Download BrainSpan developmental transcriptome data (wrapper for comprehensive download)."""
        return self.download_all_brainspan_data(developmental_stage)
    
    def integrate_brainspan_with_allen(self, developmental_stage: str) -> Optional[AtlasReference]:
        """Integrate BrainSpan and Allen Atlas data for comprehensive reference.
        
        Args:
            developmental_stage: Target developmental stage
            
        Returns:
            Integrated atlas reference
        """
        logger.info("Integrating BrainSpan and Allen Atlas data")
        
        # Download ALL available datasets comprehensively
        allen_data = self.download_all_allen_atlas_data(developmental_stage)
        brainspan_data = self.download_all_brainspan_data(developmental_stage)
        
        # Also get the basic atlas reference
        allen_ref = self.download_embryonic_reference(developmental_stage)
        
        if allen_ref is None or brainspan_data is None:
            logger.warning("Failed to download complete datasets, using available data")
            return allen_ref  # Return Allen data if available
        
        # Create enhanced atlas reference with BrainSpan integration
        enhanced_ref = AtlasReference(
            atlas_id=f"integrated_{developmental_stage}",
            developmental_stage=developmental_stage,
            coordinate_system=allen_ref.coordinate_system,
            resolution_um=allen_ref.resolution_um,
            dimensions=allen_ref.dimensions,
            region_labels=allen_ref.region_labels,
            region_names=allen_ref.region_names,
            reference_url=f"integrated://allen+brainspan_{developmental_stage}"
        )
        
        # Save comprehensive integration metadata
        integration_metadata = {
            "allen_atlas_id": allen_ref.atlas_id,
            "allen_comprehensive_data": allen_data,
            "brainspan_comprehensive_data": brainspan_data,
            "integration_date": "2025-01-04",
            "developmental_stage": developmental_stage,
            "data_sources": ["Allen Brain Atlas", "BrainSpan Atlas"],
            "total_datasets_downloaded": (
                len(allen_data.get("downloaded_files", {})) + 
                len(brainspan_data.get("downloaded_files", {}))
            ),
            "total_size_mb": (
                allen_data.get("total_size_mb", 0.0) + 
                brainspan_data.get("total_size_mb", 0.0)
            ),
            "comprehensive_integration": True
        }
        
        metadata_file = self.data_dir / f"integrated_atlas_{developmental_stage}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(integration_metadata, f, indent=2)
        
        logger.info("Successfully integrated BrainSpan and Allen Atlas data")
        logger.info(f"Integration metadata saved: {metadata_file}")
        
        return enhanced_ref
    
    def _create_synthetic_reference(self, developmental_stage: str) -> AtlasReference:
        """Create synthetic atlas reference for development/testing."""
        logger.info(f"Creating synthetic atlas reference for {developmental_stage}")
        
        # Create synthetic regional segmentation
        dimensions = (64, 96, 64)  # Reasonable size for embryonic brain
        region_labels = np.zeros(dimensions, dtype=int)
        
        # Define synthetic regions based on morphogen patterns
        x_size, y_size, z_size = dimensions
        
        # Forebrain (anterior, label 1)
        region_labels[:, :int(0.3*y_size), :] = 1
        
        # Midbrain (middle, label 2)
        region_labels[:, int(0.3*y_size):int(0.5*y_size), :] = 2
        
        # Hindbrain (posterior-middle, label 3)
        region_labels[:, int(0.5*y_size):int(0.8*y_size), :] = 3
        
        # Spinal cord (posterior, label 4)
        region_labels[:, int(0.8*y_size):, :] = 4
        
        # Add ventral-dorsal patterning
        # Ventral regions (floor plate, motor neurons)
        region_labels[:, :, :int(0.3*z_size)] = 5
        
        # Dorsal regions (roof plate, neural crest)
        region_labels[:, :, int(0.7*z_size):] = 6
        
        # Region names
        region_names = {
            0: "background",
            1: "forebrain",
            2: "midbrain", 
            3: "hindbrain",
            4: "spinal_cord",
            5: "ventral_neural_tube",
            6: "dorsal_neural_tube"
        }
        
        # Create atlas reference
        atlas_ref = AtlasReference(
            atlas_id=f"synthetic_{developmental_stage}",
            developmental_stage=developmental_stage,
            coordinate_system=CoordinateSystem.ALLEN_CCF,
            resolution_um=10.0,  # 10 µm resolution
            dimensions=dimensions,
            region_labels=region_labels,
            region_names=region_names,
            reference_url="synthetic://local"
        )
        
        return atlas_ref
    
    def _create_atlas_reference_from_api(self, api_data: Dict, 
                                        developmental_stage: str) -> AtlasReference:
        """Create atlas reference from Allen API data."""
        # Extract relevant information from API response
        atlas_id = api_data.get("id", "unknown")
        
        # For now, create synthetic data as Allen API structure is complex
        # In production, would parse actual Allen data format
        return self._create_synthetic_reference(developmental_stage)
    
    def _load_cached_atlas(self, atlas_file: Path, metadata_file: Path) -> AtlasReference:
        """Load atlas data from cache."""
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load atlas data
            atlas_data = np.load(atlas_file)
            
            atlas_ref = AtlasReference(
                atlas_id=metadata["atlas_id"],
                developmental_stage=metadata["developmental_stage"],
                coordinate_system=CoordinateSystem(metadata["coordinate_system"]),
                resolution_um=metadata["resolution_um"],
                dimensions=tuple(metadata["dimensions"]),
                region_labels=atlas_data["region_labels"],
                region_names=metadata["region_names"],
                reference_url=metadata["reference_url"]
            )
            
            logger.info(f"Loaded cached atlas data: {atlas_ref.atlas_id}")
            return atlas_ref
            
        except Exception as e:
            logger.error(f"Failed to load cached atlas: {e}")
            raise
    
    def _save_atlas_to_cache(self, atlas_ref: AtlasReference, 
                            atlas_file: Path, metadata_file: Path) -> None:
        """Save atlas data to cache."""
        try:
            # Save atlas data
            np.savez_compressed(atlas_file, region_labels=atlas_ref.region_labels)
            
            # Save metadata
            metadata = {
                "atlas_id": atlas_ref.atlas_id,
                "developmental_stage": atlas_ref.developmental_stage,
                "coordinate_system": atlas_ref.coordinate_system.value,
                "resolution_um": atlas_ref.resolution_um,
                "dimensions": list(atlas_ref.dimensions),
                "region_names": atlas_ref.region_names,
                "reference_url": atlas_ref.reference_url
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Cached atlas data: {atlas_file}")
            
        except Exception as e:
            logger.error(f"Failed to cache atlas data: {e}")
    
    def list_available_stages(self) -> List[str]:
        """List available developmental stages."""
        return self.available_stages.copy()
    
    def get_cached_atlases(self) -> List[str]:
        """Get list of cached atlas stages."""
        cached_stages = []
        
        for stage in self.available_stages:
            atlas_file = self.data_dir / f"allen_atlas_{stage}.npz"
            metadata_file = self.data_dir / f"allen_atlas_{stage}_metadata.json"
            
            if atlas_file.exists() and metadata_file.exists():
                cached_stages.append(stage)
        
        return cached_stages
    
    def validate_atlas_integrity(self, atlas_ref: AtlasReference) -> Dict[str, bool]:
        """Validate integrity of downloaded atlas data."""
        validation_results = {}
        
        # Check dimensions
        validation_results["dimensions_valid"] = len(atlas_ref.dimensions) == 3
        
        # Check region labels
        validation_results["labels_valid"] = atlas_ref.region_labels is not None
        
        # Check region names
        validation_results["names_valid"] = len(atlas_ref.region_names) > 0
        
        # Check coordinate system
        validation_results["coordinates_valid"] = atlas_ref.coordinate_system in CoordinateSystem
        
        # Check resolution
        validation_results["resolution_valid"] = atlas_ref.resolution_um > 0
        
        # Overall integrity
        validation_results["overall_valid"] = all(validation_results.values())
        
        return validation_results
