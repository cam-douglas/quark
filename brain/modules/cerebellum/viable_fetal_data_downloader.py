#!/usr/bin/env python3
"""
Viable Fetal Cerebellar Data Downloader

Downloads actual available fetal cerebellar data from viable sources identified
through research: Allen BrainSpan prenatal MRI/DTI, MDHE Carnegie stages 13-23,
and preparation for EBRAINS/HDBR registration.

Based on research findings:
- Allen BrainSpan: prenatal MRI/DTI specimens (ex vivo)
- MDHE: Carnegie stages 13-23 MR microscopy (39-156 μm resolution)
- EBRAINS: early-PCW ex vivo datasets (registration required)
- HDBR: custom acquisition possible (registration required)

Author: Quark Brain Development Team
Date: 2025-09-17
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.parse
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViableFetalDataDownloader:
    """Downloads fetal cerebellar data from actually available sources."""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize viable data downloader.
        
        Args:
            data_dir: Base directory for cerebellar data storage
        """
        self.data_dir = Path(data_dir)
        self.viable_data_dir = self.data_dir / "viable_fetal_data"
        self.allen_brainspan_dir = self.viable_data_dir / "allen_brainspan"
        self.mdhe_dir = self.viable_data_dir / "mdhe_carnegie"
        self.metadata_dir = self.viable_data_dir / "metadata"
        
        # Create directories
        for directory in [self.viable_data_dir, self.allen_brainspan_dir, self.mdhe_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized viable fetal data downloader")
        logger.info(f"Data directory: {self.viable_data_dir}")
    
    def download_allen_brainspan_prenatal(self) -> Dict[str, any]:
        """Download Allen BrainSpan prenatal MRI/DTI data.
        
        Source: download.alleninstitute.org/brainspan/MRI_DTI_data_for_prenatal_specimens/
        
        Returns:
            Download results and metadata
        """
        logger.info("Downloading Allen BrainSpan prenatal MRI/DTI data")
        
        brainspan_base_url = "https://download.alleninstitute.org/brainspan/MRI_DTI_data_for_prenatal_specimens/"
        
        brainspan_results = {
            "dataset": "Allen_BrainSpan_prenatal",
            "download_date": datetime.now().isoformat(),
            "source_url": brainspan_base_url,
            "files_downloaded": [],
            "download_status": {},
            "total_size_mb": 0
        }
        
        # Common prenatal specimen folders (based on typical BrainSpan structure)
        specimen_folders = [
            "H376.IIIA.02/",  # ~15 PCW
            "H376.IIIB.02/",  # ~16 PCW  
            "H376.IV.02/",    # ~21 PCW
            "H376.IV.03/",    # ~21 PCW
            "prenatal_specimens/",
            "early_development/"
        ]
        
        for folder in specimen_folders:
            try:
                folder_url = brainspan_base_url + folder
                logger.info(f"Attempting to access folder: {folder}")
                
                # Try to get directory listing
                response = requests.get(folder_url, timeout=30)
                if response.status_code == 200:
                    # Save directory listing for manual inspection
                    listing_file = self.allen_brainspan_dir / f"{folder.replace('/', '_')}listing.html"
                    with open(listing_file, 'w') as f:
                        f.write(response.text)
                    
                    brainspan_results["files_downloaded"].append(str(listing_file))
                    brainspan_results["download_status"][folder] = "directory_listing_success"
                    
                    file_size_mb = listing_file.stat().st_size / (1024*1024)
                    brainspan_results["total_size_mb"] += file_size_mb
                    
                    logger.info(f"✅ Downloaded directory listing for {folder}")
                else:
                    brainspan_results["download_status"][folder] = f"failed_http_{response.status_code}"
                    logger.warning(f"⚠️ Failed to access {folder}: HTTP {response.status_code}")
                    
            except Exception as e:
                brainspan_results["download_status"][folder] = f"failed: {str(e)}"
                logger.warning(f"⚠️ Failed to access {folder}: {e}")
        
        # Try direct specimen file downloads (common BrainSpan file patterns)
        specimen_files = [
            "prenatal_specimens.zip",
            "MRI_specimens.tar.gz", 
            "DTI_specimens.tar.gz",
            "metadata.json"
        ]
        
        for filename in specimen_files:
            try:
                file_url = brainspan_base_url + filename
                output_path = self.allen_brainspan_dir / filename
                
                logger.info(f"Attempting to download {filename}")
                urllib.request.urlretrieve(file_url, output_path)
                
                file_size_mb = output_path.stat().st_size / (1024*1024)
                brainspan_results["files_downloaded"].append(str(output_path))
                brainspan_results["download_status"][filename] = "success"
                brainspan_results["total_size_mb"] += file_size_mb
                
                logger.info(f"✅ Downloaded {filename} ({file_size_mb:.1f}MB)")
                
            except Exception as e:
                brainspan_results["download_status"][filename] = f"failed: {str(e)}"
                logger.warning(f"⚠️ Failed to download {filename}: {e}")
        
        return brainspan_results
    
    def download_mdhe_carnegie_stages(self) -> Dict[str, any]:
        """Download MDHE Carnegie stages 13-23 MR microscopy data.
        
        Source: Google Drive links provided in search results
        
        Returns:
            Download results and metadata
        """
        logger.info("Downloading MDHE Carnegie stages 13-23 MR microscopy data")
        
        mdhe_results = {
            "dataset": "MDHE_Carnegie_stages",
            "download_date": datetime.now().isoformat(),
            "source_description": "Multi-Dimensional Human Embryo MR microscopy",
            "target_stages": "CS13-CS23 (gestational weeks 6-8)",
            "resolution": "39-156 μm isotropic",
            "modalities": ["T1", "T2", "DWI"],
            "files_downloaded": [],
            "download_status": {},
            "total_size_mb": 0
        }
        
        # Google Drive file IDs from search results
        gdrive_files = {
            "mdhe_dataset_1": {
                "file_id": "1yLM6bt6BfaY51gnG9uTBr57BftkQ7CQ_",
                "description": "MDHE Carnegie stages dataset 1"
            },
            "mdhe_dataset_2": {
                "file_id": "1upTru9tiwUiJCfDmTu96RGijf3T0KUUS", 
                "description": "MDHE Carnegie stages dataset 2"
            }
        }
        
        for dataset_name, file_info in gdrive_files.items():
            try:
                file_id = file_info["file_id"]
                
                # Try direct download URL (may require authentication)
                download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                output_path = self.mdhe_dir / f"{dataset_name}.zip"
                
                logger.info(f"Attempting to download {dataset_name} from Google Drive")
                
                # Use requests with session to handle redirects
                session = requests.Session()
                response = session.get(download_url, timeout=60)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    file_size_mb = output_path.stat().st_size / (1024*1024)
                    mdhe_results["files_downloaded"].append(str(output_path))
                    mdhe_results["download_status"][dataset_name] = "success"
                    mdhe_results["total_size_mb"] += file_size_mb
                    
                    logger.info(f"✅ Downloaded {dataset_name} ({file_size_mb:.1f}MB)")
                else:
                    # Handle Google Drive authentication redirect
                    if "accounts.google.com" in response.url or response.status_code == 302:
                        mdhe_results["download_status"][dataset_name] = "requires_authentication"
                        logger.warning(f"⚠️ {dataset_name} requires Google authentication")
                    else:
                        mdhe_results["download_status"][dataset_name] = f"failed_http_{response.status_code}"
                        logger.warning(f"⚠️ Failed to download {dataset_name}: HTTP {response.status_code}")
                        
            except Exception as e:
                mdhe_results["download_status"][dataset_name] = f"failed: {str(e)}"
                logger.error(f"❌ Failed to download {dataset_name}: {e}")
        
        return mdhe_results
    
    def create_registration_instructions(self) -> Dict[str, any]:
        """Create detailed registration instructions for restricted datasets.
        
        Returns:
            Registration procedures and contact information
        """
        logger.info("Creating registration instructions for restricted datasets")
        
        instructions = {
            "creation_date": datetime.now().isoformat(),
            "registration_required": {
                "EBRAINS": {
                    "url": "https://ebrains.eu/register",
                    "data_portal": "https://search.kg.ebrains.eu/",
                    "search_terms": ["fetal brain development", "early PCW", "cerebellar development"],
                    "contact": "support@ebrains.eu",
                    "institution": "Jülich Research Centre",
                    "expected_data": "Early-PCW ex vivo MRI, <0.5mm resolution",
                    "access_method": "Human Data Gateway request"
                },
                "HDBR": {
                    "url": "https://www.hdbr.org/",
                    "registration_url": "https://www.hdbr.org/register",
                    "project_application": "Required for tissue/imaging requests",
                    "contact": "info@hdbr.org",
                    "institution": "MRC-Wellcome Trust Human Developmental Biology Resource",
                    "custom_acquisition": "Can coordinate MR microscopy on CS18-23 embryos",
                    "resolution_capability": "Tens of microns (exceeds requirement)",
                    "request_specifics": "Posterior fossa/cerebellum intact blocks, CS18-23"
                }
            },
            "alternative_contacts": {
                "BCH_CRL": {
                    "primary_contact": "ali.gholipour@childrens.harvard.edu",
                    "institution": "Computational Radiology Laboratory, Boston Children's Hospital",
                    "dataset": "Fetal Brain Atlas v2.0",
                    "backup_contact": "simon.warfield@childrens.harvard.edu"
                },
                "Allen_Institute": {
                    "contact": "brainspan-help@alleninstitute.org",
                    "data_portal": "https://www.brainspan.org/",
                    "specific_dataset": "prenatal MRI/DTI specimens"
                }
            },
            "email_templates": {
                "HDBR_request": """
Subject: Research Request - Early Fetal Cerebellar MR Microscopy (CS18-23)

Dear HDBR Team,

I am writing to request access to human embryonic specimens for high-resolution MR microscopy imaging of early cerebellar development.

Project: Computational modeling of cerebellar morphogenesis
Institution: Independent research (Quark Brain Development Project)

Specific Requirements:
- Carnegie stages: 18-23 (gestational weeks 7-8)
- Anatomical focus: Posterior fossa, cerebellum, rhombic lip
- Imaging modality: Ex vivo MR microscopy
- Target resolution: ≤0.5mm isotropic (preferably tens of microns)
- Sequences: T1-weighted, T2-weighted, optional DWI
- Sample size: 3-5 specimens per Carnegie stage
- File format: NIfTI or convertible format

Use case: Validation of morphogen gradient models for cerebellar specification and early foliation patterns during the critical period of Math1/Atoh1+ granule precursor specification and Ptf1a+ GABAergic lineage determination.

Could you please advise on:
1. Specimen availability for CS18-23
2. MR microscopy imaging capabilities and protocols
3. Data sharing procedures and requirements
4. Timeline for acquisition and processing

Thank you for your consideration.

Best regards,
[Your name and affiliation]
                """,
                "EBRAINS_request": """
Subject: Data Access Request - Early Fetal Brain Development (Cerebellar Focus)

Dear EBRAINS Support Team,

I am requesting access to early fetal brain development datasets through the Human Data Gateway, specifically focusing on cerebellar development during early post-conception weeks.

Research Project: Computational modeling of human cerebellar morphogenesis
Data Requirements:
- Gestational ages: 8-12 weeks post-conception
- Anatomical focus: Cerebellum, posterior fossa, rhombic lip
- Modality: Ex vivo MRI (T1, T2, DWI)
- Resolution: <0.5mm isotropic
- Sample size: Minimum 15-20 specimens

Search terms I will use:
- "fetal brain development"
- "early PCW" 
- "cerebellar development"
- "posterior fossa"
- "rhombic lip"

Could you please provide guidance on:
1. Registration process completion
2. Data search and request procedures
3. Available early fetal datasets with cerebellar coverage
4. Access timeline and requirements

Reference: Amunts et al. 2020, Nature Scientific Data

Thank you,
[Your name and affiliation]
                """
            }
        }
        
        return instructions
    
    def execute_viable_downloads(self) -> Dict[str, any]:
        """Execute downloads from all viable sources.
        
        Returns:
            Comprehensive download results
        """
        logger.info("Executing downloads from viable fetal cerebellar data sources")
        
        execution_results = {
            "execution_date": datetime.now().isoformat(),
            "viable_sources_attempted": [],
            "successful_downloads": [],
            "total_data_mb": 0,
            "download_details": {},
            "registration_instructions": {}
        }
        
        # 1. Download Allen BrainSpan prenatal data
        logger.info("=== Allen BrainSpan Prenatal MRI/DTI ===")
        brainspan_results = self.download_allen_brainspan_prenatal()
        execution_results["viable_sources_attempted"].append("Allen_BrainSpan")
        execution_results["download_details"]["Allen_BrainSpan"] = brainspan_results
        
        if any("success" in str(status) for status in brainspan_results["download_status"].values()):
            execution_results["successful_downloads"].append("Allen_BrainSpan")
            execution_results["total_data_mb"] += brainspan_results["total_size_mb"]
        
        # 2. Download MDHE Carnegie stages data
        logger.info("=== MDHE Carnegie Stages 13-23 ===")
        mdhe_results = self.download_mdhe_carnegie_stages()
        execution_results["viable_sources_attempted"].append("MDHE_Carnegie")
        execution_results["download_details"]["MDHE_Carnegie"] = mdhe_results
        
        if any("success" in str(status) for status in mdhe_results["download_status"].values()):
            execution_results["successful_downloads"].append("MDHE_Carnegie")
            execution_results["total_data_mb"] += mdhe_results["total_size_mb"]
        
        # 3. Create registration instructions
        logger.info("=== Creating Registration Instructions ===")
        registration_instructions = self.create_registration_instructions()
        execution_results["registration_instructions"] = registration_instructions
        
        # Save comprehensive results
        results_file = self.metadata_dir / "viable_download_results.json"
        with open(results_file, 'w') as f:
            json.dump(execution_results, f, indent=2)
        
        # Save registration instructions separately
        registration_file = self.metadata_dir / "registration_instructions.json"
        with open(registration_file, 'w') as f:
            json.dump(registration_instructions, f, indent=2)
        
        logger.info(f"Viable downloads completed. Results saved to {results_file}")
        return execution_results


def main():
    """Execute viable fetal cerebellar data downloads."""
    
    print("🧠 VIABLE FETAL CEREBELLAR DATA DOWNLOAD")
    print("=" * 60)
    print("Phase 1 ▸ Batch A ▸ Step A1.4 - VIABLE SOURCES EXECUTION")
    print()
    print("Sources identified from research:")
    print("• Allen BrainSpan: prenatal MRI/DTI specimens")
    print("• MDHE: Carnegie stages 13-23 MR microscopy")
    print("• EBRAINS: early-PCW datasets (registration)")
    print("• HDBR: custom acquisition (registration)")
    print()
    
    # Initialize downloader
    downloader = ViableFetalDataDownloader()
    
    # Execute downloads
    results = downloader.execute_viable_downloads()
    
    # Print execution summary
    print(f"✅ Viable source downloads completed")
    print(f"📊 Sources attempted: {len(results['viable_sources_attempted'])}")
    print(f"✅ Successful downloads: {len(results['successful_downloads'])}")
    print(f"💾 Total data downloaded: {results['total_data_mb']:.1f}MB")
    print()
    
    # Display download details
    print("📥 Download Results:")
    for source_name, details in results['download_details'].items():
        if isinstance(details, dict) and 'download_status' in details:
            success_count = sum(1 for status in details['download_status'].values() 
                              if 'success' in str(status))
            total_count = len(details['download_status'])
            print(f"  • {source_name}: {success_count}/{total_count} items successful")
            if 'total_size_mb' in details:
                print(f"    Size: {details['total_size_mb']:.1f}MB")
    
    print()
    print("📋 Registration Required:")
    reg_info = results['registration_instructions']['registration_required']
    for service, info in reg_info.items():
        print(f"  • {service}: {info['url']}")
        print(f"    Expected data: {info['expected_data']}")
    
    print()
    print("📁 Data Locations:")
    print(f"  • Allen BrainSpan: {downloader.allen_brainspan_dir}")
    print(f"  • MDHE Carnegie: {downloader.mdhe_dir}")
    print(f"  • Registration info: {downloader.metadata_dir}")
    
    print()
    print("🎯 Next Steps:")
    print("- Review downloaded Allen BrainSpan directory listings")
    print("- Complete EBRAINS and HDBR registration procedures")
    print("- Download MDHE Carnegie stages data from Google Drive")
    print("- Proceed to A1.5: Import zebrin II expression patterns")


if __name__ == "__main__":
    main()
