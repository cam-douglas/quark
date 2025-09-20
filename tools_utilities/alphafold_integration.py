#!/usr/bin/env python3
"""
AlphaFold Protein Structure Database Integration for Quark
===========================================================
This module provides integration with the AlphaFold DB API from DeepMind/EMBL-EBI.

AlphaFold predicts 3D protein structures from amino acid sequences with accuracy
competitive with experimental methods. The database contains 200+ million predictions.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from typing_extensions import Literal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    ALPHAFOLD_CONFIG = credentials['services']['alphafold']

# API endpoints
BASE_URL = ALPHAFOLD_CONFIG['endpoints']['base']
PREDICTION_URL = ALPHAFOLD_CONFIG['endpoints']['prediction']
FILES_URL = ALPHAFOLD_CONFIG['endpoints']['structure_file']


class AlphaFoldClient:
    """Client for interacting with AlphaFold Protein Structure Database API."""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize AlphaFold client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quark-AlphaFold-Integration/1.0'
        })
    
    def get_structure_info(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get metadata and information about a protein structure prediction.
        
        Args:
            uniprot_id: UniProt accession ID (e.g., "P00533" for human EGFR)
            
        Returns:
            Dictionary with structure metadata including pLDDT scores
        """
        url = f"{PREDICTION_URL}/{uniprot_id.upper()}"
        
        logger.info(f"Fetching structure info for UniProt ID: {uniprot_id}")
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse response - returns list of predictions
            data = response.json()
            if data:
                # Usually returns single prediction for UniProt ID
                prediction = data[0] if isinstance(data, list) else data
                
                # Extract pLDDT if available
                plddt = prediction.get('confidenceAvg') or prediction.get('plddt_avg') or prediction.get('plddt')
                logger.info(f"Structure found - Mean confidence: {plddt if plddt else 'N/A'}")
                
                # Normalize field names for consistency
                normalized = {
                    'uniprot_id': prediction.get('uniprotAccession', prediction.get('uniprot_id', uniprot_id)),
                    'plddt_avg': plddt,
                    'confidence_version': prediction.get('confidenceVersion'),
                    'model_version': prediction.get('modelCreatedDate') or prediction.get('latestVersion', 4),
                    'sequence_length': prediction.get('uniprotEnd', 0) - prediction.get('uniprotStart', 0) + 1,
                    'organism_scientific_name': prediction.get('organismScientificName'),
                    'gene': prediction.get('gene'),
                    'protein_full_name': prediction.get('uniprotDescription'),
                    'entry_id': prediction.get('entryId'),
                    'fragment': prediction.get('isFragment', False),
                    'is_reviewed': prediction.get('isReviewed', False)
                }
                return normalized
            else:
                logger.warning(f"No structure found for {uniprot_id}")
                return {}
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"UniProt ID {uniprot_id} not found in AlphaFold DB")
            else:
                logger.error(f"HTTP Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching structure info: {e}")
            raise
    
    def download_structure(
        self, 
        uniprot_id: str,
        format: Literal["pdb", "cif", "mmcif"] = "pdb",
        output_dir: Optional[Path] = None,
        version: int = 4  # Latest AlphaFold version
    ) -> Path:
        """
        Download protein structure file from AlphaFold.
        
        Args:
            uniprot_id: UniProt accession ID
            format: File format (pdb, cif, or mmcif)
            output_dir: Directory to save file (default: current directory)
            version: AlphaFold model version (default: 4)
            
        Returns:
            Path to downloaded file
        """
        # Construct file URL
        filename = f"AF-{uniprot_id.upper()}-F1-model_v{version}.{format}"
        url = f"{FILES_URL}/{filename}"
        
        # Set output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        logger.info(f"Downloading structure: {url}")
        try:
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Write file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Structure saved to: {output_path}")
            return output_path
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Structure file not found for {uniprot_id}")
            else:
                logger.error(f"HTTP Error: {e}")
            raise
    
    def download_pae(
        self,
        uniprot_id: str,
        output_dir: Optional[Path] = None,
        version: int = 4
    ) -> Path:
        """
        Download Predicted Aligned Error (PAE) JSON file.
        
        Args:
            uniprot_id: UniProt accession ID
            output_dir: Directory to save file
            version: AlphaFold model version
            
        Returns:
            Path to downloaded PAE JSON file
        """
        # Construct PAE URL
        filename = f"AF-{uniprot_id.upper()}-F1-predicted_aligned_error_v{version}.json"
        url = f"{FILES_URL}/{filename}"
        
        # Set output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        logger.info(f"Downloading PAE data: {url}")
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Write JSON file
            with open(output_path, 'w') as f:
                json.dump(response.json(), f, indent=2)
            
            logger.info(f"PAE data saved to: {output_path}")
            return output_path
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Error downloading PAE: {e}")
            raise
    
    def get_confidence_data(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get confidence scores and quality metrics for a structure.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            Dictionary with confidence metrics
        """
        # Get structure info
        info = self.get_structure_info(uniprot_id)
        
        if not info:
            return {}
        
        # Extract confidence metrics
        plddt = info.get("plddt_avg")
        confidence_data = {
            "uniprot_id": uniprot_id,
            "mean_plddt": plddt,
            "confidence_version": info.get("confidence_version"),
            "model_version": info.get("model_version"),
            "sequence_length": info.get("sequence_length"),
            "organism": info.get("organism_scientific_name"),
            "gene_name": info.get("gene"),
            "protein_name": info.get("protein_full_name"),
            "confidence_categories": self._categorize_confidence(plddt)
        }
        
        return confidence_data
    
    def _categorize_confidence(self, plddt: Optional[float]) -> str:
        """
        Categorize pLDDT confidence score.
        
        Args:
            plddt: Mean pLDDT score
            
        Returns:
            Confidence category string
        """
        if plddt is None:
            return "No confidence data available"
        elif plddt >= 90:
            return "Very high confidence"
        elif plddt >= 70:
            return "Confident"
        elif plddt >= 50:
            return "Low confidence"
        else:
            return "Very low confidence"
    
    def bulk_download(
        self,
        uniprot_ids: List[str],
        format: str = "pdb",
        output_dir: Optional[Path] = None,
        delay: float = 0.5
    ) -> List[Path]:
        """
        Download multiple protein structures.
        
        Args:
            uniprot_ids: List of UniProt accession IDs
            format: File format
            output_dir: Directory to save files
            delay: Delay between downloads (seconds)
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        failed_downloads = []
        
        logger.info(f"Starting bulk download of {len(uniprot_ids)} structures")
        
        for i, uniprot_id in enumerate(uniprot_ids, 1):
            try:
                logger.info(f"Downloading {i}/{len(uniprot_ids)}: {uniprot_id}")
                path = self.download_structure(uniprot_id, format, output_dir)
                downloaded_files.append(path)
                
                # Add delay to be respectful to the API
                if i < len(uniprot_ids):
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to download {uniprot_id}: {e}")
                failed_downloads.append(uniprot_id)
        
        logger.info(f"Download complete: {len(downloaded_files)} successful, {len(failed_downloads)} failed")
        
        if failed_downloads:
            logger.warning(f"Failed IDs: {', '.join(failed_downloads)}")
        
        return downloaded_files
    
    def search_by_gene(self, gene_name: str, organism: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for proteins by gene name (simplified search).
        
        Note: Full search API requires additional implementation.
        This is a placeholder for demonstration.
        
        Args:
            gene_name: Gene name to search for
            organism: Optional organism filter
            
        Returns:
            List of matching proteins
        """
        logger.info(f"Searching for gene: {gene_name}")
        
        # Note: The actual search endpoint would require more complex implementation
        # This is a simplified example
        search_results = []
        
        # Common UniProt IDs for demonstration
        # In production, this would query the actual search API
        example_mappings = {
            "EGFR": ["P00533"],  # Human EGFR
            "TP53": ["P04637"],  # Human p53
            "BRCA1": ["P38398"], # Human BRCA1
            "APP": ["P05067"],   # Human Amyloid precursor protein
            "MAPT": ["P10636"],  # Human Tau protein
            "SNCA": ["P37840"],  # Human Alpha-synuclein
        }
        
        if gene_name.upper() in example_mappings:
            for uniprot_id in example_mappings[gene_name.upper()]:
                try:
                    info = self.get_structure_info(uniprot_id)
                    if info:
                        search_results.append(info)
                except Exception:
                    pass
        
        return search_results


def fetch_brain_proteins(client: AlphaFoldClient, output_dir: Path) -> Dict[str, Any]:
    """
    Fetch AlphaFold structures for key brain-related proteins.
    
    Args:
        client: AlphaFold client instance
        output_dir: Directory to save structures
        
    Returns:
        Dictionary with download results
    """
    # Key brain proteins and their UniProt IDs
    brain_proteins = {
        # Neurotransmitter receptors
        "DRD2_dopamine_receptor": "P14416",
        "HTR2A_serotonin_receptor": "P28223", 
        "GABRA1_gaba_receptor": "P14867",
        "GRIN1_nmda_receptor": "Q05586",
        
        # Ion channels
        "SCN1A_sodium_channel": "P35498",
        "KCNQ2_potassium_channel": "O43526",
        "CACNA1A_calcium_channel": "O00555",
        
        # Synaptic proteins
        "SYN1_synapsin": "P17600",
        "SYT1_synaptotagmin": "P21579",
        "SNAP25_snap25": "P60880",
        
        # Neurodegenerative disease proteins
        "APP_amyloid_precursor": "P05067",
        "MAPT_tau": "P10636",
        "SNCA_alpha_synuclein": "P37840",
        "HTT_huntingtin": "P42858"
    }
    
    results = {
        "downloaded": [],
        "failed": [],
        "confidence_scores": {}
    }
    
    print("\nFetching brain protein structures from AlphaFold...")
    print("-" * 50)
    
    for protein_name, uniprot_id in brain_proteins.items():
        try:
            # Get confidence data
            confidence = client.get_confidence_data(uniprot_id)
            results["confidence_scores"][protein_name] = confidence
            
            print(f"\n{protein_name} ({uniprot_id}):")
            print(f"  Protein: {confidence.get('protein_name', 'N/A')}")
            
            plddt = confidence.get('mean_plddt')
            if plddt is not None:
                print(f"  Mean pLDDT: {plddt:.2f}")
            else:
                print(f"  Mean pLDDT: N/A")
            
            print(f"  Confidence: {confidence.get('confidence_categories', 'N/A')}")
            
            # Download structure
            structure_path = client.download_structure(uniprot_id, "pdb", output_dir)
            results["downloaded"].append(str(structure_path))
            
            # Small delay to be respectful
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Error: {e}")
            results["failed"].append(uniprot_id)
    
    return results


def main():
    """Example usage of AlphaFold client."""
    client = AlphaFoldClient()
    
    print("=" * 60)
    print("AlphaFold Protein Structure Database Integration")
    print("Quark System - Brain Protein Analysis")
    print("=" * 60)
    
    # Create output directory for structures
    output_dir = Path(__file__).parent.parent / "data" / "structures" / "alphafold"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example 1: Get structure info for human p53
    print("\n1. Fetching human p53 tumor suppressor (TP53)...")
    try:
        p53_info = client.get_structure_info("P04637")
        if p53_info:
            print(f"  Protein: {p53_info.get('protein_full_name', 'N/A')}")
            print(f"  Organism: {p53_info.get('organism_scientific_name', 'N/A')}")
            
            plddt = p53_info.get('plddt_avg')
            if plddt is not None:
                print(f"  Mean pLDDT: {plddt:.2f}")
            else:
                print(f"  Mean pLDDT: N/A")
            
            print(f"  Length: {p53_info.get('sequence_length', 'N/A')} residues")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Example 2: Download structure for dopamine receptor
    print("\n2. Downloading dopamine D2 receptor structure...")
    try:
        drd2_path = client.download_structure("P14416", "pdb", output_dir)
        print(f"  Saved to: {drd2_path}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Example 3: Get confidence data for amyloid precursor protein
    print("\n3. Analyzing amyloid precursor protein (Alzheimer's)...")
    try:
        app_confidence = client.get_confidence_data("P05067")
        print(f"  Gene: {app_confidence.get('gene_name', 'N/A')}")
        
        plddt = app_confidence.get('mean_plddt')
        if plddt is not None:
            print(f"  Mean pLDDT: {plddt:.2f}")
        else:
            print(f"  Mean pLDDT: N/A")
        
        print(f"  Assessment: {app_confidence.get('confidence_categories', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Example 4: Fetch multiple brain proteins
    brain_results = fetch_brain_proteins(client, output_dir)
    
    # Save results summary
    summary_path = output_dir / "brain_proteins_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(brain_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Downloaded {len(brain_results['downloaded'])} structures")
    print(f"Results saved to: {summary_path}")
    print("AlphaFold integration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
