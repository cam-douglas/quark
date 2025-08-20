"""
DANDI (Neurodata Without Borders) Client

Provides access to 300+ TB of neurophysiology data including:
- Optical imaging
- Electrophysiology recordings
- Behavioral data
- Programmatic streaming capabilities

Reference: https://www.biorxiv.org/content/10.1101/2025.07.17.663965v3
"""

import requests
import json
from typing import Dict, List, Optional, Union, Iterator
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import logging


class DANDIClient:
    """Client for DANDI Neurodata Without Borders platform"""
    
    def __init__(self, api_base_url: str = "https://api.dandiarchive.org/api"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def search_datasets(self, query: str = "", 
                       species: Optional[str] = None,
                       data_type: Optional[str] = None,
                       brain_region: Optional[str] = None) -> List[Dict]:
        """
        Search for datasets based on criteria
        
        Args:
            query: Text search query
            species: Filter by species (e.g., 'human', 'mouse', 'rat')
            data_type: Filter by data type (e.g., 'ephys', 'optical', 'behavior')
            brain_region: Filter by brain region
            
        Returns:
            List of matching dataset dictionaries
        """
        endpoint = f"{self.api_base_url}/datasets/search"
        params = {
            "query": query,
            "format": "json"
        }
        
        if species:
            params["species"] = species
        if data_type:
            params["data_type"] = data_type
        if brain_region:
            params["brain_region"] = brain_region
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()["results"]
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """
        Get detailed information about a specific dataset
        
        Args:
            dataset_id: DANDI dataset identifier
            
        Returns:
            Dataset information dictionary
        """
        endpoint = f"{self.api_base_url}/datasets/{dataset_id}"
        response = self.session.get(endpoint, params={"format": "json"})
        response.raise_for_status()
        
        return response.json()
    
    def list_dataset_files(self, dataset_id: str, 
                          file_type: Optional[str] = None) -> List[Dict]:
        """
        List files available in a dataset
        
        Args:
            dataset_id: DANDI dataset identifier
            file_type: Optional file type filter (e.g., '.nwb', '.json')
            
        Returns:
            List of file information dictionaries
        """
        endpoint = f"{self.api_base_url}/datasets/{dataset_id}/files"
        params = {"format": "json"}
        
        if file_type:
            params["file_type"] = file_type
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        files = response.json()["files"]
        
        if file_type:
            files = [f for f in files if f["path"].endswith(file_type)]
            
        return files
    
    def stream_nwb_data(self, dataset_id: str, file_path: str, 
                       chunk_size: int = 1024*1024) -> Iterator[bytes]:
        """
        Stream NWB file data in chunks
        
        Args:
            dataset_id: DANDI dataset identifier
            file_path: Path to file within dataset
            chunk_size: Size of data chunks to yield
            
        Yields:
            Data chunks as bytes
        """
        endpoint = f"{self.api_base_url}/datasets/{dataset_id}/files/{file_path}/download"
        
        response = self.session.get(endpoint, stream=True)
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk
    
    def download_nwb_file(self, dataset_id: str, file_path: str, 
                         output_path: Path) -> Path:
        """
        Download a complete NWB file
        
        Args:
            dataset_id: DANDI dataset identifier
            file_path: Path to file within dataset
            output_path: Local path to save file
            
        Returns:
            Path to downloaded file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in self.stream_nwb_data(dataset_id, file_path):
                f.write(chunk)
                
        return output_path
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict:
        """
        Get comprehensive metadata for a dataset
        
        Args:
            dataset_id: DANDI dataset identifier
            
        Returns:
            Complete dataset metadata
        """
        # Get basic dataset info
        dataset_info = self.get_dataset_info(dataset_id)
        
        # Get file listing
        files = self.list_dataset_files(dataset_id)
        
        # Get NWB metadata for key files
        nwb_metadata = {}
        for file_info in files:
            if file_info["path"].endswith('.nwb'):
                try:
                    # This would require NWB file parsing
                    # For now, return basic file info
                    nwb_metadata[file_info["path"]] = {
                        "size": file_info.get("size"),
                        "modified": file_info.get("modified"),
                        "path": file_info["path"]
                    }
                except Exception as e:
                    self.logger.warning(f"Could not parse NWB metadata for {file_info['path']}: {e}")
        
        metadata = {
            "dataset_info": dataset_info,
            "files": files,
            "nwb_metadata": nwb_metadata,
            "total_size": sum(f.get("size", 0) for f in files),
            "file_count": len(files)
        }
        
        return metadata
    
    def search_by_anatomy(self, brain_region: str, 
                         hemisphere: Optional[str] = None) -> List[Dict]:
        """
        Search datasets by anatomical criteria
        
        Args:
            brain_region: Brain region of interest
            hemisphere: Optional hemisphere filter ('left', 'right', 'both')
            
        Returns:
            List of matching datasets
        """
        query = f"brain_region:{brain_region}"
        if hemisphere:
            query += f" hemisphere:{hemisphere}"
            
        return self.search_datasets(query=query)
    
    def get_available_species(self) -> List[str]:
        """
        Get list of available species in DANDI
        
        Returns:
            List of species names
        """
        endpoint = f"{self.api_base_url}/species"
        response = self.session.get(endpoint, params={"format": "json"})
        response.raise_for_status()
        
        return response.json()["species"]
    
    def get_available_brain_regions(self) -> List[str]:
        """
        Get list of available brain regions in DANDI
        
        Returns:
            List of brain region names
        """
        endpoint = f"{self.api_base_url}/brain_regions"
        response = self.session.get(endpoint, params={"format": "json"})
        response.raise_for_status()
        
        return response.json()["brain_regions"]
    
    def estimate_download_size(self, dataset_id: str) -> Dict[str, Union[int, str]]:
        """
        Estimate download size for a dataset
        
        Args:
            dataset_id: DANDI dataset identifier
            
        Returns:
            Dictionary with size estimates and recommendations
        """
        metadata = self.get_dataset_metadata(dataset_id)
        total_size = metadata["total_size"]
        
        # Convert to human-readable format
        if total_size > 1024**4:  # TB
            size_str = f"{total_size / (1024**4):.2f} TB"
        elif total_size > 1024**3:  # GB
            size_str = f"{total_size / (1024**3):.2f} GB"
        elif total_size > 1024**2:  # MB
            size_str = f"{total_size / (1024**2):.2f} MB"
        else:
            size_str = f"{total_size} bytes"
        
        return {
            "total_size_bytes": total_size,
            "total_size_human": size_str,
            "file_count": metadata["file_count"],
            "recommendation": "streaming" if total_size > 1024**3 else "download"
        }
