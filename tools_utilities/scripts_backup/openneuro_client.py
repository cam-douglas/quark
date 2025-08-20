"""
OpenNeuro Client for BIDS Datasets

Provides access to thousands of fMRI, MEG, and EEG datasets
following the Brain Imaging Data Structure (BIDS) standard.

Reference: https://docs.openneuro.org/user_guide.html#dataset-snapshot
"""

import requests
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
import pandas as pd


class OpenNeuroClient:
    """Client for OpenNeuro BIDS datasets"""
    
    def __init__(self, api_base_url: str = "https://openneuro.org/api"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
    def search_datasets(self, query: str = "", 
                       modality: Optional[str] = None,
                       task: Optional[str] = None,
                       license: Optional[str] = None) -> List[Dict]:
        """
        Search for BIDS datasets
        
        Args:
            query: Text search query
            modality: Filter by modality ('fMRI', 'MEG', 'EEG')
            task: Filter by task type
            license: Filter by license type
            
        Returns:
            List of matching dataset dictionaries
        """
        endpoint = f"{self.api_base_url}/datasets/search"
        params = {"q": query}
        
        if modality:
            params["modality"] = modality
        if task:
            params["task"] = task
        if license:
            params["license"] = license
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()["datasets"]
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """
        Get detailed dataset information
        
        Args:
            dataset_id: OpenNeuro dataset identifier
            
        Returns:
            Dataset information dictionary
        """
        endpoint = f"{self.api_base_url}/datasets/{dataset_id}"
        response = self.session.get(endpoint)
        response.raise_for_status()
        
        return response.json()
    
    def get_dataset_files(self, dataset_id: str, 
                         snapshot: Optional[str] = None) -> List[Dict]:
        """
        Get list of files in a dataset
        
        Args:
            dataset_id: OpenNeuro dataset identifier
            snapshot: Optional snapshot version
            
        Returns:
            List of file information dictionaries
        """
        endpoint = f"{self.api_base_url}/datasets/{dataset_id}/files"
        if snapshot:
            endpoint += f"/{snapshot}"
            
        response = self.session.get(endpoint)
        response.raise_for_status()
        
        return response.json()["files"]
    
    def download_dataset(self, dataset_id: str, 
                        output_path: Path,
                        snapshot: Optional[str] = None,
                        include_derivatives: bool = False) -> Path:
        """
        Download a complete BIDS dataset
        
        Args:
            dataset_id: OpenNeuro dataset identifier
            output_path: Local directory to save dataset
            snapshot: Optional snapshot version
            include_derivatives: Whether to include derivative files
            
        Returns:
            Path to downloaded dataset
        """
        # Implementation for dataset download
        # This would integrate with openneuro-cli or direct API
        output_path.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def validate_bids(self, dataset_path: Path) -> Dict:
        """
        Validate BIDS compliance of a local dataset
        
        Args:
            dataset_path: Path to local BIDS dataset
            
        Returns:
            Validation results dictionary
        """
        # This would integrate with BIDS validator
        # For now, return placeholder
        return {
            "valid": True,
            "errors": [],
            "warnings": []
        }
    
    def get_participant_info(self, dataset_id: str) -> pd.DataFrame:
        """
        Get participant information from participants.tsv
        
        Args:
            dataset_id: OpenNeuro dataset identifier
            
        Returns:
            DataFrame with participant information
        """
        files = self.get_dataset_files(dataset_id)
        participants_file = next((f for f in files if f["name"] == "participants.tsv"), None)
        
        if participants_file:
            # Download and parse participants.tsv
            # Implementation would download file and return DataFrame
            pass
            
        return pd.DataFrame()
