"""
Allen Brain Cell Types Database Interface

Provides access to:
- Electrophysiological recordings
- Morphological reconstructions  
- Transcriptomic profiles
- Downloadable neuron models
- Allen SDK integration

Reference: https://celltypes.brain-map.org/
"""

import requests
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd


class AllenBrainInterface:
    """Interface for Allen Brain Cell Types Database"""
    
    def __init__(self, api_base_url: str = "https://api.brain-map.org/api/v2"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
    def get_cell_types(self, species: str = "human", 
                       brain_region: Optional[str] = None) -> List[Dict]:
        """
        Retrieve cell type information
        
        Args:
            species: 'human' or 'mouse'
            brain_region: Optional brain region filter
            
        Returns:
            List of cell type dictionaries
        """
        endpoint = f"{self.api_base_url}/cell_types"
        params = {
            "species": species,
            "format": "json"
        }
        
        if brain_region:
            params["brain_region"] = brain_region
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()["msg"]
    
    def get_electrophysiology(self, cell_id: int) -> Dict:
        """
        Get electrophysiological data for a specific cell
        
        Args:
            cell_id: Allen Brain cell ID
            
        Returns:
            Electrophysiology data dictionary
        """
        endpoint = f"{self.api_base_url}/electrophysiology/{cell_id}"
        response = self.session.get(endpoint, params={"format": "json"})
        response.raise_for_status()
        
        return response.json()["msg"]
    
    def get_morphology(self, cell_id: int) -> Dict:
        """
        Get morphological reconstruction data for a specific cell
        
        Args:
            cell_id: Allen Brain cell ID
            
        Returns:
            Morphology data dictionary
        """
        endpoint = f"{self.api_base_url}/morphology/{cell_id}"
        response = self.session.get(endpoint, params={"format": "json"})
        response.raise_for_status()
        
        return response.json()["msg"]
    
    def get_transcriptomics(self, cell_id: int) -> Dict:
        """
        Get transcriptomic profile for a specific cell
        
        Args:
            cell_id: Allen Brain cell ID
            
        Returns:
            Transcriptomics data dictionary
        """
        endpoint = f"{self.api_base_url}/transcriptomics/{cell_id}"
        response = self.session.get(endpoint, params={"format": "json"})
        response.raise_for_status()
        
        return response.json()["msg"]
    
    def download_neuron_model(self, cell_id: int, model_type: str, 
                             output_path: Path) -> Path:
        """
        Download neuron model files
        
        Args:
            cell_id: Allen Brain cell ID
            model_type: 'GLIF', 'perisomatic', or 'all-active'
            output_path: Directory to save model files
            
        Returns:
            Path to downloaded model files
        """
        # Implementation for downloading specific model types
        # This would integrate with Allen SDK for actual downloads
        model_endpoint = f"{self.api_base_url}/models/{cell_id}/{model_type}"
        
        # Placeholder for actual download logic
        output_path.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def search_cells(self, criteria: Dict) -> List[Dict]:
        """
        Search for cells based on multiple criteria
        
        Args:
            criteria: Dictionary of search criteria
                     (e.g., {'species': 'human', 'brain_region': 'cortex'})
                     
        Returns:
            List of matching cell dictionaries
        """
        endpoint = f"{self.api_base_url}/cell_search"
        response = self.session.post(endpoint, json=criteria)
        response.raise_for_status()
        
        return response.json()["msg"]
    
    def get_cell_metadata(self, cell_id: int) -> Dict:
        """
        Get comprehensive metadata for a cell
        
        Args:
            cell_id: Allen Brain cell ID
            
        Returns:
            Complete cell metadata dictionary
        """
        metadata = {}
        
        # Get basic cell info
        cell_info = self.get_cell_types()
        cell_data = next((c for c in cell_info if c["id"] == cell_id), None)
        if cell_data:
            metadata.update(cell_data)
        
        # Get electrophysiology if available
        try:
            ephys_data = self.get_electrophysiology(cell_id)
            metadata["electrophysiology"] = ephys_data
        except:
            metadata["electrophysiology"] = None
            
        # Get morphology if available
        try:
            morph_data = self.get_morphology(cell_id)
            metadata["morphology"] = morph_data
        except:
            metadata["morphology"] = None
            
        # Get transcriptomics if available
        try:
            transcript_data = self.get_transcriptomics(cell_id)
            metadata["transcriptomics"] = transcript_data
        except:
            metadata["transcriptomics"] = None
            
        return metadata
