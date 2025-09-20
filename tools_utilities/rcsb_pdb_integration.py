#!/usr/bin/env python3
"""
RCSB PDB Search API Integration for Quark
==========================================
This module provides integration with the RCSB Protein Data Bank Search and Data APIs.

The RCSB PDB APIs are public and don't require authentication, providing access to:
- Protein structure data (experimental and computed models)
- Sequence searches
- Structure similarity searches
- Chemical compound searches
- Biological assembly information

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from typing_extensions import Literal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    RCSB_CONFIG = credentials['services']['rcsb_pdb']

# API endpoints
SEARCH_ENDPOINT = RCSB_CONFIG['endpoints']['search']
DATA_ENDPOINT = RCSB_CONFIG['endpoints']['data']
GRAPHQL_ENDPOINT = RCSB_CONFIG['endpoints']['graphql']


class RCSBPDBClient:
    """Client for interacting with RCSB PDB Search and Data APIs."""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize RCSB PDB client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Quark-RCSB-Integration/1.0'
        })
    
    def search_text(
        self, 
        query: str, 
        return_type: Literal["entry", "polymer_entity", "assembly", "non_polymer_entity"] = "entry",
        rows: int = 25
    ) -> Dict[str, Any]:
        """
        Perform text search on PDB database.
        
        Args:
            query: Search text (e.g., "COVID-19 spike protein")
            return_type: Type of identifiers to return
            rows: Number of results to return
            
        Returns:
            Search results dictionary
        """
        search_request = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "struct.title",
                    "operator": "contains_words",
                    "value": query
                }
            },
            "return_type": return_type,
            "request_options": {
                "paginate": {
                    "rows": rows,
                    "start": 0
                }
            }
        }
        
        logger.info(f"Searching for: {query}")
        try:
            response = self.session.post(SEARCH_ENDPOINT, json=search_request, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response content: {response.text}")
            raise
    
    def search_sequence(
        self,
        sequence: str,
        identity_threshold: float = 0.9,
        return_type: str = "polymer_entity",
        rows: int = 25
    ) -> Dict[str, Any]:
        """
        Search for structures with similar sequences.
        
        Args:
            sequence: Protein sequence (single letter amino acid codes)
            identity_threshold: Sequence identity threshold (0.0 to 1.0)
            return_type: Type of identifiers to return
            rows: Number of results to return
            
        Returns:
            Search results dictionary
        """
        search_request = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "value": sequence,
                    "identity_cutoff": identity_threshold,
                    "target": "pdb_protein_sequence"
                }
            },
            "return_type": return_type,
            "request_options": {
                "paginate": {
                    "rows": rows,
                    "start": 0
                }
            }
        }
        
        logger.info(f"Searching for sequences with {identity_threshold*100}% identity")
        response = self.session.post(SEARCH_ENDPOINT, json=search_request, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def search_by_organism(
        self,
        organism: str,
        experimental_method: Optional[str] = None,
        resolution_max: Optional[float] = None,
        rows: int = 25
    ) -> Dict[str, Any]:
        """
        Search for structures from a specific organism.
        
        Args:
            organism: Organism name (e.g., "Homo sapiens")
            experimental_method: Optional experimental method filter (e.g., "X-RAY DIFFRACTION")
            resolution_max: Maximum resolution in Angstroms
            rows: Number of results to return
            
        Returns:
            Search results dictionary
        """
        queries = []
        
        # Add organism query
        queries.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.scientific_name",
                "operator": "exact_match",
                "value": organism
            }
        })
        
        # Add experimental method filter if specified
        if experimental_method:
            queries.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": experimental_method
                }
            })
        
        # Add resolution filter if specified
        if resolution_max:
            queries.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": resolution_max
                }
            })
        
        # Combine queries with AND operator
        if len(queries) == 1:
            query = queries[0]
        else:
            query = {
                "type": "group",
                "logical_operator": "and",
                "nodes": queries
            }
        
        search_request = {
            "query": query,
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "rows": rows,
                    "start": 0
                }
            }
        }
        
        logger.info(f"Searching for structures from {organism}")
        response = self.session.post(SEARCH_ENDPOINT, json=search_request, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def search_chemical_similarity(
        self,
        smiles: str,
        similarity_threshold: float = 0.7,
        rows: int = 25
    ) -> Dict[str, Any]:
        """
        Search for structures containing chemically similar compounds.
        
        Args:
            smiles: SMILES string representation of the compound
            similarity_threshold: Chemical similarity threshold (0.0 to 1.0)
            rows: Number of results to return
            
        Returns:
            Search results dictionary
        """
        search_request = {
            "query": {
                "type": "terminal",
                "service": "chemical",
                "parameters": {
                    "value": smiles,
                    "type": "descriptor",
                    "descriptor_type": "SMILES",
                    "match_type": "graph-relaxed",
                    "match_subset": False
                }
            },
            "return_type": "non_polymer_entity",
            "request_options": {
                "paginate": {
                    "rows": rows,
                    "start": 0
                }
            }
        }
        
        logger.info(f"Searching for compounds similar to SMILES: {smiles}")
        response = self.session.post(SEARCH_ENDPOINT, json=search_request, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_structure_data(self, pdb_id: str) -> Dict[str, Any]:
        """
        Get detailed data for a specific PDB entry.
        
        Args:
            pdb_id: PDB identifier (e.g., "4HHB")
            
        Returns:
            Structure data dictionary
        """
        url = f"{DATA_ENDPOINT}/rest/v1/core/entry/{pdb_id.upper()}"
        
        logger.info(f"Fetching data for PDB entry: {pdb_id}")
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def search_membrane_proteins(self, rows: int = 50) -> Dict[str, Any]:
        """
        Search for membrane proteins in the PDB.
        
        Args:
            rows: Number of results to return
            
        Returns:
            Search results dictionary
        """
        search_request = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.membrane_entity",
                    "operator": "exact_match",
                    "value": "Y"
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "rows": rows,
                    "start": 0
                },
                "sort": [
                    {
                        "sort_by": "rcsb_entry_info.resolution_combined",
                        "direction": "asc"
                    }
                ]
            }
        }
        
        logger.info("Searching for membrane proteins")
        response = self.session.post(SEARCH_ENDPOINT, json=search_request, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def search_recent_structures(self, days: int = 7, rows: int = 100) -> Dict[str, Any]:
        """
        Get recently released structures.
        
        Args:
            days: Number of days to look back
            rows: Number of results to return
            
        Returns:
            Search results dictionary
        """
        from datetime import datetime, timedelta
        
        # Calculate the date threshold
        date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        search_request = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "greater_or_equal",
                    "value": date_threshold
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "rows": rows,
                    "start": 0
                },
                "sort": [
                    {
                        "sort_by": "rcsb_accession_info.initial_release_date",
                        "direction": "desc"
                    }
                ]
            }
        }
        
        logger.info(f"Searching for structures released since {date_threshold}")
        response = self.session.post(SEARCH_ENDPOINT, json=search_request, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of RCSB PDB client."""
    client = RCSBPDBClient()
    
    print("\n" + "="*60)
    print("RCSB PDB API Integration Demo for Quark")
    print("="*60)
    
    # Example 1: Text search for COVID-19 structures
    print("\n1. Searching for COVID-19 spike protein structures...")
    covid_results = client.search_text("COVID-19 spike protein", rows=5)
    print(f"Found {covid_results.get('total_count', 0)} total results")
    if covid_results.get('result_set'):
        print("First 5 PDB IDs:")
        for entry in covid_results['result_set'][:5]:
            print(f"  - {entry['identifier']}")
    
    # Example 2: Search for recent structures
    print("\n2. Getting structures released in the last 7 days...")
    recent_results = client.search_recent_structures(days=7, rows=5)
    print(f"Found {recent_results.get('total_count', 0)} recent structures")
    if recent_results.get('result_set'):
        print("Recent PDB IDs:")
        for entry in recent_results['result_set'][:5]:
            print(f"  - {entry['identifier']}")
    
    # Example 3: Search for human protein structures
    print("\n3. Searching for high-resolution human protein structures...")
    human_results = client.search_by_organism(
        "Homo sapiens",
        experimental_method="X-RAY DIFFRACTION",
        resolution_max=2.0,
        rows=5
    )
    print(f"Found {human_results.get('total_count', 0)} human protein structures")
    if human_results.get('result_set'):
        print("High-resolution human protein PDB IDs:")
        for entry in human_results['result_set'][:5]:
            print(f"  - {entry['identifier']}")
    
    # Example 4: Get detailed data for a specific structure
    print("\n4. Getting detailed data for hemoglobin (4HHB)...")
    try:
        hemo_data = client.get_structure_data("4HHB")
        if hemo_data:
            entry_info = hemo_data.get('rcsb_entry_info', {})
            print(f"  Title: {hemo_data.get('struct', {}).get('title', 'N/A')}")
            print(f"  Resolution: {entry_info.get('resolution_combined', ['N/A'])[0]} Å")
            print(f"  Experimental Method: {entry_info.get('experimental_method', 'N/A')}")
            print(f"  Release Date: {hemo_data.get('rcsb_accession_info', {}).get('initial_release_date', 'N/A')}")
    except Exception as e:
        print(f"  Error fetching data: {e}")
    
    print("\n" + "="*60)
    print("Demo complete! RCSB PDB API is ready for use in Quark.")
    print("="*60)


if __name__ == "__main__":
    main()
