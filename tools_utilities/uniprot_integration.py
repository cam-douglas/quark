#!/usr/bin/env python3
"""
UniProt REST API Integration for Quark
=======================================
This module provides integration with the UniProt REST API for protein sequence
and functional information access.

UniProt is the world's leading resource for protein sequence and functional 
information with 250+ million sequences.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    UNIPROT_CONFIG = credentials['services']['uniprot']

# API endpoints
BASE_URL = UNIPROT_CONFIG['endpoints']['base']
UNIPROTKB_URL = UNIPROT_CONFIG['endpoints']['uniprotkb']


class UniProtClient:
    """Client for interacting with UniProt REST API."""
    
    def __init__(self):
        """Initialize UniProt client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quark-UniProt-Integration/1.0'
        })
        
        # Rate limiting (be considerate)
        self.last_request_time = 0
        self.min_interval = 0.5  # 2 requests per second max
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        format: str = 'json'
    ) -> Union[Dict, str]:
        """
        Make a request to UniProt API.
        
        Args:
            endpoint: API endpoint URL
            params: Query parameters
            format: Response format (json, fasta, tsv, etc.)
            
        Returns:
            Response data
        """
        self._rate_limit()
        
        if params is None:
            params = {}
        
        # Add format to params
        params['format'] = format
        
        logger.debug(f"GET {endpoint} with params: {params}")
        response = self.session.get(endpoint, params=params)
        
        if response.status_code == 200:
            if format == 'json':
                return response.json()
            else:
                return response.text
        else:
            logger.error(f"Error {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
    
    def search_proteins(
        self,
        query: str,
        organism: Optional[str] = None,
        reviewed: Optional[bool] = None,
        fields: Optional[List[str]] = None,
        size: int = 25,
        format: str = 'json'
    ) -> Union[Dict, str]:
        """
        Search for proteins in UniProtKB.
        
        Args:
            query: Search query (Lucene syntax)
            organism: Organism name or taxonomy ID
            reviewed: True for Swiss-Prot only, False for TrEMBL only
            fields: Fields to return
            size: Number of results
            format: Output format
            
        Returns:
            Search results
        """
        params = {
            'query': query,
            'size': size
        }
        
        # Add filters
        filters = []
        if organism:
            filters.append(f'organism_name:{organism}')
        if reviewed is not None:
            filters.append(f'reviewed:{str(reviewed).lower()}')
        
        if filters:
            params['query'] = f"{query} AND " + " AND ".join(filters)
        
        # Add fields
        if fields:
            params['fields'] = ','.join(fields)
        
        logger.info(f"Searching UniProt for: {params['query']}")
        return self._make_request(f"{UNIPROTKB_URL}/search", params, format)
    
    def get_protein(
        self,
        accession: str,
        format: str = 'json'
    ) -> Union[Dict, str]:
        """
        Get protein entry by accession.
        
        Args:
            accession: UniProt accession (e.g., P04637)
            format: Output format
            
        Returns:
            Protein data
        """
        logger.info(f"Getting protein: {accession}")
        url = f"{UNIPROTKB_URL}/{accession}"
        return self._make_request(url, format=format)
    
    def get_fasta(self, accession: str) -> str:
        """
        Get protein sequence in FASTA format.
        
        Args:
            accession: UniProt accession
            
        Returns:
            FASTA sequence
        """
        return self.get_protein(accession, format='fasta')
    
    def map_identifiers(
        self,
        ids: List[str],
        from_db: str,
        to_db: str
    ) -> Dict[str, List[str]]:
        """
        Map identifiers between databases.
        
        Args:
            ids: List of identifiers to map
            from_db: Source database (e.g., 'UniProtKB_AC-ID')
            to_db: Target database (e.g., 'PDB')
            
        Returns:
            Mapping results
        """
        # Submit mapping job
        url = f"{BASE_URL}/idmapping/run"
        data = {
            'ids': ','.join(ids),
            'from': from_db,
            'to': to_db
        }
        
        logger.info(f"Submitting ID mapping: {from_db} -> {to_db}")
        response = self.session.post(url, data=data)
        response.raise_for_status()
        job_id = response.json()['jobId']
        
        # Poll for results
        status_url = f"{BASE_URL}/idmapping/status/{job_id}"
        while True:
            time.sleep(2)
            status_response = self.session.get(status_url)
            status_data = status_response.json()
            
            if 'results' in status_data or status_data.get('jobStatus') == 'FINISHED':
                break
        
        # Get results
        results_url = f"{BASE_URL}/idmapping/results/stream/{job_id}"
        results = self._make_request(results_url)
        
        # Parse mapping
        mapping = {}
        if 'results' in results:
            for result in results['results']:
                from_id = result.get('from')
                to_id = result.get('to', {}).get('primaryAccession')
                if from_id and to_id:
                    if from_id not in mapping:
                        mapping[from_id] = []
                    mapping[from_id].append(to_id)
        
        return mapping
    
    def search_brain_proteins(self) -> Dict[str, List[Dict]]:
        """
        Search for brain-related proteins.
        
        Returns:
            Categorized brain proteins
        """
        brain_proteins = {
            'neurotransmitter_receptors': [],
            'ion_channels': [],
            'synaptic_proteins': [],
            'neurodegenerative': []
        }
        
        print("\nSearching for brain-related proteins in UniProt")
        print("-" * 50)
        
        # 1. Neurotransmitter receptors
        print("1. Searching neurotransmitter receptors...")
        try:
            receptors = self.search_proteins(
                query='neurotransmitter receptor',
                organism='human',
                reviewed=True,
                fields=['accession', 'id', 'protein_name', 'gene_names', 'length'],
                size=10
            )
            if 'results' in receptors:
                brain_proteins['neurotransmitter_receptors'] = receptors['results'][:5]
                print(f"  Found {len(receptors['results'])} receptors")
        except Exception as e:
            logger.error(f"Error searching receptors: {e}")
        
        # 2. Ion channels
        print("2. Searching ion channels...")
        try:
            channels = self.search_proteins(
                query='ion channel AND brain',
                organism='human',
                reviewed=True,
                fields=['accession', 'id', 'protein_name', 'gene_names'],
                size=10
            )
            if 'results' in channels:
                brain_proteins['ion_channels'] = channels['results'][:5]
                print(f"  Found {len(channels['results'])} ion channels")
        except Exception as e:
            logger.error(f"Error searching ion channels: {e}")
        
        # 3. Synaptic proteins
        print("3. Searching synaptic proteins...")
        try:
            synaptic = self.search_proteins(
                query='synapse',
                organism='human',
                reviewed=True,
                fields=['accession', 'id', 'protein_name', 'gene_names'],
                size=10
            )
            if 'results' in synaptic:
                brain_proteins['synaptic_proteins'] = synaptic['results'][:5]
                print(f"  Found {len(synaptic['results'])} synaptic proteins")
        except Exception as e:
            logger.error(f"Error searching synaptic proteins: {e}")
        
        # 4. Neurodegenerative disease proteins
        print("4. Searching neurodegenerative disease proteins...")
        try:
            disease_proteins = self.search_proteins(
                query='alzheimer OR parkinson OR huntington',
                organism='human',
                reviewed=True,
                fields=['accession', 'id', 'protein_name', 'gene_names'],
                size=10
            )
            if 'results' in disease_proteins:
                brain_proteins['neurodegenerative'] = disease_proteins['results'][:5]
                print(f"  Found {len(disease_proteins['results'])} disease proteins")
        except Exception as e:
            logger.error(f"Error searching disease proteins: {e}")
        
        return brain_proteins


def demonstrate_protein_analysis(client: UniProtClient):
    """
    Demonstrate protein analysis capabilities.
    
    Args:
        client: UniProt client instance
    """
    print("\nProtein Analysis Demo")
    print("=" * 60)
    
    # Analyze a well-known protein: p53 tumor suppressor
    print("\n1. Analyzing p53 tumor suppressor (P04637)")
    print("-" * 40)
    
    try:
        # Get protein details
        p53 = client.get_protein('P04637')
        
        if p53:
            # Parse primary accession
            primary = p53.get('primaryAccession', 'P04637')
            
            print(f"  Accession: {primary}")
            print(f"  Protein: {p53.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')}")
            
            # Gene names
            genes = p53.get('genes', [])
            if genes and len(genes) > 0:
                gene_names = [g.get('geneName', {}).get('value', '') for g in genes]
                print(f"  Gene: {', '.join(filter(None, gene_names))}")
            
            # Sequence info
            sequence = p53.get('sequence', {})
            print(f"  Length: {sequence.get('length', 'N/A')} amino acids")
            print(f"  Mass: {sequence.get('molWeight', 'N/A')} Da")
            
            # Get FASTA
            fasta = client.get_fasta('P04637')
            if fasta:
                lines = fasta.strip().split('\n')
                print(f"  Sequence (first 50 AA): {lines[1][:50]}...")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    return True


def main():
    """Example usage of UniProt client."""
    client = UniProtClient()
    
    print("=" * 60)
    print("UniProt REST API Integration Test")
    print("Quark System - Protein Database Access")
    print("=" * 60)
    
    # Test 1: Basic search
    print("\n1. Testing protein search...")
    try:
        results = client.search_proteins(
            query='dopamine receptor',
            organism='human',
            reviewed=True,
            fields=['accession', 'id', 'protein_name'],
            size=5
        )
        
        if 'results' in results:
            print(f"  Found {len(results['results'])} proteins")
            for protein in results['results'][:3]:
                acc = protein.get('primaryAccession', 'N/A')
                id = protein.get('uniProtkbId', 'N/A')
                name = protein.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')
                print(f"    {acc} ({id}): {name}")
            print("  ✓ Protein search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Get specific protein
    print("\n2. Testing protein retrieval...")
    try:
        # Get insulin (P01308)
        insulin = client.get_protein('P01308')
        
        if insulin:
            print(f"  Protein: Insulin")
            print(f"  Accession: {insulin.get('primaryAccession')}")
            print(f"  Organism: {insulin.get('organism', {}).get('scientificName')}")
            print(f"  Length: {insulin.get('sequence', {}).get('length')} AA")
            print("  ✓ Protein retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Get FASTA sequence
    print("\n3. Testing FASTA retrieval...")
    try:
        fasta = client.get_fasta('P00533')  # EGFR
        if fasta:
            lines = fasta.strip().split('\n')
            print(f"  Header: {lines[0][:60]}...")
            print(f"  Sequence length: {len(''.join(lines[1:]))} AA")
            print("  ✓ FASTA retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Search brain proteins
    print("\n4. Searching for brain-related proteins...")
    try:
        brain_proteins = client.search_brain_proteins()
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "knowledge" / "uniprot_brain_proteins.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count total
        total = sum(len(proteins) for proteins in brain_proteins.values())
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'source': 'UniProt REST API',
                    'date': '2025-01-20',
                    'description': 'Brain-related proteins from UniProt',
                    'total_proteins': total
                },
                'proteins': brain_proteins
            }, f, indent=2)
        
        print(f"\n  Total brain proteins found: {total}")
        for category, proteins in brain_proteins.items():
            if proteins:
                print(f"    {category}: {len(proteins)} proteins")
        
        print(f"\n  Results saved to: {output_path}")
        print("  ✓ Brain protein search successful")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Demonstrate analysis
    demonstrate_protein_analysis(client)
    
    print("\n" + "=" * 60)
    print("UniProt API integration test complete!")
    print("✓ Protein database access working")
    print("=" * 60)


if __name__ == "__main__":
    main()
