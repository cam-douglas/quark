#!/usr/bin/env python3
"""
Ensembl REST API Integration for Quark
=======================================
This module provides integration with the Ensembl REST API for genomic data access.

Ensembl provides comprehensive genomic data including genes, transcripts, proteins,
variations, regulatory features, and comparative genomics across multiple species.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    ENSEMBL_CONFIG = credentials['services']['ensembl']

# API endpoints
BASE_URL = ENSEMBL_CONFIG['endpoints']['base']


class EnsemblClient:
    """Client for interacting with Ensembl REST API."""
    
    def __init__(self, server: str = BASE_URL):
        """
        Initialize Ensembl client.
        
        Args:
            server: Ensembl server URL (default: main REST server)
        """
        self.server = server
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Rate limiting (15 requests per second max)
        self.rate_limit = 15
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a GET request to the Ensembl API.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.server}{endpoint}"
        
        # Enforce rate limiting
        self._rate_limit()
        
        logger.debug(f"GET {url}")
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            logger.warning(f"Resource not found: {url}")
            return {}
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
            response.raise_for_status()
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the Ensembl API.
        
        Args:
            endpoint: API endpoint path
            data: JSON data to post
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.server}{endpoint}"
        
        # Enforce rate limiting
        self._rate_limit()
        
        logger.debug(f"POST {url}")
        response = self.session.post(url, json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
            response.raise_for_status()
    
    def lookup_by_id(self, stable_id: str, expand: bool = False) -> Dict[str, Any]:
        """
        Find a gene, transcript, or protein by stable ID.
        
        Args:
            stable_id: Ensembl stable ID (e.g., ENSG00000139618 for BRCA2)
            expand: Whether to expand nested objects
            
        Returns:
            Feature information
        """
        endpoint = f"/lookup/id/{stable_id}"
        params = {'expand': 1} if expand else None
        
        logger.info(f"Looking up ID: {stable_id}")
        return self._get(endpoint, params)
    
    def lookup_by_symbol(
        self,
        species: str,
        symbol: str,
        expand: bool = False
    ) -> Dict[str, Any]:
        """
        Find a gene by symbol.
        
        Args:
            species: Species name (e.g., 'homo_sapiens', 'mus_musculus')
            symbol: Gene symbol (e.g., 'BRCA2', 'TP53')
            expand: Whether to expand nested objects
            
        Returns:
            Gene information
        """
        endpoint = f"/lookup/symbol/{species}/{symbol}"
        params = {'expand': 1} if expand else None
        
        logger.info(f"Looking up symbol: {symbol} in {species}")
        return self._get(endpoint, params)
    
    def get_sequence(
        self,
        stable_id: str,
        object_type: str = 'cds',
        format_type: str = 'fasta'
    ) -> str:
        """
        Get sequence for a gene, transcript, or protein.
        
        Args:
            stable_id: Ensembl stable ID
            object_type: Type of sequence ('genomic', 'cds', 'cdna', 'protein')
            format_type: Output format ('fasta' or 'text')
            
        Returns:
            Sequence string
        """
        endpoint = f"/sequence/id/{stable_id}"
        params = {'type': object_type}
        
        # Change content type for FASTA
        old_accept = self.session.headers['Accept']
        if format_type == 'fasta':
            self.session.headers['Accept'] = 'text/x-fasta'
        
        logger.info(f"Getting {object_type} sequence for {stable_id}")
        url = f"{self.server}{endpoint}"
        
        self._rate_limit()
        response = self.session.get(url, params=params)
        
        # Restore original accept header
        self.session.headers['Accept'] = old_accept
        
        if response.status_code == 200:
            return response.text
        else:
            logger.error(f"Error getting sequence: {response.status_code}")
            return ""
    
    def get_homology(
        self,
        species: str,
        gene_id: str,
        target_species: Optional[str] = None,
        homology_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get homology (ortholog/paralog) information.
        
        Args:
            species: Source species
            gene_id: Gene stable ID
            target_species: Optional target species filter
            homology_type: Type filter ('ortholog', 'paralog', etc.)
            
        Returns:
            List of homology relationships
        """
        endpoint = f"/homology/id/{species}/{gene_id}"
        params = {}
        
        if target_species:
            params['target_species'] = target_species
        if homology_type:
            params['type'] = homology_type
            
        logger.info(f"Getting homology for {gene_id}")
        result = self._get(endpoint, params)
        
        if 'data' in result and result['data']:
            return result['data'][0].get('homologies', [])
        return []
    
    def get_variants_in_region(
        self,
        species: str,
        region: str,
        feature_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get variants in a genomic region.
        
        Args:
            species: Species name
            region: Genomic region (e.g., '7:140424943-140624564')
            feature_types: Feature types to include
            
        Returns:
            List of variants
        """
        endpoint = f"/overlap/region/{species}/{region}"
        params = {'feature': feature_types} if feature_types else {'feature': 'variation'}
        
        logger.info(f"Getting variants in region {region}")
        return self._get(endpoint, params)
    
    def get_variant_by_id(self, species: str, variant_id: str) -> Dict[str, Any]:
        """
        Get variant information by ID.
        
        Args:
            species: Species name
            variant_id: Variant ID (e.g., rs699)
            
        Returns:
            Variant information
        """
        endpoint = f"/variation/{species}/{variant_id}"
        
        logger.info(f"Getting variant {variant_id}")
        return self._get(endpoint)
    
    def get_xrefs(
        self,
        stable_id: str,
        external_db: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get cross-references to external databases.
        
        Args:
            stable_id: Ensembl stable ID
            external_db: Optional external database filter
            
        Returns:
            List of cross-references
        """
        endpoint = f"/xrefs/id/{stable_id}"
        params = {'external_db': external_db} if external_db else None
        
        logger.info(f"Getting cross-references for {stable_id}")
        return self._get(endpoint, params)
    
    def get_species_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available species.
        
        Returns:
            List of species information
        """
        endpoint = "/info/species"
        
        logger.info("Getting species information")
        result = self._get(endpoint)
        return result.get('species', [])
    
    def search_brain_genes(self, species: str = "homo_sapiens") -> Dict[str, Dict[str, Any]]:
        """
        Search for key brain-related genes.
        
        Args:
            species: Species to search in
            
        Returns:
            Dictionary of brain genes and their information
        """
        # Key brain-related genes
        brain_genes = {
            # Neurotransmitter receptors
            "DRD2": "Dopamine receptor D2",
            "HTR2A": "Serotonin receptor 2A",
            "GRIN1": "NMDA receptor subunit 1",
            "GABRA1": "GABA receptor alpha 1",
            
            # Synaptic proteins
            "SYN1": "Synapsin I",
            "SYT1": "Synaptotagmin 1",
            "SNAP25": "Synaptosomal-associated protein 25",
            
            # Neurodegenerative disease genes
            "APP": "Amyloid precursor protein",
            "MAPT": "Tau protein",
            "SNCA": "Alpha-synuclein",
            "HTT": "Huntingtin",
            
            # Ion channels
            "SCN1A": "Sodium channel alpha 1",
            "KCNQ2": "Potassium channel Q2",
            "CACNA1A": "Calcium channel alpha 1A",
            
            # Neurodevelopment
            "FOXP2": "Forkhead box P2",
            "MECP2": "Methyl CpG binding protein 2",
            "FMR1": "Fragile X mental retardation 1"
        }
        
        results = {}
        
        print(f"\nSearching for brain genes in {species}")
        print("-" * 50)
        
        for symbol, description in brain_genes.items():
            try:
                gene_info = self.lookup_by_symbol(species, symbol, expand=True)
                if gene_info:
                    results[symbol] = {
                        'ensembl_id': gene_info.get('id'),
                        'description': description,
                        'biotype': gene_info.get('biotype'),
                        'chromosome': gene_info.get('seq_region_name'),
                        'start': gene_info.get('start'),
                        'end': gene_info.get('end'),
                        'strand': gene_info.get('strand'),
                        'transcript_count': len(gene_info.get('Transcript', []))
                    }
                    print(f"✓ {symbol}: {gene_info.get('id')}")
                else:
                    print(f"✗ {symbol}: Not found")
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
                
        return results


def demonstrate_comparative_genomics(client: EnsemblClient):
    """
    Demonstrate comparative genomics features.
    
    Args:
        client: Ensembl client instance
    """
    print("\nComparative Genomics Demo")
    print("=" * 60)
    
    # Find mouse orthologs of human brain genes
    human_genes = {
        "ENSG00000145675": "GABRB1",  # GABA receptor
        "ENSG00000102003": "SYP",      # Synaptophysin
        "ENSG00000142192": "APP"       # Amyloid precursor
    }
    
    orthologs = {}
    
    for gene_id, symbol in human_genes.items():
        print(f"\nFinding mouse orthologs for {symbol} ({gene_id})")
        
        try:
            homologs = client.get_homology(
                'homo_sapiens',
                gene_id,
                target_species='mus_musculus',
                homology_type='orthologues'
            )
            
            if homologs:
                for h in homologs[:1]:  # Just first ortholog
                    target = h.get('target', {})
                    orthologs[symbol] = {
                        'human_id': gene_id,
                        'mouse_id': target.get('id'),
                        'mouse_symbol': target.get('label'),
                        'identity': target.get('perc_id'),
                        'coverage': target.get('perc_cov')
                    }
                    print(f"  Mouse ortholog: {target.get('label')} ({target.get('id')})")
                    print(f"  Identity: {target.get('perc_id')}%")
            else:
                print("  No orthologs found")
                
        except Exception as e:
            print(f"  Error: {e}")
            
    return orthologs


def main():
    """Example usage of Ensembl client."""
    client = EnsemblClient()
    
    print("=" * 60)
    print("Ensembl REST API Integration Test")
    print("Quark System - Genomic Data Access")
    print("=" * 60)
    
    # Test 1: Lookup by symbol
    print("\n1. Testing gene lookup by symbol...")
    try:
        brca2 = client.lookup_by_symbol('homo_sapiens', 'BRCA2')
        if brca2:
            print(f"  Gene: {brca2.get('display_name')}")
            print(f"  ID: {brca2.get('id')}")
            print(f"  Location: chr{brca2.get('seq_region_name')}:{brca2.get('start')}-{brca2.get('end')}")
            print(f"  Description: {brca2.get('description', 'N/A')[:60]}...")
            print("  ✓ Gene lookup successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Get sequence
    print("\n2. Testing sequence retrieval...")
    try:
        # Get protein sequence for a small transcript
        sequence = client.get_sequence('ENST00000646891', 'protein', 'fasta')
        if sequence:
            # Handle JSON response
            if sequence.startswith('{'):
                seq_data = json.loads(sequence)
                seq = seq_data.get('seq', '')
                print(f"  ID: {seq_data.get('id')}")
                print(f"  Sequence length: {len(seq)} amino acids")
                print(f"  First 50 AA: {seq[:50]}...")
            else:
                # Handle FASTA format
                lines = sequence.strip().split('\n')
                print(f"  Header: {lines[0][:70]}...")
                seq = ''.join(lines[1:])
                print(f"  Sequence length: {len(seq)} amino acids")
                print(f"  First 50 AA: {seq[:50]}...")
            print("  ✓ Sequence retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Get variant information
    print("\n3. Testing variant lookup...")
    try:
        # Famous APOE variant associated with Alzheimer's
        variant = client.get_variant_by_id('homo_sapiens', 'rs429358')
        if variant:
            print(f"  Variant: {variant.get('name')}")
            print(f"  MAF: {variant.get('MAF')}")
            print(f"  Consequence: {variant.get('most_severe_consequence')}")
            print("  ✓ Variant lookup successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Get cross-references
    print("\n4. Testing cross-references...")
    try:
        xrefs = client.get_xrefs('ENSG00000139618')  # BRCA2
        if xrefs:
            print(f"  Found {len(xrefs)} cross-references")
            # Show a few examples
            dbs = set()
            for xref in xrefs[:10]:
                dbs.add(xref.get('dbname'))
            print(f"  External databases: {', '.join(list(dbs)[:5])}")
            print("  ✓ Cross-references successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Search for brain genes
    print("\n5. Searching for brain-related genes...")
    try:
        brain_genes = client.search_brain_genes('homo_sapiens')
        
        # Count by category
        found = len([g for g in brain_genes.values() if g.get('ensembl_id')])
        print(f"\nFound {found}/{len(brain_genes)} brain genes")
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "knowledge" / "ensembl_brain_genes.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'source': 'Ensembl REST API',
                    'date': '2025-01-20',
                    'species': 'homo_sapiens',
                    'description': 'Brain-related genes from Ensembl'
                },
                'genes': brain_genes
            }, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        print("  ✓ Brain gene search successful")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 6: Comparative genomics
    orthologs = demonstrate_comparative_genomics(client)
    
    print("\n" + "=" * 60)
    print("Ensembl REST API integration test complete!")
    print("✓ All basic functions tested")
    print("=" * 60)


if __name__ == "__main__":
    main()
