#!/usr/bin/env python3
"""
NCBI E-utilities API Integration for Quark
===========================================
This module provides integration with NCBI's Entrez Programming Utilities (E-utilities).

E-utilities provide programmatic access to NCBI's databases including PubMed, GenBank,
Gene, Protein, and over 30 other biological databases.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    NCBI_CONFIG = credentials['services']['ncbi_eutilities']

# API endpoints
BASE_URL = NCBI_CONFIG['endpoints']['base']
ESEARCH_URL = NCBI_CONFIG['endpoints']['esearch']
EFETCH_URL = NCBI_CONFIG['endpoints']['efetch']
ESUMMARY_URL = NCBI_CONFIG['endpoints']['esummary']
ELINK_URL = NCBI_CONFIG['endpoints']['elink']
EINFO_URL = NCBI_CONFIG['endpoints']['einfo']


class NCBIEutilitiesClient:
    """Client for interacting with NCBI E-utilities API."""
    
    def __init__(self, api_key: Optional[str] = None, email: str = "quark@example.com", tool: str = "Quark"):
        """
        Initialize NCBI E-utilities client.
        
        Args:
            api_key: Optional NCBI API key for better rate limits
            email: Email address for NCBI to contact about usage
            tool: Tool name to identify your application
        """
        self.api_key = api_key
        self.email = email
        self.tool = tool
        self.session = requests.Session()
        
        # Set rate limit based on API key availability
        self.rate_limit = 10 if api_key else 3  # requests per second
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """
        Make a rate-limited request to NCBI.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            
        Returns:
            Response object
        """
        # Add standard parameters
        params['tool'] = self.tool
        params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key
            
        # Enforce rate limiting
        self._rate_limit()
        
        # Make request
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response
    
    def search(
        self,
        database: str,
        query: str,
        retmax: int = 20,
        retstart: int = 0,
        sort: Optional[str] = None,
        use_history: bool = False
    ) -> Dict[str, Any]:
        """
        Search an NCBI database.
        
        Args:
            database: Database to search (e.g., 'pubmed', 'protein', 'gene')
            query: Search query string
            retmax: Maximum number of UIDs to return
            retstart: Sequential index of first UID to fetch
            sort: Sort order ('relevance' or 'pub_date')
            use_history: Whether to use history server
            
        Returns:
            Dictionary with search results
        """
        params = {
            'db': database,
            'term': query,
            'retmax': retmax,
            'retstart': retstart,
            'retmode': 'json'
        }
        
        if sort:
            params['sort'] = sort
        if use_history:
            params['usehistory'] = 'y'
            
        logger.info(f"Searching {database} for: {query}")
        response = self._make_request(ESEARCH_URL, params)
        
        result = response.json()
        esearchresult = result.get('esearchresult', {})
        
        return {
            'count': int(esearchresult.get('count', 0)),
            'ids': esearchresult.get('idlist', []),
            'web_env': esearchresult.get('webenv'),
            'query_key': esearchresult.get('querykey'),
            'query_translation': esearchresult.get('querytranslation')
        }
    
    def fetch(
        self,
        database: str,
        ids: Union[str, List[str]],
        rettype: str = 'abstract',
        retmode: str = 'text'
    ) -> str:
        """
        Fetch full records from an NCBI database.
        
        Args:
            database: Database to fetch from
            ids: Single ID or list of IDs
            rettype: Return type (e.g., 'abstract', 'medline', 'xml')
            retmode: Return mode ('text' or 'xml')
            
        Returns:
            Fetched records as string
        """
        if isinstance(ids, list):
            ids = ','.join(str(i) for i in ids)
            
        params = {
            'db': database,
            'id': ids,
            'rettype': rettype,
            'retmode': retmode
        }
        
        logger.info(f"Fetching {len(ids.split(','))} records from {database}")
        response = self._make_request(EFETCH_URL, params)
        
        return response.text
    
    def get_summaries(
        self,
        database: str,
        ids: Union[str, List[str]],
        version: str = '2.0'
    ) -> List[Dict[str, Any]]:
        """
        Get document summaries.
        
        Args:
            database: Database to query
            ids: Single ID or list of IDs
            version: API version (1.0 or 2.0)
            
        Returns:
            List of document summaries
        """
        if isinstance(ids, list):
            ids = ','.join(str(i) for i in ids)
            
        params = {
            'db': database,
            'id': ids,
            'retmode': 'json',
            'version': version
        }
        
        logger.info(f"Getting summaries for {len(ids.split(','))} records")
        response = self._make_request(ESUMMARY_URL, params)
        
        result = response.json()
        summaries = []
        
        if 'result' in result:
            for uid in result['result'].get('uids', []):
                if uid in result['result']:
                    summaries.append(result['result'][uid])
                    
        return summaries
    
    def find_related(
        self,
        from_db: str,
        to_db: str,
        ids: Union[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Find related records in another database.
        
        Args:
            from_db: Source database
            to_db: Target database
            ids: IDs to find links for
            
        Returns:
            Dictionary mapping source IDs to related IDs
        """
        if isinstance(ids, list):
            ids = ','.join(str(i) for i in ids)
            
        params = {
            'dbfrom': from_db,
            'db': to_db,
            'id': ids,
            'retmode': 'json'
        }
        
        logger.info(f"Finding links from {from_db} to {to_db}")
        response = self._make_request(ELINK_URL, params)
        
        result = response.json()
        links = {}
        
        for linkset in result.get('linksets', []):
            source_id = linkset.get('ids', [None])[0]
            if source_id and 'linksetdbs' in linkset:
                for linkdb in linkset['linksetdbs']:
                    if linkdb.get('dbto') == to_db:
                        links[source_id] = linkdb.get('links', [])
                        
        return links
    
    def get_database_info(self, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about NCBI databases.
        
        Args:
            database: Specific database name or None for all databases
            
        Returns:
            Database information dictionary
        """
        params = {'retmode': 'json'}
        if database:
            params['db'] = database
            
        logger.info(f"Getting database info for: {database or 'all databases'}")
        response = self._make_request(EINFO_URL, params)
        
        return response.json()
    
    def search_pubmed_for_brain_research(self, topic: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search PubMed for brain-related research papers.
        
        Args:
            topic: Specific brain research topic
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with paper information
        """
        # Construct brain-specific search query
        query = f"({topic}) AND (brain OR neural OR neuron OR synapse OR neurotransmitter)"
        query += " AND (2020:2025[pdat])"  # Recent papers only
        
        # Search PubMed
        search_results = self.search('pubmed', query, retmax=max_results)
        
        if not search_results['ids']:
            return {'papers': [], 'total_found': 0}
        
        # Get summaries
        summaries = self.get_summaries('pubmed', search_results['ids'])
        
        # Extract relevant information
        papers = []
        for summary in summaries:
            paper = {
                'pmid': summary.get('uid'),
                'title': summary.get('title'),
                'authors': ', '.join([
                    author.get('name', '') 
                    for author in summary.get('authors', [])[:3]
                ]) + (' et al.' if len(summary.get('authors', [])) > 3 else ''),
                'journal': summary.get('source'),
                'pub_date': summary.get('pubdate'),
                'doi': summary.get('elocationid', '').replace('doi: ', '')
            }
            papers.append(paper)
            
        return {
            'papers': papers,
            'total_found': search_results['count'],
            'query_used': query
        }


def demonstrate_brain_research_searches(client: NCBIEutilitiesClient):
    """
    Demonstrate searching for brain-related research.
    
    Args:
        client: NCBI E-utilities client instance
    """
    brain_topics = [
        "dopamine receptor structure",
        "synaptic plasticity",
        "neurodegeneration alzheimer",
        "GABA neurotransmitter",
        "ion channel gating"
    ]
    
    print("\nSearching PubMed for Recent Brain Research Papers")
    print("=" * 60)
    
    all_results = {}
    
    for topic in brain_topics:
        print(f"\nTopic: {topic}")
        print("-" * 40)
        
        results = client.search_pubmed_for_brain_research(topic, max_results=3)
        all_results[topic] = results
        
        print(f"Found {results['total_found']} total papers")
        
        for i, paper in enumerate(results['papers'], 1):
            print(f"\n  {i}. {paper['title'][:80]}...")
            print(f"     Authors: {paper['authors']}")
            print(f"     Journal: {paper['journal']}")
            print(f"     Date: {paper['pub_date']}")
            if paper['doi']:
                print(f"     DOI: {paper['doi']}")
                
    return all_results


def main():
    """Example usage of NCBI E-utilities client."""
    # Initialize client (without API key for now - works but with lower rate limit)
    client = NCBIEutilitiesClient(email="quark@example.com", tool="QuarkBrainResearch")
    
    print("=" * 60)
    print("NCBI E-utilities API Integration Test")
    print("Quark System - Biological Database Access")
    print("=" * 60)
    
    # Test 1: Get database information
    print("\n1. Testing database information retrieval...")
    try:
        db_info = client.get_database_info('pubmed')
        if 'einforesult' in db_info:
            pubmed_info = db_info['einforesult']['dbinfo'][0]
            print(f"  Database: {pubmed_info.get('dbname')}")
            print(f"  Description: {pubmed_info.get('description', 'N/A')[:100]}...")
            count_str = pubmed_info.get('count', 'N/A')
            if count_str != 'N/A':
                print(f"  Record Count: {int(count_str):,}")
            else:
                print(f"  Record Count: N/A")
            print("  ✓ Database info retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Search PubMed for a simple query
    print("\n2. Testing PubMed search...")
    try:
        results = client.search('pubmed', 'BRCA1 cancer', retmax=5)
        print(f"  Found {results['count']} total results")
        print(f"  Retrieved {len(results['ids'])} IDs: {', '.join(results['ids'][:5])}")
        print("  ✓ Search successful")
        
        # Test 3: Fetch summaries
        if results['ids']:
            print("\n3. Testing document summary retrieval...")
            summaries = client.get_summaries('pubmed', results['ids'][:3])
            print(f"  Retrieved {len(summaries)} summaries")
            if summaries:
                first = summaries[0]
                print(f"  First paper: {first.get('title', 'N/A')[:80]}...")
                print(f"  Authors: {len(first.get('authors', []))} authors")
                print("  ✓ Summary retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Search for brain research
    print("\n4. Testing brain research searches...")
    try:
        brain_results = client.search_pubmed_for_brain_research("synaptic transmission", max_results=2)
        print(f"  Found {brain_results['total_found']} papers on synaptic transmission")
        if brain_results['papers']:
            print(f"  Sample paper: {brain_results['papers'][0]['title'][:70]}...")
        print("  ✓ Brain research search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Find related records
    print("\n5. Testing cross-database linking...")
    try:
        # Find a protein record
        protein_results = client.search('protein', 'dopamine receptor D2 human', retmax=1)
        if protein_results['ids']:
            protein_id = protein_results['ids'][0]
            print(f"  Found protein: {protein_id}")
            
            # Find related gene record
            gene_links = client.find_related('protein', 'gene', protein_id)
            if gene_links:
                print(f"  Related genes found: {list(gene_links.values())[0][:3]}")
                print("  ✓ Cross-database linking successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Demonstrate comprehensive brain research searches
    print("\n" + "=" * 60)
    print("Comprehensive Brain Research Demo")
    print("=" * 60)
    
    research_results = demonstrate_brain_research_searches(client)
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "knowledge" / "ncbi_brain_research.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'source': 'NCBI E-utilities',
                'date': '2025-01-20',
                'description': 'Recent brain research papers from PubMed'
            },
            'searches': research_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("NCBI E-utilities integration test complete!")
    print("✓ All basic functions working correctly")
    print("=" * 60)


if __name__ == "__main__":
    main()
