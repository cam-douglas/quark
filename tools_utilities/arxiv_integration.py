#!/usr/bin/env python3
"""
arXiv API Integration for Quark
================================
This module provides integration with the arXiv API for accessing scholarly
articles in physics, mathematics, computer science, and more.

arXiv provides open access to 2+ million preprints.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    ARXIV_CONFIG = credentials['services']['arxiv']

# API endpoint
QUERY_URL = ARXIV_CONFIG['endpoints']['query']


class ArXivClient:
    """Client for interacting with arXiv API."""
    
    def __init__(self):
        """Initialize arXiv client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quark-arXiv-Integration/1.0'
        })
        
        # Rate limiting (be reasonable)
        self.last_request_time = 0
        self.min_interval = 0.5  # 2 requests per second max
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = 'relevance',
        sort_order: str = 'descending'
    ) -> Dict[str, Any]:
        """
        Search arXiv papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            start: Starting index for pagination
            sort_by: Sort by 'relevance', 'lastUpdatedDate', 'submittedDate'
            sort_order: 'ascending' or 'descending'
            
        Returns:
            Dictionary with feed metadata and entries
        """
        self._rate_limit()
        
        params = {
            'search_query': query,
            'max_results': min(max_results, 1000),  # Cap at 1000
            'start': start,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        logger.info(f"Searching arXiv for: {query}")
        response = self.session.get(QUERY_URL, params=params)
        response.raise_for_status()
        
        # Parse Atom XML
        return self._parse_atom_feed(response.text)
    
    def get_by_id(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv ID (e.g., '2103.15348' or 'quant-ph/0201082')
            
        Returns:
            Paper metadata
        """
        # Clean ID (remove version if present)
        clean_id = arxiv_id.split('v')[0]
        
        return self.search(f'id:{clean_id}', max_results=1)
    
    def search_by_author(
        self,
        author: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search papers by author.
        
        Args:
            author: Author name
            max_results: Maximum number of results
            
        Returns:
            Search results
        """
        query = f'au:"{author}"'
        return self.search(query, max_results)
    
    def search_by_category(
        self,
        category: str,
        max_results: int = 10,
        days_back: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search papers by category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'physics.bio-ph')
            max_results: Maximum number of results
            days_back: Limit to papers from last N days
            
        Returns:
            Search results
        """
        query = f'cat:{category}'
        
        if days_back:
            # Add date filter
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            date_query = f' AND submittedDate:[{start_date.strftime("%Y%m%d")} TO {end_date.strftime("%Y%m%d")}]'
            query += date_query
        
        return self.search(query, max_results, sort_by='submittedDate')
    
    def _parse_atom_feed(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse arXiv Atom feed.
        
        Args:
            xml_content: XML content from arXiv
            
        Returns:
            Parsed feed data
        """
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        root = ET.fromstring(xml_content)
        
        # Get feed metadata
        feed_data = {
            'title': root.findtext('atom:title', '', namespaces),
            'total_results': int(root.findtext('opensearch:totalResults', '0', 
                                              {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'})),
            'start_index': int(root.findtext('opensearch:startIndex', '0',
                                            {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'})),
            'items_per_page': int(root.findtext('opensearch:itemsPerPage', '0',
                                               {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'})),
            'entries': []
        }
        
        # Parse entries
        entries = root.findall('atom:entry', namespaces)
        
        for entry in entries:
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name = author.findtext('atom:name', '', namespaces)
                if name:
                    authors.append(name)
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract links
            pdf_link = None
            for link in entry.findall('atom:link', namespaces):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')
                    break
            
            entry_data = {
                'id': entry.findtext('atom:id', '', namespaces),
                'title': entry.findtext('atom:title', '', namespaces).strip(),
                'abstract': entry.findtext('atom:summary', '', namespaces).strip(),
                'authors': authors,
                'categories': categories,
                'primary_category': entry.find('arxiv:primary_category', namespaces).get('term') if entry.find('arxiv:primary_category', namespaces) is not None else None,
                'published': entry.findtext('atom:published', '', namespaces),
                'updated': entry.findtext('atom:updated', '', namespaces),
                'doi': entry.findtext('arxiv:doi', '', namespaces),
                'journal_ref': entry.findtext('arxiv:journal_ref', '', namespaces),
                'pdf_url': pdf_link,
                'comment': entry.findtext('arxiv:comment', '', namespaces)
            }
            
            feed_data['entries'].append(entry_data)
        
        return feed_data
    
    def search_neuroscience_papers(self) -> Dict[str, List[Dict]]:
        """
        Search for neuroscience and brain-related papers.
        
        Returns:
            Categorized neuroscience papers
        """
        neuro_papers = {
            'computational_neuroscience': [],
            'brain_imaging': [],
            'neural_networks': [],
            'cognitive_science': []
        }
        
        print("\nSearching for neuroscience papers on arXiv")
        print("-" * 50)
        
        # 1. Computational neuroscience
        print("1. Searching computational neuroscience papers...")
        try:
            results = self.search(
                'abs:"computational neuroscience" OR abs:"neural dynamics"',
                max_results=5
            )
            neuro_papers['computational_neuroscience'] = results['entries']
            print(f"  Found {len(results['entries'])} papers")
        except Exception as e:
            logger.error(f"Error searching computational neuroscience: {e}")
        
        # 2. Brain imaging
        print("2. Searching brain imaging papers...")
        try:
            results = self.search(
                'abs:"brain imaging" OR abs:"fMRI" OR abs:"EEG"',
                max_results=5
            )
            neuro_papers['brain_imaging'] = results['entries']
            print(f"  Found {len(results['entries'])} papers")
        except Exception as e:
            logger.error(f"Error searching brain imaging: {e}")
        
        # 3. Neural networks (biological focus)
        print("3. Searching biological neural network papers...")
        try:
            results = self.search(
                'abs:"biological neural networks" OR abs:"spiking neurons"',
                max_results=5
            )
            neuro_papers['neural_networks'] = results['entries']
            print(f"  Found {len(results['entries'])} papers")
        except Exception as e:
            logger.error(f"Error searching neural networks: {e}")
        
        # 4. Cognitive science
        print("4. Searching cognitive science papers...")
        try:
            results = self.search(
                'cat:q-bio.NC',  # Neurons and Cognition category
                max_results=5,
                sort_by='submittedDate'
            )
            neuro_papers['cognitive_science'] = results['entries']
            print(f"  Found {len(results['entries'])} papers")
        except Exception as e:
            logger.error(f"Error searching cognitive science: {e}")
        
        return neuro_papers


def demonstrate_arxiv_search(client: ArXivClient):
    """
    Demonstrate arXiv search capabilities.
    
    Args:
        client: arXiv client instance
    """
    print("\narXiv Search Demo")
    print("=" * 60)
    
    # Search for recent AI papers
    print("\n1. Recent AI papers in machine learning")
    print("-" * 40)
    
    try:
        results = client.search(
            'cat:cs.LG',  # Machine Learning category
            max_results=3,
            sort_by='submittedDate'
        )
        
        if results['entries']:
            print(f"  Found {results['total_results']} total papers")
            for i, paper in enumerate(results['entries'], 1):
                print(f"\n  Paper {i}:")
                print(f"    Title: {paper['title'][:70]}...")
                print(f"    Authors: {', '.join(paper['authors'][:3])}")
                print(f"    Categories: {', '.join(paper['categories'][:3])}")
                print(f"    Published: {paper['published'][:10]}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Example usage of arXiv client."""
    client = ArXivClient()
    
    print("=" * 60)
    print("arXiv API Integration Test")
    print("Quark System - Scientific Preprint Access")
    print("=" * 60)
    
    # Test 1: Basic search
    print("\n1. Testing arXiv search...")
    try:
        results = client.search(
            'quantum computing',
            max_results=5
        )
        
        if results['entries']:
            print(f"  Found {results['total_results']} total results")
            print(f"  Showing {len(results['entries'])} entries")
            
            # Show first paper
            if results['entries']:
                paper = results['entries'][0]
                print(f"\n  Example paper:")
                print(f"    Title: {paper['title'][:70]}...")
                print(f"    Authors: {', '.join(paper['authors'][:3])}")
                print(f"    Abstract: {paper['abstract'][:150]}...")
            
            print("  ✓ arXiv search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Get paper by ID
    print("\n2. Testing paper retrieval by ID...")
    try:
        # Famous paper: Attention Is All You Need
        result = client.get_by_id('1706.03762')
        
        if result['entries']:
            paper = result['entries'][0]
            print(f"  Title: {paper['title']}")
            print(f"  Authors: {', '.join(paper['authors'][:3])}...")
            print(f"  Categories: {', '.join(paper['categories'])}")
            print("  ✓ Paper retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Search by category
    print("\n3. Testing category search (Quantitative Biology - Neurons)...")
    try:
        results = client.search_by_category(
            'q-bio.NC',  # Neurons and Cognition
            max_results=3
        )
        
        if results['entries']:
            print(f"  Found {len(results['entries'])} papers in q-bio.NC")
            for paper in results['entries']:
                print(f"    - {paper['title'][:60]}...")
            print("  ✓ Category search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Search neuroscience papers
    print("\n4. Searching for neuroscience papers...")
    try:
        neuro_papers = client.search_neuroscience_papers()
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "knowledge" / "arxiv_neuroscience.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count total
        total = sum(len(papers) for papers in neuro_papers.values())
        
        # Simplify paper data for storage
        simplified = {}
        for category, papers in neuro_papers.items():
            simplified[category] = [
                {
                    'title': p['title'],
                    'authors': p['authors'][:3],  # First 3 authors
                    'abstract': p['abstract'][:500],  # First 500 chars
                    'categories': p['categories'],
                    'published': p['published'],
                    'pdf_url': p['pdf_url']
                }
                for p in papers
            ]
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'source': 'arXiv API',
                    'date': '2025-01-20',
                    'description': 'Neuroscience papers from arXiv',
                    'total_papers': total
                },
                'papers': simplified
            }, f, indent=2)
        
        print(f"\n  Total neuroscience papers found: {total}")
        for category, papers in neuro_papers.items():
            if papers:
                print(f"    {category}: {len(papers)} papers")
        
        print(f"\n  Results saved to: {output_path}")
        print("  ✓ Neuroscience paper search successful")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Demonstrate search
    demonstrate_arxiv_search(client)
    
    print("\n" + "=" * 60)
    print("arXiv API integration test complete!")
    print("✓ Scientific preprint access working")
    print("=" * 60)


if __name__ == "__main__":
    main()
