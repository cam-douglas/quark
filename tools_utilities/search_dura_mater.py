#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from ncbi_eutilities_integration import NCBIEutilitiesClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_dura_mater_development(client: NCBIEutilitiesClient, max_results: int = 5):
    """
    Search PubMed for research on the embryonic development of the dura mater.
    """
    query = "(dura mater) AND (embryonic OR fetal OR development)"
    
    # Search PubMed
    search_results = client.search('pubmed', query, retmax=max_results)
    
    if not search_results['ids']:
        return {'papers': [], 'total_found': 0}
    
    # Get summaries
    summaries = client.get_summaries('pubmed', search_results['ids'])
    
    # Extract relevant information
    papers = []
    for summary in summaries:
        abstract = client.fetch('pubmed', summary.get('uid'), rettype='abstract', retmode='text')
        paper = {
            'pmid': summary.get('uid'),
            'title': summary.get('title'),
            'authors': ', '.join([
                author.get('name', '') 
                for author in summary.get('authors', [])[:3]
            ]) + (' et al.' if len(summary.get('authors', [])) > 3 else ''),
            'journal': summary.get('source'),
            'pub_date': summary.get('pubdate'),
            'doi': summary.get('elocationid', '').replace('doi: ', ''),
            'abstract': abstract
        }
        papers.append(paper)
        
    return {
        'papers': papers,
        'total_found': search_results['count'],
        'query_used': query
    }

if __name__ == "__main__":
    client = NCBIEutilitiesClient(email="quark@example.com", tool="QuarkDuraMaterResearch")
    
    print("=" * 60)
    print("Searching for Dura Mater Embryonic Development Research")
    print("=" * 60)
    
    results = search_dura_mater_development(client)
    
    print(f"Found {results['total_found']} total papers")
    print("\n---\n")
    
    for i, paper in enumerate(results['papers'], 1):
        print(f"### Paper {i} ###")
        print(f"Title: {paper['title']}")
        print(f"Authors: {paper['authors']}")
        print(f"Journal: {paper['journal']} ({paper['pub_date']})")
        print(f"DOI: {paper['doi']}")
        print("\nAbstract:")
        print(paper['abstract'])
        print("\n---\n")
