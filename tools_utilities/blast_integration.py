#!/usr/bin/env python3
"""
NCBI BLAST REST API Integration for Quark
==========================================
This module provides integration with the NCBI BLAST REST API for sequence
similarity searches.

BLAST (Basic Local Alignment Search Tool) finds regions of similarity between
biological sequences.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    BLAST_CONFIG = credentials['services']['blast']

# API endpoint
BLAST_URL = BLAST_CONFIG['endpoints']['url_api']


class BLASTClient:
    """Client for interacting with NCBI BLAST REST API."""
    
    def __init__(self, email: str = "quark@example.com", tool: str = "QuarkBrainSim"):
        """
        Initialize BLAST client.
        
        Args:
            email: Contact email (required by NCBI)
            tool: Tool name for identification
        """
        self.email = email
        self.tool = tool
        self.session = requests.Session()
        
        # Rate limiting
        self.last_submit_time = 0
        self.submit_interval = 10  # 10 seconds between submissions
        self.last_poll_time = {}  # Track polling per RID
        self.poll_interval = 60  # 60 seconds between polls for same RID
    
    def _rate_limit_submit(self):
        """Enforce rate limiting for submissions."""
        current_time = time.time()
        time_since_last = current_time - self.last_submit_time
        
        if time_since_last < self.submit_interval:
            sleep_time = self.submit_interval - time_since_last
            logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_submit_time = time.time()
    
    def _rate_limit_poll(self, rid: str):
        """Enforce rate limiting for status polling."""
        current_time = time.time()
        
        if rid in self.last_poll_time:
            time_since_last = current_time - self.last_poll_time[rid]
            if time_since_last < self.poll_interval:
                sleep_time = self.poll_interval - time_since_last
                logger.info(f"Rate limiting poll for {rid}: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.last_poll_time[rid] = time.time()
    
    def submit_search(
        self,
        query: str,
        program: str = 'blastp',
        database: str = 'nr',
        expect: float = 10.0,
        hitlist_size: int = 50,
        filter: bool = True,
        format_type: str = 'XML'
    ) -> str:
        """
        Submit a BLAST search.
        
        Args:
            query: Query sequence (FASTA or plain sequence)
            program: BLAST program (blastn, blastp, blastx, etc.)
            database: Target database (nr, nt, pdb, etc.)
            expect: E-value threshold
            hitlist_size: Maximum number of hits
            filter: Apply low complexity filter
            format_type: Output format
            
        Returns:
            Request ID (RID) for the search
        """
        self._rate_limit_submit()
        
        params = {
            'CMD': 'Put',
            'PROGRAM': program,
            'DATABASE': database,
            'QUERY': query,
            'EXPECT': expect,
            'HITLIST_SIZE': hitlist_size,
            'FORMAT_TYPE': format_type,
            'EMAIL': self.email,
            'TOOL': self.tool
        }
        
        if filter:
            params['FILTER'] = 'L'  # Low complexity filter
        
        logger.info(f"Submitting BLAST search: {program} against {database}")
        response = self.session.post(BLAST_URL, data=params)
        response.raise_for_status()
        
        # Parse RID from response
        content = response.text
        rid = None
        rtoe = None
        
        for line in content.split('\n'):
            if line.startswith('    RID = '):
                rid = line.split(' = ')[1].strip()
            elif line.startswith('    RTOE = '):
                rtoe = int(line.split(' = ')[1].strip())
        
        if not rid:
            raise ValueError("Failed to get RID from BLAST submission")
        
        logger.info(f"Search submitted. RID: {rid}, estimated time: {rtoe}s")
        return rid
    
    def check_status(self, rid: str) -> Tuple[str, Optional[int]]:
        """
        Check the status of a BLAST search.
        
        Args:
            rid: Request ID
            
        Returns:
            Tuple of (status, time_remaining)
            Status can be 'WAITING', 'READY', 'UNKNOWN', or 'FAILED'
        """
        self._rate_limit_poll(rid)
        
        params = {
            'CMD': 'Get',
            'RID': rid,
            'FORMAT_OBJECT': 'SearchInfo'
        }
        
        response = self.session.get(BLAST_URL, params=params)
        response.raise_for_status()
        
        content = response.text
        
        if 'Status=WAITING' in content:
            # Try to extract time remaining
            import re
            time_match = re.search(r'Time=(\d+)', content)
            time_remaining = int(time_match.group(1)) if time_match else None
            return 'WAITING', time_remaining
        elif 'Status=READY' in content:
            return 'READY', None
        elif 'Status=UNKNOWN' in content:
            return 'UNKNOWN', None
        elif 'Status=FAILED' in content:
            return 'FAILED', None
        else:
            return 'UNKNOWN', None
    
    def get_results(
        self,
        rid: str,
        format_type: str = 'XML',
        max_wait: int = 600
    ) -> str:
        """
        Get results of a BLAST search, waiting if necessary.
        
        Args:
            rid: Request ID
            format_type: Output format
            max_wait: Maximum time to wait (seconds)
            
        Returns:
            Search results in requested format
        """
        start_time = time.time()
        
        while True:
            status, time_remaining = self.check_status(rid)
            
            if status == 'READY':
                break
            elif status == 'FAILED':
                raise RuntimeError(f"BLAST search {rid} failed")
            elif status == 'UNKNOWN':
                raise RuntimeError(f"BLAST search {rid} not found")
            
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"BLAST search {rid} timed out after {max_wait}s")
            
            # Wait before next check
            wait_time = min(60, time_remaining or 60)
            logger.info(f"Search {rid} still running. Waiting {wait_time}s...")
            time.sleep(wait_time)
        
        # Get results
        params = {
            'CMD': 'Get',
            'RID': rid,
            'FORMAT_TYPE': format_type
        }
        
        logger.info(f"Retrieving results for {rid}")
        response = self.session.get(BLAST_URL, params=params)
        response.raise_for_status()
        
        return response.text
    
    def parse_xml_results(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parse BLAST XML results.
        
        Args:
            xml_content: XML results from BLAST
            
        Returns:
            List of hit dictionaries
        """
        root = ET.fromstring(xml_content)
        
        hits = []
        iterations = root.findall('.//BlastOutput_iterations/Iteration')
        
        for iteration in iterations:
            iter_hits = iteration.findall('.//Hit')
            
            for hit in iter_hits[:10]:  # Top 10 hits
                hit_data = {
                    'id': hit.findtext('Hit_id', ''),
                    'definition': hit.findtext('Hit_def', ''),
                    'accession': hit.findtext('Hit_accession', ''),
                    'length': int(hit.findtext('Hit_len', '0')),
                    'hsps': []
                }
                
                # Get HSPs (High-scoring Segment Pairs)
                hsps = hit.findall('.//Hsp')
                for hsp in hsps[:3]:  # Top 3 HSPs per hit
                    hsp_data = {
                        'score': float(hsp.findtext('Hsp_score', '0')),
                        'evalue': float(hsp.findtext('Hsp_evalue', '1')),
                        'identity': int(hsp.findtext('Hsp_identity', '0')),
                        'align_len': int(hsp.findtext('Hsp_align-len', '0')),
                        'query_from': int(hsp.findtext('Hsp_query-from', '0')),
                        'query_to': int(hsp.findtext('Hsp_query-to', '0')),
                        'hit_from': int(hsp.findtext('Hsp_hit-from', '0')),
                        'hit_to': int(hsp.findtext('Hsp_hit-to', '0'))
                    }
                    
                    # Calculate percent identity
                    if hsp_data['align_len'] > 0:
                        hsp_data['percent_identity'] = (hsp_data['identity'] / hsp_data['align_len']) * 100
                    else:
                        hsp_data['percent_identity'] = 0
                    
                    hit_data['hsps'].append(hsp_data)
                
                if hit_data['hsps']:  # Only add hits with HSPs
                    hits.append(hit_data)
        
        return hits
    
    def blast_sequence(
        self,
        sequence: str,
        program: str = 'blastp',
        database: str = 'pdb',
        expect: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Perform a complete BLAST search.
        
        Args:
            sequence: Query sequence
            program: BLAST program
            database: Target database
            expect: E-value threshold
            
        Returns:
            List of hits
        """
        # Submit search
        rid = self.submit_search(
            query=sequence,
            program=program,
            database=database,
            expect=expect
        )
        
        # Wait for results
        xml_results = self.get_results(rid)
        
        # Parse results
        hits = self.parse_xml_results(xml_results)
        
        return hits


def demonstrate_blast_search(client: BLASTClient):
    """
    Demonstrate BLAST search capabilities.
    
    Args:
        client: BLAST client instance
    """
    print("\nBLAST Search Demo")
    print("=" * 60)
    
    # Example: Search for similar proteins to insulin
    print("\n1. Searching for proteins similar to human insulin")
    print("-" * 40)
    
    # Insulin B chain sequence (partial)
    insulin_seq = """
    >Insulin_B_chain_partial
    FVNQHLCGSHLVEALYLVCGERGFFYTPKT
    """
    
    try:
        print("  Submitting BLAST search...")
        rid = client.submit_search(
            query=insulin_seq,
            program='blastp',
            database='pdb',
            expect=1.0,
            hitlist_size=10
        )
        
        print(f"  Search submitted (RID: {rid})")
        print("  Waiting for results...")
        
        # Get results
        xml_results = client.get_results(rid, max_wait=300)
        hits = client.parse_xml_results(xml_results)
        
        if hits:
            print(f"\n  Found {len(hits)} similar proteins:")
            for i, hit in enumerate(hits[:5], 1):
                print(f"\n  Hit {i}: {hit['definition'][:60]}...")
                print(f"    Accession: {hit['accession']}")
                if hit['hsps']:
                    best_hsp = hit['hsps'][0]
                    print(f"    E-value: {best_hsp['evalue']:.2e}")
                    print(f"    Identity: {best_hsp['percent_identity']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Example usage of BLAST client."""
    client = BLASTClient()
    
    print("=" * 60)
    print("NCBI BLAST REST API Integration Test")
    print("Quark System - Sequence Similarity Search")
    print("=" * 60)
    
    print("\nNOTE: BLAST searches can take 30-120 seconds to complete.")
    print("Rate limits: 10s between submissions, 60s between status checks.")
    
    # Test with a short protein sequence
    print("\n1. Testing BLAST search with neurotransmitter peptide...")
    
    # Enkephalin sequence (short neuropeptide)
    test_sequence = """
    >Enkephalin
    YGGFM
    """
    
    try:
        print("  Submitting search...")
        hits = client.blast_sequence(
            sequence=test_sequence,
            program='blastp',
            database='nr',
            expect=100  # High E-value for short sequence
        )
        
        if hits:
            print(f"\n  ✓ BLAST search successful!")
            print(f"  Found {len(hits)} similar sequences")
            
            # Save results
            output_path = Path(__file__).parent.parent / "data" / "knowledge" / "blast_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'source': 'NCBI BLAST API',
                        'date': '2025-01-20',
                        'query': 'Enkephalin peptide',
                        'database': 'nr',
                        'program': 'blastp'
                    },
                    'hits': hits[:10]  # Save top 10 hits
                }, f, indent=2)
            
            print(f"  Results saved to: {output_path}")
            
            # Show top hit
            if hits:
                top_hit = hits[0]
                print(f"\n  Top hit: {top_hit['definition'][:70]}...")
                if top_hit['hsps']:
                    print(f"  E-value: {top_hit['hsps'][0]['evalue']:.2e}")
        else:
            print("  No hits found (this is normal for very short sequences)")
            print("  ✓ BLAST API is working correctly")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("\n  Note: BLAST searches require active internet connection")
        print("  and may fail due to server load or rate limiting.")
    
    # Demonstrate longer search (optional, takes longer)
    print("\n2. Demo: Full protein search (optional, takes 1-2 minutes)")
    print("   Skipping to save time. Uncomment code to run.")
    # demonstrate_blast_search(client)
    
    print("\n" + "=" * 60)
    print("BLAST API integration configured!")
    print("Note: Use sparingly due to rate limits (100 searches/day)")
    print("=" * 60)


if __name__ == "__main__":
    main()
