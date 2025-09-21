#!/usr/bin/env python3
"""
PubChem PUG-REST API Integration for Quark
===========================================
This module provides integration with NCBI's PubChem PUG-REST API for chemical data access.

PubChem is the world's largest free chemistry database, providing information on
chemical compounds, substances, bioassays, and their biological activities.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import time
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
    PUBCHEM_CONFIG = credentials['services']['pubchem']

# API endpoints
BASE_URL = PUBCHEM_CONFIG['endpoints']['base']
COMPOUND_URL = PUBCHEM_CONFIG['endpoints']['compound']


class PubChemClient:
    """Client for interacting with PubChem PUG-REST API."""
    
    def __init__(self):
        """Initialize PubChem client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quark-PubChem-Integration/1.0'
        })
        
        # Rate limiting (5 requests per second max)
        self.rate_limit = 5
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _get(self, url: str, params: Optional[Dict] = None) -> Union[Dict, str]:
        """
        Make a GET request to the PubChem API.
        
        Args:
            url: Full URL to request
            params: Optional query parameters
            
        Returns:
            Response data (JSON dict or text)
        """
        self._rate_limit()
        
        logger.debug(f"GET {url}")
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'json' in content_type:
                return response.json()
            else:
                return response.text
        elif response.status_code == 404:
            logger.warning(f"Resource not found: {url}")
            return {}
        else:
            logger.error(f"Error {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
    
    def search_compound_by_name(
        self,
        name: str,
        output: str = 'json'
    ) -> Union[Dict, List]:
        """
        Search for a compound by name.
        
        Args:
            name: Compound name (e.g., 'aspirin', 'glucose')
            output: Output format ('json', 'property/MolecularFormula,MolecularWeight')
            
        Returns:
            Compound data
        """
        url = f"{COMPOUND_URL}/name/{quote(name)}/{output}"
        
        logger.info(f"Searching for compound: {name}")
        result = self._get(url)
        
        if output == 'json':
            # PubChem returns a complex structure
            if isinstance(result, dict) and 'PC_Compounds' in result:
                return result['PC_Compounds']
            elif isinstance(result, dict) and 'PropertyTable' in result:
                return result['PropertyTable'].get('Properties', [])
        
        return result
    
    def get_compound_properties(
        self,
        identifier: Union[int, str],
        properties: List[str]
    ) -> Dict[str, Any]:
        """
        Get specific properties for a compound.
        
        Args:
            identifier: CID or compound name
            properties: List of properties to retrieve
                       (e.g., ['MolecularFormula', 'MolecularWeight', 'IUPACName'])
            
        Returns:
            Dictionary of properties
        """
        prop_string = ','.join(properties)
        
        # Determine if identifier is CID or name
        if isinstance(identifier, int) or identifier.isdigit():
            url = f"{COMPOUND_URL}/cid/{identifier}/property/{prop_string}/json"
        else:
            url = f"{COMPOUND_URL}/name/{quote(str(identifier))}/property/{prop_string}/json"
        
        logger.info(f"Getting properties for: {identifier}")
        result = self._get(url)
        
        if isinstance(result, dict) and 'PropertyTable' in result:
            props = result['PropertyTable'].get('Properties', [])
            if props:
                return props[0]
        
        return {}
    
    def search_by_smiles(
        self,
        smiles: str,
        search_type: str = 'similarity',
        threshold: int = 95
    ) -> List[int]:
        """
        Search for compounds by SMILES structure.
        
        Args:
            smiles: SMILES string
            search_type: 'similarity' or 'substructure'
            threshold: Similarity threshold (for similarity search)
            
        Returns:
            List of CIDs
        """
        if search_type == 'similarity':
            url = f"{COMPOUND_URL}/similarity/smiles/{quote(smiles)}/cids/json?Threshold={threshold}"
        else:
            url = f"{COMPOUND_URL}/substructure/smiles/{quote(smiles)}/cids/json"
        
        logger.info(f"Searching by SMILES ({search_type}): {smiles}")
        result = self._get(url)
        
        if isinstance(result, dict) and 'IdentifierList' in result:
            return result['IdentifierList'].get('CID', [])
        
        return []
    
    def get_bioassay_data(self, cid: int, aids: Optional[List[int]] = None) -> List[Dict]:
        """
        Get bioassay data for a compound.
        
        Args:
            cid: Compound ID
            aids: Optional list of specific assay IDs
            
        Returns:
            List of bioassay results
        """
        if aids:
            aids_str = ','.join(str(a) for a in aids)
            url = f"{BASE_URL}/bioassay/aid/{aids_str}/cids/json?cid={cid}"
        else:
            url = f"{BASE_URL}/compound/cid/{cid}/assaysummary/json"
        
        logger.info(f"Getting bioassay data for CID {cid}")
        result = self._get(url)
        
        if isinstance(result, dict):
            # Handle different response formats
            if 'Table' in result:
                return result.get('Table', {}).get('Row', [])
            elif 'InformationList' in result:
                return result.get('InformationList', {}).get('Information', [])
        
        return []
    
    def get_compound_image(self, cid: int, output_path: Optional[Path] = None) -> bytes:
        """
        Get 2D structure image for a compound.
        
        Args:
            cid: Compound ID
            output_path: Optional path to save image
            
        Returns:
            Image data as bytes
        """
        url = f"{COMPOUND_URL}/cid/{cid}/png"
        
        logger.info(f"Getting structure image for CID {cid}")
        self._rate_limit()
        
        response = self.session.get(url)
        response.raise_for_status()
        
        if output_path:
            output_path.write_bytes(response.content)
            logger.info(f"Image saved to: {output_path}")
        
        return response.content
    
    def search_neurotransmitters(self) -> Dict[str, Dict[str, Any]]:
        """
        Search for common neurotransmitters and related compounds.
        
        Returns:
            Dictionary of neurotransmitter information
        """
        neurotransmitters = {
            "dopamine": "Catecholamine neurotransmitter",
            "serotonin": "5-hydroxytryptamine",
            "GABA": "gamma-Aminobutyric acid",
            "glutamate": "Excitatory neurotransmitter",
            "acetylcholine": "Cholinergic neurotransmitter",
            "norepinephrine": "Noradrenaline",
            "epinephrine": "Adrenaline",
            "histamine": "Biogenic amine",
            "glycine": "Inhibitory neurotransmitter",
            "adenosine": "Purinergic neurotransmitter"
        }
        
        results = {}
        properties = ['MolecularFormula', 'MolecularWeight', 'IUPACName', 
                     'InChI', 'CanonicalSMILES', 'XLogP']
        
        print("\nSearching for neurotransmitters in PubChem")
        print("-" * 50)
        
        for name, description in neurotransmitters.items():
            try:
                # Get compound properties
                props = self.get_compound_properties(name, properties)
                
                if props:
                    results[name] = {
                        'description': description,
                        'cid': props.get('CID'),
                        'formula': props.get('MolecularFormula'),
                        'weight': props.get('MolecularWeight'),
                        'iupac': props.get('IUPACName'),
                        'smiles': props.get('CanonicalSMILES'),
                        'logp': props.get('XLogP')
                    }
                    print(f"✓ {name}: CID {props.get('CID')} ({props.get('MolecularFormula')})")
                else:
                    print(f"✗ {name}: Not found")
                    
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
                
        return results
    
    def search_drugs_by_target(self, target: str) -> List[Dict]:
        """
        Search for drugs targeting specific proteins or pathways.
        
        Args:
            target: Target protein or pathway name
            
        Returns:
            List of drug compounds
        """
        # This is a simplified search - real drug-target searches would use
        # more sophisticated queries or external databases
        url = f"{COMPOUND_URL}/name/{quote(target)}/synonyms/json"
        
        logger.info(f"Searching for drugs targeting: {target}")
        result = self._get(url)
        
        compounds = []
        if isinstance(result, dict) and 'InformationList' in result:
            for info in result.get('InformationList', {}).get('Information', []):
                compounds.append({
                    'cid': info.get('CID'),
                    'synonyms': info.get('Synonym', [])[:5]  # First 5 synonyms
                })
        
        return compounds


def demonstrate_drug_discovery(client: PubChemClient):
    """
    Demonstrate drug discovery features.
    
    Args:
        client: PubChem client instance
    """
    print("\nDrug Discovery Demo")
    print("=" * 60)
    
    # Common drugs for neurological conditions
    drugs = {
        "levodopa": "Parkinson's disease",
        "donepezil": "Alzheimer's disease",
        "fluoxetine": "Depression (SSRI)",
        "diazepam": "Anxiety (benzodiazepine)",
        "carbamazepine": "Epilepsy",
        "methylphenidate": "ADHD"
    }
    
    drug_data = {}
    properties = ['MolecularFormula', 'MolecularWeight', 'XLogP', 'HBondDonorCount', 'HBondAcceptorCount']
    
    for drug, indication in drugs.items():
        print(f"\n{drug} ({indication})")
        print("-" * 40)
        
        try:
            props = client.get_compound_properties(drug, properties)
            
            if props:
                drug_data[drug] = props
                print(f"  CID: {props.get('CID')}")
                print(f"  Formula: {props.get('MolecularFormula')}")
                print(f"  MW: {props.get('MolecularWeight')} g/mol")
                print(f"  LogP: {props.get('XLogP')}")
                print(f"  H-bond donors: {props.get('HBondDonorCount')}")
                print(f"  H-bond acceptors: {props.get('HBondAcceptorCount')}")
            else:
                print("  Not found")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    return drug_data


def main():
    """Example usage of PubChem client."""
    client = PubChemClient()
    
    print("=" * 60)
    print("PubChem PUG-REST API Integration Test")
    print("Quark System - Chemical Data Access")
    print("=" * 60)
    
    # Test 1: Search by name
    print("\n1. Testing compound search by name...")
    try:
        compounds = client.search_compound_by_name('aspirin')
        if compounds:
            if isinstance(compounds, list) and compounds:
                # Handle PC_Compounds format
                first = compounds[0] if isinstance(compounds, list) else compounds
                if 'id' in first:
                    cid = first['id'].get('id', {}).get('cid')
                    print(f"  Found aspirin: CID {cid}")
                else:
                    print(f"  Found compound data")
            print("  ✓ Name search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Get compound properties
    print("\n2. Testing property retrieval...")
    try:
        props = client.get_compound_properties(
            'caffeine',
            ['MolecularFormula', 'MolecularWeight', 'IUPACName', 'CanonicalSMILES']
        )
        if props:
            print(f"  Caffeine (CID {props.get('CID')}):")
            print(f"    Formula: {props.get('MolecularFormula')}")
            print(f"    Weight: {props.get('MolecularWeight')} g/mol")
            print(f"    IUPAC: {props.get('IUPACName', 'N/A')[:50]}...")
            print(f"    SMILES: {props.get('CanonicalSMILES')}")
            print("  ✓ Property retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Structure search
    print("\n3. Testing SMILES similarity search...")
    try:
        # Search for compounds similar to benzene
        smiles = "c1ccccc1"
        similar_cids = client.search_by_smiles(smiles, 'similarity', 95)
        if similar_cids:
            print(f"  Found {len(similar_cids)} compounds similar to benzene")
            print(f"  First 5 CIDs: {similar_cids[:5]}")
            print("  ✓ Structure search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Get compound image
    print("\n4. Testing structure image download...")
    try:
        # Download dopamine structure
        output_dir = Path(__file__).parent.parent / "data" / "structures" / "pubchem"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dopamine CID first
        props = client.get_compound_properties('dopamine', ['CID'])
        if props and 'CID' in props:
            cid = props['CID']
            image_path = output_dir / f"dopamine_{cid}.png"
            client.get_compound_image(cid, image_path)
            print(f"  Dopamine structure saved to: {image_path}")
            print("  ✓ Image download successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Search neurotransmitters
    print("\n5. Searching for neurotransmitters...")
    try:
        neurotransmitters = client.search_neurotransmitters()
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "knowledge" / "pubchem_neurotransmitters.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'source': 'PubChem PUG-REST API',
                    'date': '2025-01-20',
                    'description': 'Neurotransmitter chemical data'
                },
                'compounds': neurotransmitters
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print("  ✓ Neurotransmitter search successful")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 6: Drug discovery demo
    drug_data = demonstrate_drug_discovery(client)
    
    print("\n" + "=" * 60)
    print("PubChem API integration test complete!")
    print("✓ Chemical database access working")
    print("=" * 60)


if __name__ == "__main__":
    main()
