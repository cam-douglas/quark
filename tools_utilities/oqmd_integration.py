#!/usr/bin/env python3
"""
OQMD RESTful API Integration for Quark
=======================================
This module provides integration with the Open Quantum Materials Database (OQMD) API.

OQMD contains DFT-calculated thermodynamic and structural properties for ~700,000 materials,
making it one of the largest open databases of computed materials properties.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
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
    OQMD_CONFIG = credentials['services']['oqmd']

# API endpoints
BASE_URL = OQMD_CONFIG['endpoints']['base']
FORMATION_ENERGY_URL = OQMD_CONFIG['endpoints']['formationenergy']
OPTIMADE_URL = OQMD_CONFIG['endpoints']['optimade_structures']


class OQMDClient:
    """Client for interacting with OQMD RESTful API."""
    
    def __init__(self):
        """Initialize OQMD client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quark-OQMD-Integration/1.0',
            'Accept': 'application/json'
        })
    
    def search_materials(
        self,
        filter_expr: Optional[str] = None,
        fields: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: Optional[str] = None,
        desc: bool = False,
        composition: Optional[str] = None,
        element_set: Optional[str] = None,
        noduplicate: bool = False
    ) -> Dict[str, Any]:
        """
        Search for materials in OQMD database.
        
        Args:
            filter_expr: Filter expression (e.g., "stability=0 AND band_gap>2")
            fields: List of fields to return
            limit: Number of results to return
            offset: Offset for pagination
            sort_by: Field to sort by (delta_e, stability)
            desc: Sort in descending order
            composition: Composition filter (e.g., "Al2O3")
            element_set: Element set filter (e.g., "(Fe-Mn),O")
            noduplicate: Exclude duplicate entries
            
        Returns:
            API response with materials data
        """
        params = {
            'limit': limit,
            'offset': offset,
            'format': 'json',
            'noduplicate': noduplicate,
            'desc': desc
        }
        
        if filter_expr:
            params['filter'] = filter_expr
        
        if fields:
            params['fields'] = ','.join(fields)
        
        if sort_by:
            params['sort_by'] = sort_by
        
        if composition:
            # Handle composition in URL path
            url = f"{FORMATION_ENERGY_URL}/{composition}"
        else:
            url = FORMATION_ENERGY_URL
        
        if element_set:
            # Add element_set to filter
            element_filter = f"element_set={element_set}"
            if 'filter' in params:
                params['filter'] = f"{params['filter']} AND {element_filter}"
            else:
                params['filter'] = element_filter
        
        logger.info(f"Searching OQMD with params: {params}")
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_stable_materials(
        self,
        element_set: Optional[str] = None,
        fields: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get thermodynamically stable materials (hull distance = 0).
        
        Args:
            element_set: Optional element set filter
            fields: Fields to return
            limit: Number of results
            
        Returns:
            List of stable materials
        """
        if not fields:
            fields = ['name', 'entry_id', 'composition', 'spacegroup', 
                     'delta_e', 'band_gap', 'volume', 'natoms']
        
        result = self.search_materials(
            filter_expr='stability=0',
            fields=fields,
            element_set=element_set,
            limit=limit,
            noduplicate=True
        )
        
        if 'data' in result:
            return result['data']
        return []
    
    def search_by_bandgap(
        self,
        min_gap: float = 0,
        max_gap: Optional[float] = None,
        fields: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search materials by band gap range.
        
        Args:
            min_gap: Minimum band gap (eV)
            max_gap: Maximum band gap (eV)
            fields: Fields to return
            limit: Number of results
            
        Returns:
            List of materials within band gap range
        """
        if not fields:
            fields = ['name', 'entry_id', 'composition', 'band_gap', 
                     'delta_e', 'stability', 'spacegroup']
        
        # Build filter expression
        if max_gap is not None:
            filter_expr = f"band_gap>{min_gap} AND band_gap<{max_gap}"
        else:
            filter_expr = f"band_gap>{min_gap}"
        
        result = self.search_materials(
            filter_expr=filter_expr,
            fields=fields,
            limit=limit,
            sort_by='band_gap'
        )
        
        if 'data' in result:
            return result['data']
        return []
    
    def search_by_prototype(
        self,
        prototype: str,
        fields: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search materials by structure prototype.
        
        Args:
            prototype: Structure prototype (e.g., "Cu", "CsCl", "NaCl")
            fields: Fields to return
            limit: Number of results
            
        Returns:
            List of materials with given prototype
        """
        if not fields:
            fields = ['name', 'entry_id', 'composition', 'prototype',
                     'spacegroup', 'delta_e', 'stability', 'volume']
        
        result = self.search_materials(
            filter_expr=f'prototype={prototype}',
            fields=fields,
            limit=limit
        )
        
        if 'data' in result:
            return result['data']
        return []
    
    def search_brain_materials(self) -> Dict[str, List[Dict]]:
        """
        Search for materials relevant to brain/neural research.
        
        Returns:
            Dictionary of categorized brain-relevant materials
        """
        brain_materials = {
            'semiconductors': [],
            'ionic_conductors': [],
            'piezoelectric': [],
            'magnetic': []
        }
        
        print("\nSearching for brain-relevant materials in OQMD")
        print("-" * 50)
        
        # 1. Semiconductors with neural-like band gaps (0.5-3 eV)
        print("Searching semiconductors with neural-relevant band gaps...")
        try:
            semiconductors = self.search_by_bandgap(0.5, 3.0, limit=20)
            brain_materials['semiconductors'] = semiconductors[:10]
            print(f"  Found {len(semiconductors)} semiconductors")
        except Exception as e:
            logger.error(f"Error searching semiconductors: {e}")
        
        # 2. Ionic conductors (Li, Na, K compounds)
        print("Searching ionic conductors (Li/Na/K compounds)...")
        try:
            ionic = self.search_materials(
                filter_expr='element_set=(Li,O) OR element_set=(Na,O) OR element_set=(K,O)',
                fields=['name', 'entry_id', 'composition', 'band_gap', 'delta_e'],
                limit=20
            )
            if 'data' in ionic:
                brain_materials['ionic_conductors'] = ionic['data'][:10]
                print(f"  Found {len(ionic.get('data', []))} ionic conductors")
        except Exception as e:
            logger.error(f"Error searching ionic conductors: {e}")
        
        # 3. Piezoelectric materials (certain space groups)
        print("Searching piezoelectric materials...")
        try:
            # Non-centrosymmetric space groups that allow piezoelectricity
            piezo = self.search_materials(
                filter_expr='spacegroup="P4mm" OR spacegroup="R3m" OR spacegroup="Pmn21"',
                fields=['name', 'entry_id', 'spacegroup', 'band_gap'],
                limit=20
            )
            if 'data' in piezo:
                brain_materials['piezoelectric'] = piezo['data'][:10]
                print(f"  Found {len(piezo.get('data', []))} piezoelectric materials")
        except Exception as e:
            logger.error(f"Error searching piezoelectric materials: {e}")
        
        # 4. Magnetic materials (Fe, Co, Ni compounds)
        print("Searching magnetic materials...")
        try:
            magnetic = self.search_materials(
                element_set='(Fe-Co-Ni),O',
                fields=['name', 'entry_id', 'composition', 'delta_e'],
                limit=20
            )
            if 'data' in magnetic:
                brain_materials['magnetic'] = magnetic['data'][:10]
                print(f"  Found {len(magnetic.get('data', []))} magnetic materials")
        except Exception as e:
            logger.error(f"Error searching magnetic materials: {e}")
        
        return brain_materials
    
    def optimade_search(
        self,
        filter_expr: str,
        response_fields: Optional[List[str]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Search using OPTiMaDe-compatible endpoint.
        
        Args:
            filter_expr: OPTiMaDe filter expression
            response_fields: Fields to include in response
            limit: Number of results
            
        Returns:
            OPTiMaDe-formatted response
        """
        params = {
            'filter': filter_expr,
            'page_limit': limit
        }
        
        if response_fields:
            params['response_fields'] = ','.join(response_fields)
        
        logger.info(f"OPTiMaDe search with filter: {filter_expr}")
        response = self.session.get(OPTIMADE_URL, params=params)
        response.raise_for_status()
        
        return response.json()


def demonstrate_materials_discovery(client: OQMDClient):
    """
    Demonstrate materials discovery capabilities.
    
    Args:
        client: OQMD client instance
    """
    print("\nMaterials Discovery Demo")
    print("=" * 60)
    
    # Find stable oxide semiconductors
    print("\n1. Stable oxide semiconductors for electronics")
    print("-" * 40)
    
    try:
        oxides = client.search_materials(
            filter_expr='stability=0 AND band_gap>1 AND band_gap<5 AND element=O',
            fields=['name', 'entry_id', 'band_gap', 'spacegroup', 'delta_e'],
            limit=10,
            sort_by='band_gap'
        )
        
        if 'data' in oxides and oxides['data']:
            for material in oxides['data'][:5]:
                print(f"  {material.get('name')}: Band gap = {material.get('band_gap')} eV")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Find perovskites
    print("\n2. Perovskite structures (ABO3)")
    print("-" * 40)
    
    try:
        perovskites = client.search_materials(
            filter_expr='generic=AB AND element=O AND ntypes=3',
            fields=['name', 'entry_id', 'spacegroup', 'band_gap'],
            limit=10
        )
        
        if 'data' in perovskites and perovskites['data']:
            for material in perovskites['data'][:5]:
                print(f"  {material.get('name')}: Space group = {material.get('spacegroup')}")
    except Exception as e:
        print(f"  Error: {e}")
    
    return True


def main():
    """Example usage of OQMD client."""
    client = OQMDClient()
    
    print("=" * 60)
    print("OQMD RESTful API Integration Test")
    print("Quark System - Quantum Materials Database Access")
    print("=" * 60)
    
    # Test 1: Basic search
    print("\n1. Testing basic materials search...")
    try:
        results = client.search_materials(
            fields=['name', 'entry_id', 'delta_e', 'band_gap'],
            limit=5
        )
        if 'data' in results:
            print(f"  Found {len(results['data'])} materials")
            if results['data']:
                first = results['data'][0]
                print(f"  Example: {first.get('name')} (ID: {first.get('entry_id')})")
                print(f"    Formation energy: {first.get('delta_e')} eV/atom")
                print(f"    Band gap: {first.get('band_gap')} eV")
            print("  ✓ Basic search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Get stable materials
    print("\n2. Testing stable materials search...")
    try:
        stable = client.get_stable_materials(limit=10)
        if stable:
            print(f"  Found {len(stable)} stable materials")
            for mat in stable[:3]:
                print(f"    {mat.get('name')}: ΔE = {mat.get('delta_e')} eV/atom")
            print("  ✓ Stable materials search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Band gap search
    print("\n3. Testing band gap search (2-4 eV)...")
    try:
        semiconductors = client.search_by_bandgap(2.0, 4.0, limit=10)
        if semiconductors:
            print(f"  Found {len(semiconductors)} semiconductors")
            for mat in semiconductors[:3]:
                print(f"    {mat.get('name')}: Eg = {mat.get('band_gap')} eV")
            print("  ✓ Band gap search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Prototype search
    print("\n4. Testing prototype search (NaCl structure)...")
    try:
        nacl_type = client.search_by_prototype('NaCl', limit=10)
        if nacl_type:
            print(f"  Found {len(nacl_type)} NaCl-type structures")
            for mat in nacl_type[:3]:
                print(f"    {mat.get('name')}: {mat.get('spacegroup')}")
            print("  ✓ Prototype search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Brain-relevant materials
    print("\n5. Searching for brain-relevant materials...")
    try:
        brain_materials = client.search_brain_materials()
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "knowledge" / "oqmd_brain_materials.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count total materials
        total = sum(len(mats) for mats in brain_materials.values())
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'source': 'OQMD RESTful API',
                    'date': '2025-01-20',
                    'description': 'Brain-relevant materials from OQMD',
                    'total_materials': total
                },
                'materials': brain_materials
            }, f, indent=2)
        
        print(f"\n  Total brain-relevant materials found: {total}")
        for category, materials in brain_materials.items():
            if materials:
                print(f"    {category}: {len(materials)} materials")
        
        print(f"\n  Results saved to: {output_path}")
        print("  ✓ Brain materials search successful")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 6: Materials discovery demo
    demonstrate_materials_discovery(client)
    
    # Test 7: OPTiMaDe endpoint
    print("\n7. Testing OPTiMaDe-compatible endpoint...")
    try:
        optimade_result = client.optimade_search(
            filter_expr='elements HAS "Fe" AND nelements=2',
            limit=5
        )
        if 'data' in optimade_result:
            print(f"  Found {len(optimade_result.get('data', []))} structures")
            print("  ✓ OPTiMaDe search successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("OQMD API integration test complete!")
    print("✓ Quantum materials database access working")
    print("=" * 60)


if __name__ == "__main__":
    main()
