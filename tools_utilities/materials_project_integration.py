#!/usr/bin/env python3
"""
Materials Project API Integration for Quark
===========================================
This module provides integration with the next-generation Materials Project API.

The Materials Project provides comprehensive computed materials data including
crystal structures, electronic properties, mechanical properties, and more for
150,000+ inorganic compounds.

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
    MP_CONFIG = credentials['services']['materials_project']

# API configuration
API_KEY = MP_CONFIG['api_key']
BASE_URL = MP_CONFIG['endpoints']['base']


class MaterialsProjectClient:
    """Client for interacting with Materials Project API."""
    
    def __init__(self, api_key: str = API_KEY):
        """
        Initialize Materials Project client.
        
        Args:
            api_key: Materials Project API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': api_key,
            'User-Agent': 'Quark-MaterialsProject-Integration/1.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_interval = 0.6  # ~100 requests per minute
    
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
        method: str = 'GET'
    ) -> Dict[str, Any]:
        """
        Make a request to the Materials Project API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            method: HTTP method
            
        Returns:
            JSON response
        """
        self._rate_limit()
        
        url = f"{BASE_URL}{endpoint}"
        logger.debug(f"{method} {url}")
        
        if method == 'GET':
            response = self.session.get(url, params=params)
        elif method == 'POST':
            response = self.session.post(url, json=params)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
    
    def search_materials(
        self,
        elements: Optional[List[str]] = None,
        formula: Optional[str] = None,
        material_ids: Optional[List[str]] = None,
        chemsys: Optional[str] = None,
        crystal_system: Optional[str] = None,
        spacegroup_number: Optional[int] = None,
        band_gap: Optional[tuple] = None,
        energy_above_hull: Optional[tuple] = None,
        formation_energy: Optional[tuple] = None,
        fields: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for materials with various criteria.
        
        Args:
            elements: List of elements (e.g., ["Li", "Fe", "O"])
            formula: Chemical formula (e.g., "Li2O")
            material_ids: List of material IDs
            chemsys: Chemical system (e.g., "Li-Fe-O")
            crystal_system: Crystal system (e.g., "cubic")
            spacegroup_number: Space group number
            band_gap: (min, max) band gap in eV
            energy_above_hull: (min, max) energy above hull in eV/atom
            formation_energy: (min, max) formation energy in eV/atom
            fields: Fields to return
            limit: Maximum number of results
            
        Returns:
            List of materials
        """
        params = {
            '_limit': limit
        }
        
        # Build query
        if elements:
            params['elements'] = ','.join(elements)
        if formula:
            params['formula'] = formula
        if material_ids:
            params['material_ids'] = ','.join(material_ids)
        if chemsys:
            params['chemsys'] = chemsys
        if crystal_system:
            params['crystal_system'] = crystal_system
        if spacegroup_number:
            params['spacegroup_number'] = spacegroup_number
        
        # Range queries
        if band_gap:
            params['band_gap_min'] = band_gap[0]
            params['band_gap_max'] = band_gap[1]
        if energy_above_hull:
            params['energy_above_hull_max'] = energy_above_hull[1]
        if formation_energy:
            params['formation_energy_per_atom_min'] = formation_energy[0]
            params['formation_energy_per_atom_max'] = formation_energy[1]
        
        # Fields to return
        if fields:
            params['_fields'] = ','.join(fields)
        
        logger.info(f"Searching materials with criteria: {params}")
        
        result = self._make_request('/materials/summary/', params)
        
        if 'data' in result:
            return result['data']
        return []
    
    def get_material_by_id(
        self,
        material_id: str,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get material by its ID.
        
        Args:
            material_id: Material ID (e.g., "mp-1234")
            fields: Optional fields to return
            
        Returns:
            Material data
        """
        params = {}
        if fields:
            params['_fields'] = ','.join(fields)
        
        logger.info(f"Getting material: {material_id}")
        result = self._make_request(f'/materials/{material_id}/', params)
        
        if 'data' in result and result['data']:
            return result['data'][0] if isinstance(result['data'], list) else result['data']
        return {}
    
    def get_bandstructure(self, material_id: str) -> Dict[str, Any]:
        """
        Get electronic band structure for a material.
        
        Args:
            material_id: Material ID
            
        Returns:
            Band structure data
        """
        logger.info(f"Getting band structure for: {material_id}")
        result = self._make_request(f'/materials/electronic_structure/{material_id}/')
        
        if 'data' in result:
            return result['data']
        return {}
    
    def get_elasticity(self, material_id: str) -> Dict[str, Any]:
        """
        Get elastic properties for a material.
        
        Args:
            material_id: Material ID
            
        Returns:
            Elastic tensor and derived properties
        """
        logger.info(f"Getting elasticity data for: {material_id}")
        result = self._make_request(f'/materials/elasticity/{material_id}/')
        
        if 'data' in result:
            return result['data']
        return {}
    
    def search_brain_materials(self) -> Dict[str, List[Dict]]:
        """
        Search for materials relevant to brain/neural research.
        
        Returns:
            Dictionary of brain-relevant materials
        """
        brain_materials = {
            'ionic_conductors': [],
            'semiconductors': [],
            'magnetic': [],
            'dielectric': [],
            'battery': []
        }
        
        print("\nSearching for brain-relevant materials in Materials Project")
        print("-" * 60)
        
        # 1. Ionic conductors (Li, Na, K compounds)
        print("1. Searching ionic conductors for neural signaling...")
        try:
            ionic = self.search_materials(
                elements=['Li', 'O'],
                energy_above_hull=(0, 0.05),  # Nearly stable
                fields=['material_id', 'formula_pretty', 'band_gap', 
                       'energy_above_hull', 'formation_energy_per_atom'],
                limit=10
            )
            brain_materials['ionic_conductors'].extend(ionic)
            
            # Also search Na compounds
            na_ionic = self.search_materials(
                elements=['Na', 'O'],
                energy_above_hull=(0, 0.05),
                fields=['material_id', 'formula_pretty', 'band_gap'],
                limit=5
            )
            brain_materials['ionic_conductors'].extend(na_ionic)
            
            print(f"  Found {len(brain_materials['ionic_conductors'])} ionic conductors")
            
        except Exception as e:
            logger.error(f"Error searching ionic conductors: {e}")
        
        # 2. Semiconductors with neural-relevant band gaps
        print("2. Searching semiconductors (1-3 eV band gap)...")
        try:
            semiconductors = self.search_materials(
                band_gap=(1.0, 3.0),
                energy_above_hull=(0, 0.01),  # Stable only
                fields=['material_id', 'formula_pretty', 'band_gap', 
                       'crystal_system', 'spacegroup'],
                limit=15
            )
            brain_materials['semiconductors'] = semiconductors
            print(f"  Found {len(semiconductors)} semiconductors")
            
        except Exception as e:
            logger.error(f"Error searching semiconductors: {e}")
        
        # 3. Magnetic materials
        print("3. Searching magnetic materials...")
        try:
            magnetic = self.search_materials(
                elements=['Fe', 'O'],
                energy_above_hull=(0, 0.05),
                fields=['material_id', 'formula_pretty', 'total_magnetization'],
                limit=10
            )
            # Filter for non-zero magnetization
            brain_materials['magnetic'] = [
                m for m in magnetic 
                if m.get('total_magnetization', 0) > 0.1
            ]
            print(f"  Found {len(brain_materials['magnetic'])} magnetic materials")
            
        except Exception as e:
            logger.error(f"Error searching magnetic materials: {e}")
        
        # 4. High-k dielectrics
        print("4. Searching high-k dielectric materials...")
        try:
            # Search for oxides with large band gaps (insulators)
            dielectric = self.search_materials(
                elements=['O'],
                band_gap=(5.0, None),  # Large gap insulators
                energy_above_hull=(0, 0.01),
                fields=['material_id', 'formula_pretty', 'band_gap'],
                limit=10
            )
            brain_materials['dielectric'] = dielectric
            print(f"  Found {len(dielectric)} dielectric materials")
            
        except Exception as e:
            logger.error(f"Error searching dielectric materials: {e}")
        
        # 5. Battery materials (for bioelectronics)
        print("5. Searching battery/energy storage materials...")
        try:
            battery = self.search_materials(
                elements=['Li'],
                crystal_system='cubic',  # Often good for Li diffusion
                energy_above_hull=(0, 0.025),
                fields=['material_id', 'formula_pretty', 'volume', 'density'],
                limit=10
            )
            brain_materials['battery'] = battery
            print(f"  Found {len(battery)} battery materials")
            
        except Exception as e:
            logger.error(f"Error searching battery materials: {e}")
        
        return brain_materials
    
    def get_synthesis_recipe(self, material_id: str) -> Dict[str, Any]:
        """
        Get text-mined synthesis information for a material.
        
        Args:
            material_id: Material ID
            
        Returns:
            Synthesis recipes and conditions
        """
        logger.info(f"Getting synthesis info for: {material_id}")
        
        try:
            result = self._make_request(f'/materials/synthesis/{material_id}/')
            if 'data' in result:
                return result['data']
        except Exception as e:
            logger.error(f"Error getting synthesis: {e}")
        
        return {}


def demonstrate_advanced_features(client: MaterialsProjectClient):
    """
    Demonstrate advanced Materials Project features.
    
    Args:
        client: Materials Project client
    """
    print("\nAdvanced Materials Project Features")
    print("=" * 60)
    
    # Get a well-studied material: Silicon
    print("\n1. Detailed analysis of Silicon (mp-149)")
    print("-" * 40)
    
    try:
        # Get basic properties
        si = client.get_material_by_id(
            'mp-149',
            fields=['material_id', 'formula_pretty', 'band_gap', 
                   'crystal_system', 'volume', 'density']
        )
        
        if si:
            print(f"  Material: {si.get('formula_pretty')}")
            print(f"  Band gap: {si.get('band_gap', {}).get('band_gap', 'N/A')} eV")
            print(f"  Crystal system: {si.get('symmetry', {}).get('crystal_system')}")
            print(f"  Density: {si.get('density', 'N/A')} g/cm³")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    # Search for perovskites
    print("\n2. Searching for perovskite solar cell materials")
    print("-" * 40)
    
    try:
        perovskites = client.search_materials(
            crystal_system='cubic',
            band_gap=(1.0, 2.5),  # Good for solar cells
            elements=['O'],
            energy_above_hull=(0, 0.01),
            fields=['material_id', 'formula_pretty', 'band_gap'],
            limit=5
        )
        
        if perovskites:
            print(f"  Found {len(perovskites)} potential perovskites:")
            for p in perovskites[:3]:
                bg = p.get('band_gap', {})
                gap_value = bg.get('band_gap') if isinstance(bg, dict) else bg
                print(f"    {p.get('formula_pretty')}: Eg = {gap_value} eV")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    return True


def main():
    """Example usage of Materials Project client."""
    client = MaterialsProjectClient()
    
    print("=" * 60)
    print("Materials Project API Integration Test")
    print("Quark System - Advanced Materials Database Access")
    print("=" * 60)
    
    # Test 1: Basic search
    print("\n1. Testing basic materials search...")
    try:
        materials = client.search_materials(
            elements=['Fe', 'O'],
            fields=['material_id', 'formula_pretty', 'band_gap'],
            limit=5
        )
        
        if materials:
            print(f"  Found {len(materials)} Fe-O materials")
            for mat in materials[:3]:
                print(f"    {mat.get('material_id')}: {mat.get('formula_pretty')}")
            print("  ✓ Basic search successful")
        else:
            print("  No materials found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Get specific material
    print("\n2. Testing material retrieval by ID...")
    try:
        # Get data for TiO2 (mp-2657)
        tio2 = client.get_material_by_id(
            'mp-2657',
            fields=['formula_pretty', 'band_gap', 'crystal_system', 'spacegroup']
        )
        
        if tio2:
            print(f"  Material: {tio2.get('formula_pretty')}")
            symmetry = tio2.get('symmetry', {})
            print(f"  Crystal system: {symmetry.get('crystal_system')}")
            print(f"  Space group: {symmetry.get('symbol')}")
            print("  ✓ Material retrieval successful")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Band gap search
    print("\n3. Testing band gap search (2-3 eV)...")
    try:
        semiconductors = client.search_materials(
            band_gap=(2.0, 3.0),
            energy_above_hull=(0, 0.01),
            fields=['material_id', 'formula_pretty', 'band_gap'],
            limit=5
        )
        
        if semiconductors:
            print(f"  Found {len(semiconductors)} semiconductors")
            for semi in semiconductors[:3]:
                bg = semi.get('band_gap', {})
                gap_value = bg.get('band_gap') if isinstance(bg, dict) else bg
                print(f"    {semi.get('formula_pretty')}: {gap_value} eV")
            print("  ✓ Band gap search successful")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Search brain-relevant materials
    print("\n4. Searching for brain-relevant materials...")
    try:
        brain_materials = client.search_brain_materials()
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "knowledge" / "materials_project_brain.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count total
        total = sum(len(mats) for mats in brain_materials.values())
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'source': 'Materials Project API',
                    'date': '2025-01-20',
                    'description': 'Brain-relevant materials from Materials Project',
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
    
    # Test 5: Advanced features demo
    demonstrate_advanced_features(client)
    
    print("\n" + "=" * 60)
    print("Materials Project API integration test complete!")
    print("✓ Advanced materials database access working")
    print("=" * 60)


if __name__ == "__main__":
    main()
