#!/usr/bin/env python3
"""
RCSB PDB Brain Protein Research Example
========================================
This example demonstrates using the RCSB PDB API to search for 
brain-related proteins and neurotransmitter receptors.

Author: Quark System
Date: 2025-01-20
"""

import json
from pathlib import Path
from typing import Dict, List

from rcsb_pdb_integration import RCSBPDBClient


def search_neurotransmitter_receptors(client: RCSBPDBClient) -> Dict[str, List]:
    """
    Search for various neurotransmitter receptor structures.
    
    Args:
        client: RCSB PDB client instance
        
    Returns:
        Dictionary of receptor types and their PDB IDs
    """
    receptors = {
        "dopamine": [],
        "serotonin": [],
        "GABA": [],
        "glutamate": [],
        "acetylcholine": []
    }
    
    print("\nSearching for neurotransmitter receptor structures...")
    print("-" * 50)
    
    for receptor_type in receptors:
        query = f"{receptor_type} receptor human"
        results = client.search_text(query, rows=10)
        
        if results.get('result_set'):
            for entry in results['result_set']:
                receptors[receptor_type].append(entry['identifier'])
            
            count = results.get('total_count', 0)
            print(f"{receptor_type.capitalize()} receptors: {count} total structures")
            print(f"  Sample PDB IDs: {', '.join(receptors[receptor_type][:3])}")
    
    return receptors


def search_ion_channels(client: RCSBPDBClient) -> Dict[str, List]:
    """
    Search for ion channel structures important in neural signaling.
    
    Args:
        client: RCSB PDB client instance
        
    Returns:
        Dictionary of ion channel types and their PDB IDs
    """
    channels = {
        "sodium channel": [],
        "potassium channel": [],
        "calcium channel": [],
        "chloride channel": []
    }
    
    print("\nSearching for ion channel structures...")
    print("-" * 50)
    
    for channel_type in channels:
        query = f"{channel_type} human"
        results = client.search_text(query, rows=10)
        
        if results.get('result_set'):
            for entry in results['result_set']:
                channels[channel_type].append(entry['identifier'])
            
            count = results.get('total_count', 0)
            print(f"{channel_type.capitalize()}: {count} total structures")
            if channels[channel_type]:
                print(f"  Sample PDB IDs: {', '.join(channels[channel_type][:3])}")
    
    return channels


def search_brain_disease_proteins(client: RCSBPDBClient) -> Dict[str, List]:
    """
    Search for proteins associated with neurological diseases.
    
    Args:
        client: RCSB PDB client instance
        
    Returns:
        Dictionary of disease-related proteins and their PDB IDs
    """
    disease_proteins = {
        "amyloid beta": [],  # Alzheimer's
        "tau protein": [],   # Alzheimer's
        "alpha synuclein": [],  # Parkinson's
        "huntingtin": [],    # Huntington's
        "prion protein": []  # Prion diseases
    }
    
    print("\nSearching for neurological disease-related proteins...")
    print("-" * 50)
    
    for protein in disease_proteins:
        results = client.search_text(protein, rows=10)
        
        if results.get('result_set'):
            for entry in results['result_set']:
                disease_proteins[protein].append(entry['identifier'])
            
            count = results.get('total_count', 0)
            print(f"{protein.capitalize()}: {count} total structures")
            if disease_proteins[protein]:
                print(f"  Sample PDB IDs: {', '.join(disease_proteins[protein][:3])}")
                
                # Get details for the first structure
                if disease_proteins[protein]:
                    try:
                        details = client.get_structure_data(disease_proteins[protein][0])
                        if details:
                            title = details.get('struct', {}).get('title', 'N/A')
                            print(f"  Example structure: {title[:80]}...")
                    except Exception:
                        pass
    
    return disease_proteins


def analyze_brain_protein_statistics(client: RCSBPDBClient) -> None:
    """
    Analyze statistics about brain-related protein structures.
    
    Args:
        client: RCSB PDB client instance
    """
    print("\nAnalyzing brain protein structure statistics...")
    print("-" * 50)
    
    # Search for brain-specific terms
    brain_terms = [
        "brain",
        "neural",
        "neuronal",
        "synapse",
        "synaptic",
        "neurotransmitter"
    ]
    
    total_brain_structures = 0
    
    for term in brain_terms:
        results = client.search_text(term, rows=1)
        count = results.get('total_count', 0)
        total_brain_structures += count
        print(f"Structures with '{term}': {count:,}")
    
    print(f"\nEstimated total brain-related structures: {total_brain_structures:,}")
    print("(Note: Some structures may be counted multiple times)")


def save_results(receptors: Dict, channels: Dict, disease_proteins: Dict) -> None:
    """
    Save search results to a JSON file.
    
    Args:
        receptors: Neurotransmitter receptor results
        channels: Ion channel results
        disease_proteins: Disease protein results
    """
    output_path = Path(__file__).parent.parent / "data" / "knowledge" / "brain_proteins_pdb.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "metadata": {
            "source": "RCSB PDB",
            "date": "2025-01-20",
            "description": "Brain-related protein structures from PDB"
        },
        "neurotransmitter_receptors": receptors,
        "ion_channels": channels,
        "neurological_disease_proteins": disease_proteins
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run brain protein searches."""
    print("=" * 60)
    print("RCSB PDB Brain Protein Research")
    print("Quark System Integration")
    print("=" * 60)
    
    # Initialize client
    client = RCSBPDBClient()
    
    # Search for different categories of brain proteins
    receptors = search_neurotransmitter_receptors(client)
    channels = search_ion_channels(client)
    disease_proteins = search_brain_disease_proteins(client)
    
    # Analyze statistics
    analyze_brain_protein_statistics(client)
    
    # Save results
    save_results(receptors, channels, disease_proteins)
    
    print("\n" + "=" * 60)
    print("Brain protein research complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
