# RCSB PDB API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active

## Overview

The RCSB Protein Data Bank (PDB) API integration provides Quark with access to the world's largest repository of 3D structural data for biological macromolecules. This integration enables searching, retrieving, and analyzing protein structures, including experimental structures and computed models.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API with unrestricted access
- **Multiple Search Types**: Text, sequence, structure, chemical, motif searches
- **Comprehensive Data**: Access to experimental and computed structure models
- **Python SDK**: Official `rcsb-api` package installed

### Search Services Available
1. **Text Search**: Search by keywords, organism, experimental method
2. **Sequence Search**: Find structures with similar protein sequences
3. **Structure Search**: 3D shape similarity searches
4. **Chemical Search**: Find structures with specific compounds/ligands
5. **Sequence Motif**: Search for specific sequence patterns
6. **Structure Motif**: Search for specific 3D arrangements

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"rcsb_pdb": {
    "service": "RCSB Protein Data Bank",
    "api_key": "none_required",
    "endpoints": {
        "search": "https://search.rcsb.org/rcsbsearch/v2/query",
        "data": "https://data.rcsb.org",
        "graphql": "https://data.rcsb.org/graphql"
    }
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/rcsb_pdb_integration.py`

### Usage Examples

```python
from tools_utilities.rcsb_pdb_integration import RCSBPDBClient

# Initialize client
client = RCSBPDBClient()

# Search for COVID-19 spike proteins
results = client.search_text("COVID-19 spike protein", rows=10)

# Search for human proteins with high resolution
human_proteins = client.search_by_organism(
    "Homo sapiens",
    experimental_method="X-RAY DIFFRACTION",
    resolution_max=2.0
)

# Get recent structures
recent = client.search_recent_structures(days=7)

# Get detailed structure data
structure = client.get_structure_data("4HHB")  # Hemoglobin
```

## Available Methods

### Core Search Methods
- `search_text(query, return_type, rows)` - Text-based search
- `search_sequence(sequence, identity_threshold, rows)` - Sequence similarity search
- `search_by_organism(organism, experimental_method, resolution_max)` - Organism-specific search
- `search_chemical_similarity(smiles, similarity_threshold)` - Chemical compound search
- `search_membrane_proteins(rows)` - Find membrane proteins
- `search_recent_structures(days, rows)` - Recently released structures
- `get_structure_data(pdb_id)` - Get detailed data for specific PDB entry

## Integration with Quark

### Use Cases for Brain Simulation
1. **Protein Structure Analysis**: Analyze neurotransmitter receptors and ion channels
2. **Drug Target Research**: Find structures of potential therapeutic targets
3. **Comparative Analysis**: Compare brain protein structures across species
4. **Structural Validation**: Validate predicted structures against experimental data

### Scientific Research Applications
- Structural biology research
- Drug discovery and design
- Protein engineering
- Evolutionary studies
- Disease mechanism understanding

## Testing

Run the integration test:
```bash
python tools_utilities/rcsb_pdb_integration.py
```

## Dependencies

- `requests`: HTTP client for API calls
- `rcsb-api`: Official Python SDK (installed)
- `typing_extensions`: Type hints support

## References

### API Documentation
- [RCSB PDB Search API](https://search.rcsb.org/index.html#search-api)
- [RCSB PDB Data API](https://data.rcsb.org)
- [Python SDK Documentation](https://github.com/rcsb/py-rcsb-api)

### Publications
- Rose et al. (2021) *J Mol Biol* DOI: 10.1016/j.jmb.2020.11.003
- Bittrich et al. (2023) *J Mol Biol* DOI: 10.1016/j.jmb.2023.167994

## Support

- **Mailing List**: api+subscribe@rcsb.org
- **Contact**: info@rcsb.org
- **Issues**: Contact Quark system administrators

## Status

✅ **Integration Complete**: API configured, Python module created, and tested successfully.
