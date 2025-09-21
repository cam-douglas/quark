# OQMD RESTful API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active  
**Reference**: [OQMD Documentation](https://static.oqmd.org/static/docs/restful.html)

## Overview

The Open Quantum Materials Database (OQMD) API integration provides Quark with access to DFT-calculated thermodynamic and structural properties for approximately 700,000 materials. OQMD is one of the largest open databases of computed materials properties.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API
- **700,000+ Materials**: Extensive database coverage
- **DFT Properties**: Formation energies, band gaps, stability
- **Complex Filtering**: AND/OR logical expressions
- **OPTiMaDe Compatible**: v0.9.5 specification support
- **Multiple Output Formats**: JSON, XML, YAML

### Available Services
1. **Formation Energy Queries**: Thermodynamic stability data
2. **Band Gap Searches**: Electronic properties
3. **Structure Prototypes**: Crystal structure classification
4. **Element Filters**: Composition-based searches
5. **Stability Analysis**: Hull distance calculations
6. **ICSD Integration**: Links to experimental structures

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"oqmd": {
    "service": "OQMD RESTful API",
    "endpoints": {
        "base": "http://oqmd.org/oqmdapi",
        "formationenergy": "http://oqmd.org/oqmdapi/formationenergy",
        "optimade": "http://oqmd.org/optimade"
    },
    "api_key": "none_required",
    "database_size": "~700,000 materials"
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/oqmd_integration.py`

### Usage Examples

```python
from tools_utilities.oqmd_integration import OQMDClient

# Initialize client
client = OQMDClient()

# Search materials with filters
results = client.search_materials(
    filter_expr='stability=0 AND band_gap>2',
    fields=['name', 'entry_id', 'delta_e', 'band_gap'],
    limit=100
)

# Get stable materials
stable = client.get_stable_materials(
    element_set='(Fe-Mn),O',
    limit=50
)

# Search by band gap
semiconductors = client.search_by_bandgap(
    min_gap=1.5, 
    max_gap=3.0
)

# Search by structure prototype
perovskites = client.search_by_prototype('CaTiO3')

# OPTiMaDe-compatible search
optimade_results = client.optimade_search(
    filter_expr='elements HAS "Li" AND nelements=3'
)
```

## Brain-Relevant Materials Retrieved

Successfully identified 30 materials relevant to brain/neural research:

### Categories Found
- **Semiconductors** (10): Neural-relevant band gaps (0.5-3 eV)
- **Ionic Conductors** (10): Li/Na/K compounds for ion transport
- **Magnetic Materials** (10): Fe/Co/Ni oxides

### Example Materials

| Material | Type | Band Gap (eV) | Formation Energy (eV/atom) |
|----------|------|---------------|---------------------------|
| YbSe | Semiconductor | 2.083 | -2.106 |
| SiC | Semiconductor | 2.625 | -0.367 |
| Li2O | Ionic | 6.014 | -3.025 |
| Fe2O3 | Magnetic | 2.331 | -1.687 |

## Query Capabilities

### Filterable Fields
- `composition`: Chemical formula (e.g., "Al2O3")
- `element_set`: Element combinations (e.g., "(Fe-Mn),O")
- `stability`: Hull distance (0 = stable)
- `delta_e`: Formation energy
- `band_gap`: Electronic band gap
- `spacegroup`: Crystallographic space group
- `prototype`: Structure type
- `natoms`: Number of atoms
- `volume`: Unit cell volume
- `ntypes`: Number of element types

### Filter Syntax
```
# Complex logical expressions
filter='stability=0 AND band_gap>1 AND band_gap<5'
filter='element_set=(Al-Fe),O AND (ntypes>=3 OR stability<0.1)'
filter='spacegroup="Fm-3m" AND element=O'
```

## Integration with Quark

### Use Cases for Brain Simulation
1. **Semiconductor Selection**: Materials for neural interfaces
2. **Ion Conductor Design**: Artificial synaptic materials
3. **Magnetic Materials**: Neuromorphic memory devices
4. **Bandgap Engineering**: Optoelectronic neural components
5. **Stability Analysis**: Biocompatible materials

### Scientific Applications
- Materials discovery
- High-throughput screening
- Property prediction validation
- Phase diagram construction
- Structure-property relationships

## Data Storage

Generated data saved to:
- `/data/knowledge/oqmd_brain_materials.json`

## Testing

Run the integration test:
```bash
python tools_utilities/oqmd_integration.py
```

## OPTiMaDe Compatibility

OQMD supports the OPTiMaDe specification v0.9.5:
- Endpoint: `http://oqmd.org/optimade`
- Standardized query format
- Interoperable with other materials databases

## References

### Documentation
- [OQMD REST API](https://static.oqmd.org/static/docs/restful.html)
- [qmpy Python Package](https://github.com/wolverton-research-group/qmpy)

### Citation
- Saal et al. (2013) *JOM* - "Materials Database Infrastructure"
- Kirklin et al. (2015) *npj Comput. Mater.* - "The Open Quantum Materials Database"

### Support
- Mailing list: api+subscribe@rcsb.org
- Website: http://oqmd.org

## Status

✅ **Integration Complete**: API configured, tested, and 30 brain-relevant materials retrieved from 700,000+ material database.
