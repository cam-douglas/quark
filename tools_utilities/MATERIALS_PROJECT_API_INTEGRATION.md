# Materials Project API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active  
**API Key**: Stored securely in credentials file  
**Reference**: [Materials Project API](https://next-gen.materialsproject.org/api)

## Overview

The Materials Project API integration provides Quark with access to comprehensive computed materials data for 150,000+ inorganic compounds. This next-generation API offers crystal structures, electronic properties, mechanical properties, thermodynamic data, and more.

## Key Features

### API Capabilities
- **API Key Required**: Authentication via X-API-KEY header
- **150,000+ Materials**: Extensive inorganic compound database
- **DFT Calculations**: GGA+U with careful convergence
- **Rich Properties**: Electronic, mechanical, thermodynamic
- **Rate Limited**: 100 req/min, 5000 req/day
- **Advanced Features**: Band structures, phonons, synthesis

### Available Endpoints
1. **Materials Summary**: Basic properties and structure
2. **Electronic Structure**: Band structures, DOS
3. **Elasticity**: Elastic tensors, moduli
4. **Dielectric**: Dielectric and piezoelectric properties
5. **Magnetism**: Magnetic ordering and moments
6. **Phonons**: Phonon band structures
7. **Thermodynamics**: Formation energies, phase diagrams
8. **Surfaces**: Surface energies and structures
9. **Synthesis**: Text-mined synthesis recipes
10. **XAS Spectra**: X-ray absorption spectra

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"materials_project": {
    "service": "Materials Project API",
    "endpoints": {
        "base": "https://api.materialsproject.org",
        "materials": ".../materials",
        "summary": ".../materials/summary",
        "electronic_structure": ".../materials/electronic_structure",
        "elasticity": ".../materials/elasticity"
    },
    "api_key": "O0oXYcZo6YumgKUOzDJCx9mFiAk9pP4l",
    "rate_limits": {
        "requests_per_minute": 100,
        "requests_per_day": 5000
    }
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/materials_project_integration.py`

### Usage Examples

```python
from tools_utilities.materials_project_integration import MaterialsProjectClient

# Initialize with API key
client = MaterialsProjectClient()

# Search materials
materials = client.search_materials(
    elements=['Li', 'Fe', 'O'],
    band_gap=(1.0, 3.0),
    energy_above_hull=(0, 0.05),
    limit=100
)

# Get specific material
material = client.get_material_by_id(
    'mp-149',  # Silicon
    fields=['formula_pretty', 'band_gap', 'crystal_system']
)

# Get band structure
bands = client.get_bandstructure('mp-149')

# Get elastic properties
elastic = client.get_elasticity('mp-149')

# Get synthesis information
synthesis = client.get_synthesis_recipe('mp-149')
```

## Brain-Relevant Materials Retrieved

Successfully identified 46 materials relevant to brain/neural research:

### Categories Found
- **Ionic Conductors** (15): Li/Na/K compounds for ion transport
- **Semiconductors** (15): 1-3 eV band gap materials
- **Magnetic Materials** (6): Fe-based compounds with magnetization
- **Dielectric Materials** (10): High band gap insulators

### Example Materials

| Category | Formula | Material ID | Band Gap (eV) | Energy Above Hull |
|----------|---------|-------------|---------------|-------------------|
| Ionic | LiMn₂O₄ | mp-1097058 | 0.348 | 0.020 |
| Semiconductor | Ac₂S₃ | mp-1244983 | 2.296 | 0.000 |
| Magnetic | Fe₂O₃ | mp-19770 | 2.140 | 0.000 |
| Dielectric | HfO₂ | mp-352 | 5.680 | 0.000 |

## Search Capabilities

### Query Parameters
- **Elements**: Filter by chemical elements
- **Formula**: Exact chemical formula
- **Chemical System**: e.g., "Li-Fe-O"
- **Crystal System**: Cubic, Tetragonal, etc.
- **Space Group**: Crystallographic space group
- **Band Gap Range**: Electronic band gap (eV)
- **Energy Above Hull**: Thermodynamic stability
- **Formation Energy**: Per-atom formation energy
- **Properties**: Elastic, magnetic, dielectric

### Advanced Features
```python
# Complex searches
semiconductors = client.search_materials(
    band_gap=(1.5, 2.5),  # Solar cell range
    crystal_system='Cubic',
    energy_above_hull=(0, 0.01),  # Stable only
    elements=['O']  # Oxides
)

# Property retrieval
for mat_id in material_ids:
    bands = client.get_bandstructure(mat_id)
    elastic = client.get_elasticity(mat_id)
```

## Data Quality

### Calculation Standards
- **Method**: GGA+U with optimized U values
- **Convergence**: Strict criteria for accuracy
- **Validation**: Benchmarked against experiments
- **Corrections**: Advanced schemes for known DFT errors
- **Provenance**: Full calculation history tracked

## Integration with Quark

### Use Cases for Brain Simulation
1. **Neural Interface Materials**: Biocompatible semiconductors
2. **Ion Channel Models**: Ionic conductor properties
3. **Magnetic Memory**: Spintronic materials
4. **Dielectric Barriers**: Insulating materials
5. **Energy Storage**: Battery materials for implants

### Scientific Applications
- Materials discovery and design
- Property prediction validation
- Structure-property relationships
- Phase stability analysis
- Synthesis planning

## Data Storage

Generated data saved to:
- `/data/knowledge/materials_project_brain.json`

## Testing

Run the integration test:
```bash
python tools_utilities/materials_project_integration.py
```

## Python Client

Official Python client available:
```bash
pip install mp-api
```

Our integration provides a custom lightweight client optimized for Quark's needs.

## Rate Limits

- **100 requests per minute**: Enforced automatically
- **5000 requests per day**: Per API key
- **Rate limiting**: Built into client with 0.6s minimum interval

## References

### Documentation
- [Materials Project API Docs](https://next-gen.materialsproject.org/api)
- [GitHub Repository](https://github.com/materialsproject/api)
- [Materials Explorer](https://next-gen.materialsproject.org)

### Citation
- Jain et al. (2013) *APL Materials* - "Commentary: The Materials Project"
- Ong et al. (2015) *Comp. Mater. Sci.* - "The Materials Application Programming Interface"

### Support
- Email: support@materialsproject.org
- [Terms of Use](https://next-gen.materialsproject.org/terms)

## Notes

- API key is required for all requests
- Data is regularly updated with new calculations
- Supports both REST and Python API access
- Integrates with pymatgen for advanced analysis

## Status

✅ **Integration Complete**: API configured with authentication, tested successfully, and 46 brain-relevant materials retrieved from 150,000+ material database.
