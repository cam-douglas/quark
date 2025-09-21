# PubChem PUG-REST API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active

## Overview

The PubChem PUG-REST API integration provides Quark with access to the world's largest free chemistry database. PubChem contains information on 100+ million chemical compounds, substances, bioassays, and their biological activities.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API with rate limiting
- **5 req/sec Rate Limit**: Up to 400 requests per minute
- **Chemical Search**: By name, SMILES, InChI, formula
- **Structure Operations**: Similarity and substructure searches
- **Bioactivity Data**: Access to bioassay results
- **Image Generation**: 2D structure images in PNG format

### Available Services
1. **Compound Search**: Find chemicals by various identifiers
2. **Property Retrieval**: Get molecular properties and descriptors
3. **Structure Search**: Similarity and substructure matching
4. **Bioassay Data**: Biological activity information
5. **Cross-References**: Links to other databases
6. **Image Download**: 2D molecular structure images

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"pubchem": {
    "service": "PubChem PUG-REST",
    "endpoints": {
        "base": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
        "compound": ".../compound",
        "bioassay": ".../bioassay"
    },
    "api_key": "none_required",
    "rate_limits": {
        "requests_per_second": "5 req/sec"
    }
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/pubchem_integration.py`

### Usage Examples

```python
from tools_utilities.pubchem_integration import PubChemClient

# Initialize client
client = PubChemClient()

# Search for compound by name
compounds = client.search_compound_by_name('aspirin')

# Get molecular properties
props = client.get_compound_properties(
    'dopamine',
    ['MolecularFormula', 'MolecularWeight', 'IUPACName']
)

# Search by SMILES structure
similar = client.search_by_smiles('CC(C)CC1=CC=CC=C1', 'similarity', 95)

# Download structure image
client.get_compound_image(681, output_path='dopamine.png')  # CID 681 = dopamine

# Search neurotransmitters
neurotransmitters = client.search_neurotransmitters()
```

## Neurotransmitters Successfully Retrieved

During testing, retrieved chemical data for 10 neurotransmitters:

| Compound | CID | Formula | MW (g/mol) | LogP |
|----------|-----|---------|------------|------|
| Dopamine | 681 | C8H11NO2 | 153.18 | -1.0 |
| Serotonin | 5202 | C10H12N2O | 176.21 | 0.2 |
| GABA | 119 | C4H9NO2 | 103.12 | -3.2 |
| Glutamate | 33032 | C5H9NO4 | 147.13 | -3.7 |
| Acetylcholine | 187 | C7H16NO2+ | 146.21 | -4.4 |
| Norepinephrine | 439260 | C8H11NO3 | 169.18 | -1.2 |
| Epinephrine | 5816 | C9H13NO3 | 183.20 | -1.2 |
| Histamine | 774 | C5H9N3 | 111.15 | -0.7 |
| Glycine | 750 | C2H5NO2 | 75.07 | -3.2 |
| Adenosine | 60961 | C10H13N5O4 | 267.24 | -1.1 |

## Neurological Drugs Retrieved

Successfully retrieved data for drugs used in neurological conditions:

- **Levodopa** (Parkinson's): CID 6047, C9H11NO4
- **Donepezil** (Alzheimer's): CID 3152, C24H29NO3
- **Fluoxetine** (Depression): CID 3386, C17H18F3NO
- **Diazepam** (Anxiety): CID 3016, C16H13ClN2O
- **Carbamazepine** (Epilepsy): CID 2554, C15H12N2O
- **Methylphenidate** (ADHD): CID 4158, C14H19NO2

## Integration with Quark

### Use Cases for Brain Simulation
1. **Neurotransmitter Properties**: Chemical data for signaling molecules
2. **Drug-Target Analysis**: Study drug interactions
3. **Metabolite Information**: Brain metabolites and pathways
4. **Toxicology Data**: Neurotoxin information
5. **Structure-Activity**: Relationship analysis

### Scientific Applications
- Drug discovery and design
- Pharmacophore modeling
- ADMET predictions
- Chemical similarity analysis
- Bioactivity profiling

## Data Storage

Generated data saved to:
- `/data/knowledge/pubchem_neurotransmitters.json`
- `/data/structures/pubchem/` (structure images)

## Testing

Run the integration test:
```bash
python tools_utilities/pubchem_integration.py
```

## Rate Limits

- Maximum 5 requests per second
- 400 requests per minute limit
- Automatic rate limiting in client
- Violators may be temporarily blocked

## References

### Documentation
- [PubChem PUG-REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
- [Tutorial](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial)

### Citation
- Kim et al. (2023) *Nucleic Acids Research* - "PubChem 2023 update"

### Support
- Email: pubchem-help@ncbi.nlm.nih.gov

## Status

✅ **Integration Complete**: API configured, tested, and neurotransmitter/drug data successfully retrieved.
