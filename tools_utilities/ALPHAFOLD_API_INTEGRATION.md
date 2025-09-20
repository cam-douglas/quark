# AlphaFold Protein Structure Database API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active

## Overview

The AlphaFold Protein Structure Database API integration provides Quark with access to over 200 million AI-predicted 3D protein structures from DeepMind and EMBL-EBI. AlphaFold uses deep learning to predict protein structures from amino acid sequences with accuracy competitive with experimental methods.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API with free access (CC-BY 4.0 license)
- **200+ Million Structures**: Complete UniProt coverage
- **High Accuracy**: Competitive with experimental structure determination
- **Multiple Formats**: PDB, mmCIF, PAE JSON, CSV downloads
- **Bulk Downloads**: Up to 100 files at once

### Available Data
1. **3D Structure Files**: PDB and mmCIF format atomic coordinates
2. **Confidence Metrics**: pLDDT scores (per-residue confidence 0-100)
3. **PAE Plots**: Predicted Aligned Error visualizations
4. **TED Domains**: Domain assignments with CATH classifications
5. **AlphaMissense**: Integrated missense variant predictions

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"alphafold": {
    "service": "AlphaFold Protein Structure Database",
    "api_key": "none_required",
    "endpoints": {
        "base": "https://alphafold.ebi.ac.uk/api",
        "prediction": "https://alphafold.ebi.ac.uk/api/prediction",
        "structure_file": "https://alphafold.ebi.ac.uk/files"
    },
    "license": "CC-BY 4.0 - Free for academic and commercial use"
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/alphafold_integration.py`

### Usage Examples

```python
from tools_utilities.alphafold_integration import AlphaFoldClient

# Initialize client
client = AlphaFoldClient()

# Get structure information
info = client.get_structure_info("P04637")  # Human p53

# Download structure file
structure_path = client.download_structure(
    "P14416",  # Dopamine D2 receptor
    format="pdb",
    output_dir="data/structures"
)

# Get confidence data
confidence = client.get_confidence_data("P05067")  # Amyloid precursor

# Download PAE data
pae_path = client.download_pae("P10636")  # Tau protein

# Bulk download multiple structures
uniprot_ids = ["P14416", "P28223", "P14867"]  # Brain receptors
paths = client.bulk_download(uniprot_ids, format="pdb")
```

## Available Methods

### Core Methods
- `get_structure_info(uniprot_id)` - Get metadata for a structure
- `download_structure(uniprot_id, format, output_dir)` - Download 3D structure
- `download_pae(uniprot_id, output_dir)` - Download PAE JSON
- `get_confidence_data(uniprot_id)` - Get confidence metrics
- `bulk_download(uniprot_ids, format)` - Download multiple structures

### Confidence Categories
- **Very high (pLDDT > 90)**: Highly accurate backbone
- **Confident (70 < pLDDT < 90)**: Confident backbone prediction  
- **Low (50 < pLDDT < 70)**: Low confidence, caution needed
- **Very low (pLDDT < 50)**: Should not be interpreted

## Brain Proteins Successfully Downloaded

During testing, the following brain-related protein structures were retrieved:

### Neurotransmitter Receptors
- **P14416**: Dopamine D2 receptor
- **P28223**: Serotonin 5-HT2A receptor
- **P14867**: GABA-A receptor alpha-1
- **Q05586**: NMDA receptor subunit 1

### Ion Channels
- **P35498**: Sodium channel SCN1A
- **O43526**: Potassium channel KCNQ2
- **O00555**: Calcium channel CACNA1A

### Synaptic Proteins
- **P17600**: Synapsin-1
- **P21579**: Synaptotagmin-1
- **P60880**: SNAP-25

### Neurodegenerative Disease Proteins
- **P05067**: Amyloid precursor protein (APP)
- **P10636**: Tau protein (MAPT)
- **P37840**: Alpha-synuclein (SNCA)

## Integration with Quark

### Use Cases for Brain Simulation
1. **Structural Analysis**: Analyze 3D structures of brain proteins
2. **Drug Target Modeling**: Predict binding sites and interactions
3. **Mutation Effects**: Study structural impacts of genetic variants
4. **Protein Complexes**: Model protein-protein interactions
5. **Comparative Studies**: Compare predicted vs experimental structures

### Scientific Applications
- Neurodegenerative disease research
- Drug discovery for neurological targets
- Brain receptor structure-function studies
- Synaptic protein analysis
- Ion channel gating mechanisms

## Testing

Run the integration test:
```bash
python tools_utilities/alphafold_integration.py
```

This will:
- Fetch structure information for p53
- Download dopamine receptor structure
- Analyze amyloid precursor protein
- Download 13 key brain protein structures

## Data Storage

Downloaded structures are saved to:
```
/Users/camdouglas/quark/data/structures/alphafold/
```

Summary JSON with confidence scores:
```
/Users/camdouglas/quark/data/structures/alphafold/brain_proteins_summary.json
```

## References

### Documentation
- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [AlphaFold API Docs](https://alphafold.ebi.ac.uk/api-docs)
- [GitHub Repository](https://github.com/deepmind/alphafold)

### Publications
- Jumper et al. (2021) *Nature* - "Highly accurate protein structure prediction with AlphaFold"
- Varadi et al. (2024) *Nucleic Acids Research* - "AlphaFold Protein Structure Database in 2024"
- Cheng et al. (2023) *Science* - "Accurate proteome-wide missense variant effect prediction with AlphaMissense"

## Support

- **Contact**: alphafold@deepmind.com
- **Issues**: Contact Quark system administrators

## Notes

- Structures are predictions and should be interpreted with confidence scores
- Not all proteins have AlphaFold structures (e.g., very large proteins like full-length Huntingtin)
- Latest update (March 2025): TED domain integration and bulk downloads added
- AlphaFold 3 includes enhanced multimer prediction capabilities

## Status

✅ **Integration Complete**: API configured, Python module created, and 13 brain protein structures successfully downloaded.
