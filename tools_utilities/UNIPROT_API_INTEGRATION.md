# UniProt REST API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active  
**Reference**: [UniProt API Documentation](https://www.uniprot.org/help/programmatic_access)

## Overview

The UniProt REST API integration provides Quark with access to the Universal Protein Resource, the world's most comprehensive resource for protein sequence and functional information. UniProt contains 250+ million protein sequences with extensive annotations.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API
- **250+ Million Sequences**: Comprehensive protein database
- **Rich Annotations**: Function, structure, pathways, diseases
- **Cross-References**: Links to 180+ databases
- **Multiple Formats**: JSON, XML, FASTA, TSV, GFF
- **ID Mapping**: Convert between database identifiers

### Available Services
1. **Protein Search**: Query UniProtKB with Lucene syntax
2. **Entry Retrieval**: Get specific proteins by accession
3. **Sequence Access**: FASTA sequences and variants
4. **ID Mapping**: Convert between UniProt, PDB, RefSeq, etc.
5. **Proteomes**: Complete proteome sets
6. **Taxonomy**: Taxonomic classification
7. **Keywords & GO**: Functional annotations

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"uniprot": {
    "service": "UniProt REST API",
    "endpoints": {
        "base": "https://rest.uniprot.org",
        "uniprotkb": "https://rest.uniprot.org/uniprotkb",
        "idmapping": "https://rest.uniprot.org/idmapping"
    },
    "api_key": "none_required",
    "database_size": "250+ million protein sequences"
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/uniprot_integration.py`

### Usage Examples

```python
from tools_utilities.uniprot_integration import UniProtClient

# Initialize client
client = UniProtClient()

# Search proteins
results = client.search_proteins(
    query='dopamine receptor',
    organism='human',
    reviewed=True,  # Swiss-Prot only
    fields=['accession', 'id', 'protein_name', 'gene_names']
)

# Get specific protein
p53 = client.get_protein('P04637')  # p53 tumor suppressor

# Get FASTA sequence
fasta = client.get_fasta('P00533')  # EGFR

# Map identifiers
mapping = client.map_identifiers(
    ids=['P04637', 'P00533'],
    from_db='UniProtKB_AC-ID',
    to_db='PDB'
)
```

## Brain Proteins Retrieved

Successfully identified 20 brain-related proteins across 4 categories:

### Categories
- **Neurotransmitter Receptors** (5): Dopamine, serotonin, GABA receptors
- **Ion Channels** (5): Sodium, potassium, calcium channels
- **Synaptic Proteins** (5): Synapsin, synaptophysin, PSD-95
- **Neurodegenerative** (5): APP, tau, alpha-synuclein

### Example Proteins

| Accession | ID | Protein | Gene |
|-----------|----|---------| -----|
| P14416 | DRD2_HUMAN | D(2) dopamine receptor | DRD2 |
| P21917 | DRD4_HUMAN | D(4) dopamine receptor | DRD4 |
| P04637 | P53_HUMAN | Cellular tumor antigen p53 | TP53 |
| P05067 | A4_HUMAN | Amyloid-beta precursor protein | APP |

## Search Capabilities

### Query Syntax
- **Lucene Query Language**: Boolean operators, wildcards
- **Field Searches**: `gene:BRCA2`, `organism:human`
- **Filters**: Reviewed status, existence level, length
- **Pagination**: Size and cursor parameters

### Example Queries
```python
# Complex search
results = client.search_proteins(
    query='(kinase OR phosphatase) AND reviewed:true',
    organism='9606',  # Human taxonomy ID
    fields=['accession', 'id', 'ec', 'keywords']
)

# Disease-associated proteins
results = client.search_proteins(
    query='disease:alzheimer',
    reviewed=True
)
```

## Integration with Quark

### Use Cases for Brain Simulation
1. **Protein Networks**: Build protein interaction networks
2. **Sequence Analysis**: Compare protein sequences
3. **Functional Annotation**: Get protein functions and pathways
4. **Disease Modeling**: Study disease-associated proteins
5. **Cross-Database Links**: Connect to PDB, AlphaFold, etc.

### Scientific Applications
- Proteomics analysis
- Evolutionary studies
- Drug target identification
- Pathway reconstruction
- Disease mechanism research

## Data Storage

Generated data saved to:
- `/data/knowledge/uniprot_brain_proteins.json`

## Testing

Run the integration test:
```bash
python tools_utilities/uniprot_integration.py
```

## Rate Limits

- **No explicit limit**: But be considerate
- **Recommended**: Max 2 requests per second
- **Bulk Downloads**: Use FTP for large datasets

## Additional Resources

### SPARQL Endpoint
```python
# For complex queries
sparql_url = "https://sparql.uniprot.org/sparql"
```

### FTP Downloads
- Full database: `ftp://ftp.uniprot.org/pub/databases/uniprot/`
- Current release: `ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/`

## References

### Documentation
- [UniProt Help](https://www.uniprot.org/help)
- [API Documentation](https://www.uniprot.org/help/api)
- [Query Syntax](https://www.uniprot.org/help/query-fields)

### Citation
- The UniProt Consortium (2023) *Nucleic Acids Research* - "UniProt: the Universal Protein Knowledgebase"

### Support
- Help: https://www.uniprot.org/contact
- Forum: https://www.uniprot.org/help/community

## Status

✅ **Integration Complete**: API configured, tested, and 20 brain-related proteins retrieved from 250+ million protein database.
