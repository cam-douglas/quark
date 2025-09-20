# Ensembl REST API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active

## Overview

The Ensembl REST API integration provides Quark with comprehensive genomic data access including genes, transcripts, proteins, variations, regulatory features, and comparative genomics across multiple species. Ensembl is one of the world's leading genomic databases, covering vertebrates and other genomes.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API with generous rate limits
- **15 req/sec Rate Limit**: Fair use policy for all users
- **Multiple Species**: Human, mouse, rat, zebrafish, and 100+ others
- **Rich Data Types**: Genes, transcripts, proteins, variants, regulation
- **Comparative Genomics**: Orthologs, paralogs, synteny, alignments
- **Cross-References**: Links to UniProt, RefSeq, OMIM, etc.

### Available Services
1. **Lookup**: Find features by ID or symbol
2. **Sequence**: Retrieve DNA, cDNA, protein sequences
3. **Variation**: Access variant data and consequences
4. **Homology**: Find orthologs and paralogs
5. **Regulation**: Regulatory features and binding sites
6. **Cross-References**: External database mappings
7. **VEP**: Variant Effect Predictor
8. **Overlap**: Find features in genomic regions

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"ensembl": {
    "service": "Ensembl REST API",
    "endpoints": {
        "base": "https://rest.ensembl.org",
        "lookup": ".../lookup",
        "sequence": ".../sequence",
        "variation": ".../variation",
        "homology": ".../homology",
        "xrefs": ".../xrefs"
    },
    "api_key": "none_required",
    "rate_limits": {
        "requests_per_second": "15 req/sec"
    }
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/ensembl_integration.py`

### Usage Examples

```python
from tools_utilities.ensembl_integration import EnsemblClient

# Initialize client
client = EnsemblClient()

# Look up gene by symbol
gene = client.lookup_by_symbol('homo_sapiens', 'BRCA2')
print(f"Gene ID: {gene['id']}")

# Get protein sequence
sequence = client.get_sequence('ENST00000646891', 'protein', 'fasta')

# Find variants in a gene
variant = client.get_variant_by_id('homo_sapiens', 'rs429358')

# Get mouse orthologs
homologs = client.get_homology(
    'homo_sapiens',
    'ENSG00000142192',  # APP gene
    target_species='mus_musculus'
)

# Get cross-references
xrefs = client.get_xrefs('ENSG00000139618')  # BRCA2

# Search brain genes
brain_genes = client.search_brain_genes('homo_sapiens')
```

## Available Methods

### Core Methods
- `lookup_by_id(stable_id, expand)` - Find feature by Ensembl ID
- `lookup_by_symbol(species, symbol, expand)` - Find gene by symbol
- `get_sequence(stable_id, object_type, format)` - Get sequences
- `get_homology(species, gene_id, target_species)` - Find homologs
- `get_variants_in_region(species, region)` - Get regional variants
- `get_variant_by_id(species, variant_id)` - Get variant details
- `get_xrefs(stable_id, external_db)` - Get cross-references
- `get_species_info()` - List available species

### Specialized Methods
- `search_brain_genes(species)` - Search for brain-related genes

## Brain Genes Successfully Retrieved

During testing, the following 17 brain-related genes were found:

### Neurotransmitter Receptors
- **DRD2** (ENSG00000149295): Dopamine receptor D2
- **HTR2A** (ENSG00000102468): Serotonin receptor 2A
- **GRIN1** (ENSG00000176884): NMDA receptor subunit 1
- **GABRA1** (ENSG00000022355): GABA receptor alpha 1

### Synaptic Proteins
- **SYN1** (ENSG00000008056): Synapsin I
- **SYT1** (ENSG00000067715): Synaptotagmin 1
- **SNAP25** (ENSG00000132639): Synaptosomal-associated protein 25

### Neurodegenerative Disease Genes
- **APP** (ENSG00000142192): Amyloid precursor protein
- **MAPT** (ENSG00000186868): Tau protein
- **SNCA** (ENSG00000145335): Alpha-synuclein
- **HTT** (ENSG00000197386): Huntingtin

### Ion Channels
- **SCN1A** (ENSG00000144285): Sodium channel alpha 1
- **KCNQ2** (ENSG00000075043): Potassium channel Q2
- **CACNA1A** (ENSG00000141837): Calcium channel alpha 1A

### Neurodevelopment Genes
- **FOXP2** (ENSG00000128573): Language and speech
- **MECP2** (ENSG00000169057): Rett syndrome
- **FMR1** (ENSG00000102081): Fragile X syndrome

## Integration with Quark

### Use Cases for Brain Simulation
1. **Gene Expression**: Access gene and transcript information
2. **Protein Sequences**: Download protein sequences for modeling
3. **Genetic Variants**: Study disease-associated variants
4. **Comparative Analysis**: Compare across species
5. **Regulatory Elements**: Understand gene regulation
6. **Pathway Mapping**: Link to metabolic pathways

### Scientific Applications
- Gene-disease associations
- Evolutionary conservation analysis
- Variant impact prediction
- Cross-species comparisons
- Regulatory network analysis
- Protein domain mapping

## Data Formats

### Supported Output Formats
- **JSON**: Default for most endpoints
- **FASTA**: Sequence data
- **XML**: Alternative to JSON
- **BED**: Genomic features
- **GFF3**: Gene annotations
- **PhyloXML**: Phylogenetic trees
- **OrthoXML**: Orthology data

## Testing

Run the integration test:
```bash
python tools_utilities/ensembl_integration.py
```

This will:
- Look up genes by symbol
- Retrieve sequences
- Find variants
- Get cross-references
- Search for brain genes
- Find orthologs

## Data Storage

Brain gene data saved to:
```
/Users/camdouglas/quark/data/knowledge/ensembl_brain_genes.json
```

## Example Species Codes

### Common Species
- `homo_sapiens` - Human
- `mus_musculus` - Mouse
- `rattus_norvegicus` - Rat
- `danio_rerio` - Zebrafish
- `drosophila_melanogaster` - Fruit fly
- `caenorhabditis_elegans` - C. elegans
- `saccharomyces_cerevisiae` - Yeast

## Rate Limits and Best Practices

### Rate Limiting
- Maximum 15 requests per second
- Automatic rate limiting in client
- No authentication required

### Best Practices
- Cache results when possible
- Use bulk endpoints for multiple queries
- Be considerate of server resources
- Include meaningful User-Agent headers

## References

### Documentation
- [Ensembl REST API](https://rest.ensembl.org/)
- [API Documentation](https://rest.ensembl.org/documentation)
- [GitHub Repository](https://github.com/Ensembl/ensembl-rest)

### Publications
- Cunningham et al. (2022) *Nucleic Acids Research* - "Ensembl 2022"

### Support
- **Help Desk**: https://www.ensembl.org/Help/Contact
- **Mailing Lists**: https://www.ensembl.org/info/about/contact/mailing.html

## Notes

- Ensembl releases new versions quarterly
- Current version: Release 113 (as of testing)
- Includes both experimental and predicted data
- Cross-references to many external databases
- Supports multiple genome assemblies

## Status

✅ **Integration Complete**: API configured, Python module created, and successfully tested with 17 brain-related genes retrieved and comparative genomics demonstrated.
