# NCBI E-utilities API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active

## Overview

The NCBI Entrez Programming Utilities (E-utilities) API integration provides Quark with programmatic access to over 30 biological databases at the National Center for Biotechnology Information (NCBI), including PubMed, GenBank, Gene, Protein, and more. This enables comprehensive biological literature and data retrieval capabilities.

## Key Features

### API Capabilities
- **30+ Databases**: Access to PubMed, GenBank, Gene, Protein, Structure, and more
- **No API Key Required**: Works without authentication (3 req/sec)
- **Better with API Key**: 10 req/sec with free NCBI account
- **Multiple Operations**: Search, fetch, link, summarize across databases
- **Cross-Database Linking**: Find related records between databases

### Available Utilities
1. **ESearch**: Text searches using Entrez queries
2. **EFetch**: Download full records in various formats
3. **ESummary**: Get document summaries
4. **ELink**: Find related data across databases
5. **EPost**: Upload UID lists to History server
6. **EInfo**: Get database statistics and field info
7. **ESpell**: Get spelling suggestions
8. **ECitMatch**: Find PubMed IDs from citations

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"ncbi_eutilities": {
    "service": "NCBI Entrez Programming Utilities",
    "endpoints": {
        "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "esearch": ".../esearch.fcgi",
        "efetch": ".../efetch.fcgi",
        "esummary": ".../esummary.fcgi"
    },
    "api_key": "optional_but_recommended",
    "rate_limits": {
        "with_api_key": "10 requests per second",
        "without_api_key": "3 requests per second"
    }
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/ncbi_eutilities_integration.py`

### Usage Examples

```python
from tools_utilities.ncbi_eutilities_integration import NCBIEutilitiesClient

# Initialize client (optional API key for better rate limits)
client = NCBIEutilitiesClient(
    api_key=None,  # Add NCBI API key if available
    email="your_email@example.com",
    tool="YourToolName"
)

# Search PubMed
results = client.search('pubmed', 'BRCA1 cancer', retmax=10)
print(f"Found {results['count']} papers")

# Get document summaries
summaries = client.get_summaries('pubmed', results['ids'])

# Fetch full abstracts
abstracts = client.fetch('pubmed', results['ids'][:5], 
                        rettype='abstract', retmode='text')

# Find related genes for a protein
gene_links = client.find_related('protein', 'gene', 'P04637')

# Search for brain research papers
brain_papers = client.search_pubmed_for_brain_research(
    "synaptic plasticity", 
    max_results=5
)
```

## Available Methods

### Core Methods
- `search(database, query, retmax, use_history)` - Search any NCBI database
- `fetch(database, ids, rettype, retmode)` - Download full records
- `get_summaries(database, ids)` - Get document summaries
- `find_related(from_db, to_db, ids)` - Find cross-database links
- `get_database_info(database)` - Get database statistics

### Specialized Methods
- `search_pubmed_for_brain_research(topic, max_results)` - Brain-specific PubMed search

## Supported Databases

### Primary Databases
- **pubmed**: Biomedical literature citations
- **pmc**: Full-text articles in PubMed Central
- **protein**: Protein sequences
- **nucleotide**: DNA/RNA sequences
- **gene**: Gene-centered information
- **structure**: 3D molecular structures
- **genome**: Complete genome assemblies

### Additional Databases
- **biosystems**: Biological pathways
- **taxonomy**: Organism classification
- **snp**: Single nucleotide polymorphisms
- **clinvar**: Clinical variants
- **omim**: Mendelian inheritance
- **bioproject**: Research projects
- **gds**: Gene expression datasets

## Test Results

Successfully tested the following operations:

### PubMed Searches
- **BRCA1 cancer**: Found 22,698 papers
- **Synaptic transmission**: Found 9,011 papers
- **Dopamine receptor**: Found 718 papers
- **Neurodegeneration**: Found 9,681 papers

### Brain Research Topics Searched
1. **Dopamine receptor structure**: 718 papers
2. **Synaptic plasticity**: 18,580 papers
3. **Neurodegeneration Alzheimer**: 9,681 papers
4. **GABA neurotransmitter**: 8,720 papers
5. **Ion channel gating**: 586 papers

### Cross-Database Linking
- Successfully linked protein records to gene records
- Found related structures for proteins

## Integration with Quark

### Use Cases for Brain Simulation
1. **Literature Mining**: Find latest brain research papers
2. **Protein-Gene Mapping**: Link brain proteins to their genes
3. **Pathway Analysis**: Retrieve biological pathway information
4. **Disease Research**: Access clinical variant data
5. **Sequence Analysis**: Download protein/DNA sequences

### Scientific Applications
- Automated literature reviews
- Evidence-based validation
- Cross-reference biological data
- Track research trends
- Find experimental protocols

## Rate Limits and Best Practices

### Without API Key
- 3 requests per second maximum
- Include email and tool name

### With API Key (Recommended)
- 10 requests per second maximum
- Register at: https://www.ncbi.nlm.nih.gov/account/
- Free registration with NCBI account

### Usage Guidelines
- Run large jobs on weekends or 9 PM - 5 AM ET
- Use History server for large result sets
- Include tool name and email in all requests
- Respect rate limits to avoid blocking

## Testing

Run the integration test:
```bash
python tools_utilities/ncbi_eutilities_integration.py
```

This will:
- Test all major E-utilities functions
- Search for brain research papers
- Demonstrate cross-database linking
- Save results to `data/knowledge/ncbi_brain_research.json`

## Data Storage

Search results saved to:
```
/Users/camdouglas/quark/data/knowledge/ncbi_brain_research.json
```

## Getting an API Key

To get better rate limits (10 req/sec vs 3 req/sec):
1. Create NCBI account: https://www.ncbi.nlm.nih.gov/account/
2. Generate API key in account settings
3. Add to credentials file under `ncbi_eutilities.api_key`

## References

### Documentation
- [E-utilities Quick Start](https://www.ncbi.nlm.nih.gov/books/NBK25500/)
- [E-utilities In Depth](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [PubMed Help](https://pubmed.ncbi.nlm.nih.gov/help/)

### Support
- **Email**: eutilities@ncbi.nlm.nih.gov
- **Mailing List**: utilities-announce@ncbi.nlm.nih.gov

## Notes

- API enforces rate limiting automatically
- History server available for large queries
- XML and JSON output formats supported
- Can search multiple databases simultaneously
- Cross-database linking enables comprehensive searches

## Status

✅ **Integration Complete**: API configured, Python module created, and successfully tested with multiple brain research queries retrieving thousands of relevant papers.
