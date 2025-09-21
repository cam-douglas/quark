# NCBI BLAST REST API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active  
**Reference**: [BLAST Developer Info](https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html)

## Overview

The NCBI BLAST REST API integration provides Quark with sequence similarity search capabilities. BLAST (Basic Local Alignment Search Tool) finds regions of similarity between biological sequences, helping identify homologous proteins and genes.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API (email recommended)
- **Multiple Programs**: blastn, blastp, blastx, tblastn, tblastx
- **Large Databases**: nr, nt, RefSeq, PDB, Swiss-Prot
- **Advanced Algorithms**: PSI-BLAST, PHI-BLAST, DELTA-BLAST
- **Statistical Analysis**: E-values, bit scores, percent identity
- **Multiple Output Formats**: XML, JSON, CSV, HTML

### Available Services
1. **Sequence Search**: Find similar sequences
2. **Database Selection**: Search specific sequence databases
3. **Alignment Generation**: Pairwise and multiple alignments
4. **Statistical Scoring**: E-values and significance
5. **Filter Options**: Low complexity, repeats
6. **Batch Processing**: Multiple queries in one search

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"blast": {
    "service": "NCBI BLAST REST API",
    "endpoints": {
        "base": "https://blast.ncbi.nlm.nih.gov/Blast.cgi",
        "url_api": "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
    },
    "api_key": "none_required",
    "authentication": "Public API - email parameter recommended"
}
```

## ⚠️ IMPORTANT: Rate Limits

**STRICT LIMITS ENFORCED**:
- **Submission**: No more than once every 10 seconds
- **Status Polling**: No more than once per minute per RID
- **Daily Limit**: 100 searches per 24 hours before throttling
- **Large Jobs**: Run on weekends or 9pm-5am ET weekdays

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/blast_integration.py`

### Usage Examples

```python
from tools_utilities.blast_integration import BLASTClient

# Initialize with contact info
client = BLASTClient(
    email="your_email@example.com",
    tool="YourToolName"
)

# Submit a search
rid = client.submit_search(
    query=">MyProtein\nMKLLILSLVLAFSSATAAFAAIPQNIRIGG",
    program='blastp',
    database='pdb',
    expect=1.0
)

# Get results (waits automatically)
xml_results = client.get_results(rid)
hits = client.parse_xml_results(xml_results)

# Complete search in one call
hits = client.blast_sequence(
    sequence="YGGFM",  # Enkephalin peptide
    program='blastp',
    database='nr'
)
```

## BLAST Programs

| Program | Query | Database | Use Case |
|---------|-------|----------|----------|
| blastn | Nucleotide | Nucleotide | DNA/RNA similarity |
| blastp | Protein | Protein | Protein similarity |
| blastx | Nucleotide | Protein | Translated search |
| tblastn | Protein | Nucleotide | Search DNA for proteins |
| tblastx | Nucleotide | Nucleotide | Six-frame translation |
| megablast | Nucleotide | Nucleotide | Fast DNA comparison |

## Databases

### Protein Databases
- **nr**: Non-redundant protein sequences
- **refseq_protein**: NCBI Reference proteins
- **swissprot**: Swiss-Prot curated proteins
- **pdb**: Protein Data Bank sequences

### Nucleotide Databases
- **nt**: Nucleotide sequences
- **refseq_rna**: Reference RNA sequences
- **est**: Expressed sequence tags
- **gss**: Genome survey sequences

## Search Parameters

### Key Parameters
```python
params = {
    'PROGRAM': 'blastp',        # Algorithm
    'DATABASE': 'nr',            # Target database
    'EXPECT': 10.0,              # E-value threshold
    'HITLIST_SIZE': 50,          # Max hits to return
    'FILTER': 'L',               # Low complexity filter
    'MATRIX': 'BLOSUM62',        # Scoring matrix
    'WORD_SIZE': 3,              # Word size for initial matches
    'GAPCOSTS': '11 1'           # Gap open and extend costs
}
```

## Integration with Quark

### Use Cases for Brain Simulation
1. **Protein Identification**: Find homologs of brain proteins
2. **Gene Discovery**: Identify related genes across species
3. **Evolutionary Analysis**: Study conservation of neural proteins
4. **Function Prediction**: Infer function from sequence similarity
5. **Drug Target Search**: Find similar proteins as drug targets

### Scientific Applications
- Homology detection
- Phylogenetic analysis
- Primer design
- Gene annotation
- Structural prediction

## Best Practices

### Optimize Searches
```python
# Batch short sequences
sequences = [seq1, seq2, seq3]
combined = '\n'.join([f">seq{i}\n{s}" for i, s in enumerate(sequences)])

# Use appropriate E-value
# Strict: 0.001
# Standard: 10
# Exploratory: 100

# Select specific database
# Faster: pdb, swissprot
# Comprehensive: nr, nt
```

### Error Handling
```python
try:
    hits = client.blast_sequence(sequence)
except TimeoutError:
    print("Search took too long")
except RuntimeError as e:
    print(f"Search failed: {e}")
```

## Testing

Run the integration test (takes 1-2 minutes):
```bash
python tools_utilities/blast_integration.py
```

## Alternative Access

### High-Volume Projects
For projects with >100 searches/day:
1. **Standalone BLAST+**: Download and run locally
2. **Cloud BLAST**: Use AWS/GCP instances
3. **Docker**: NCBI BLAST Docker image
4. **Elastic BLAST**: Scalable cloud solution

### Download BLAST+
```bash
# macOS
brew install blast

# Linux
apt-get install ncbi-blast+

# From source
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
```

## Example Workflow

```python
# Find proteins similar to a neurotransmitter receptor
dopamine_receptor_seq = """
>DRD2_fragment
MDPLNLSWYDDDLERQNWSRPFNGSDGKADRPHYNYYATLLTLLIAVIVFGNVLVCMAVSREKALQTTT
"""

# Search PDB for structural homologs
hits = client.blast_sequence(
    sequence=dopamine_receptor_seq,
    program='blastp',
    database='pdb',
    expect=0.001  # Strict threshold for structural matches
)

# Analyze results
for hit in hits[:5]:
    print(f"PDB: {hit['accession']}")
    print(f"Identity: {hit['hsps'][0]['percent_identity']:.1f}%")
    print(f"E-value: {hit['hsps'][0]['evalue']:.2e}")
```

## References

### Documentation
- [BLAST Help](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs)
- [BLAST API](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=DeveloperInfo)
- [BLAST+ User Manual](https://www.ncbi.nlm.nih.gov/books/NBK279690/)

### Citation
- Altschul et al. (1990) *J Mol Biol* - "Basic local alignment search tool"
- Camacho et al. (2009) *BMC Bioinformatics* - "BLAST+: architecture and applications"

### Support
- Email: blast-help@ncbi.nlm.nih.gov
- Web: https://blast.ncbi.nlm.nih.gov/Blast.cgi

## Status

✅ **Integration Complete**: API configured and tested. Note: Use sparingly due to strict rate limits (100 searches/day).
