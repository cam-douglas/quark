# arXiv API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active  
**Reference**: [arXiv API Documentation](https://info.arxiv.org/help/api/)

## Overview

The arXiv API integration provides Quark with access to over 2 million open-access scholarly articles in physics, mathematics, computer science, quantitative biology, and other fields. arXiv is a free distribution service and open-access archive maintained by Cornell University.

## Key Features

### API Capabilities
- **No Authentication Required**: Public API
- **2+ Million Preprints**: Extensive research archive
- **Multiple Fields**: Physics, Math, CS, Biology, Finance
- **Full-Text Search**: Title, abstract, author, category
- **Atom Feed Format**: Standard XML response
- **Version History**: Access all paper versions
- **Direct PDF Access**: Download papers directly

### Available Services
1. **Search**: Query papers by various criteria
2. **ID Lookup**: Direct access by arXiv ID
3. **Category Browse**: Papers by subject category
4. **Author Search**: Find papers by author name
5. **Date Filtering**: Recent submissions
6. **Sorting Options**: Relevance, date, etc.

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"arxiv": {
    "service": "arXiv API",
    "endpoints": {
        "base": "http://export.arxiv.org/api",
        "query": "http://export.arxiv.org/api/query",
        "oai": "http://export.arxiv.org/oai2"
    },
    "api_key": "none_required",
    "database_size": "2+ million preprints"
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/arxiv_integration.py`

### Usage Examples

```python
from tools_utilities.arxiv_integration import ArXivClient

# Initialize client
client = ArXivClient()

# Search papers
results = client.search(
    query='quantum computing',
    max_results=10,
    sort_by='relevance'
)

# Get paper by ID
paper = client.get_by_id('1706.03762')  # Attention Is All You Need

# Search by author
papers = client.search_by_author('Yann LeCun')

# Search by category
recent_ml = client.search_by_category(
    'cs.LG',  # Machine Learning
    days_back=7
)

# Search neuroscience papers
neuro_papers = client.search_neuroscience_papers()
```

## Subject Categories

### Primary Categories Relevant to Quark

| Code | Category | Description |
|------|----------|-------------|
| q-bio.NC | Neurons and Cognition | Computational neuroscience |
| q-bio.QM | Quantitative Methods | Computational biology |
| cs.NE | Neural and Evolutionary Computing | Neural networks |
| cs.AI | Artificial Intelligence | AI research |
| cs.LG | Machine Learning | ML algorithms |
| physics.bio-ph | Biological Physics | Biophysics |
| stat.ML | Machine Learning | Statistical ML |
| math.DS | Dynamical Systems | Neural dynamics |

## Query Syntax

### Search Fields
- **ti**: Title (`ti:neural`)
- **au**: Author (`au:Hinton`)
- **abs**: Abstract (`abs:brain`)
- **co**: Comment (`co:journal`)
- **jr**: Journal reference
- **cat**: Category (`cat:q-bio.NC`)
- **id**: arXiv ID

### Boolean Operators
```python
# AND operator
query = 'ti:neural AND au:Hinton'

# OR operator
query = 'abs:fMRI OR abs:EEG'

# ANDNOT operator
query = 'cat:cs.LG ANDNOT ti:survey'

# Complex queries
query = '(ti:brain OR ti:neural) AND cat:q-bio.NC'
```

### Date Ranges
```python
# Papers from specific date range
query = 'submittedDate:[20250101 TO 20250120]'

# Papers updated recently
query = 'lastUpdatedDate:[20250101 TO *]'
```

## Neuroscience Papers Retrieved

Successfully identified 20 neuroscience papers across 4 categories:

### Categories Found
- **Computational Neuroscience** (5): Neural dynamics, brain modeling
- **Brain Imaging** (5): fMRI, EEG, neuroimaging methods
- **Neural Networks** (5): Biological neural networks, spiking neurons
- **Cognitive Science** (5): Cognition, perception, behavior

### Example Papers
- Neural dynamics and computation
- Brain connectivity analysis
- Spiking neural network models
- Cognitive architectures

## Search Parameters

### Pagination
```python
params = {
    'start': 0,           # Starting index
    'max_results': 100,   # Results per request (max 30000)
}
```

### Sorting
```python
sort_options = {
    'relevance': 'Best match to query',
    'lastUpdatedDate': 'Recently updated first',
    'submittedDate': 'Recently submitted first'
}
```

## Integration with Quark

### Use Cases for Brain Simulation
1. **Literature Review**: Stay current with neuroscience research
2. **Method Discovery**: Find new computational techniques
3. **Algorithm Research**: Neural network architectures
4. **Theory Development**: Mathematical models of brain
5. **Validation Sources**: Compare with published results

### Scientific Applications
- Research tracking
- Literature mining
- Citation networks
- Trend analysis
- Collaboration discovery

## Best Practices

### Efficient Searching
```python
# Be specific with queries
query = 'ti:"spiking neural" AND cat:q-bio.NC'

# Use field prefixes
query = 'au:"Geoffrey Hinton" AND ti:deep'

# Limit results appropriately
max_results = 100  # Don't request more than needed

# Cache results locally
import json
with open('arxiv_cache.json', 'w') as f:
    json.dump(results, f)
```

### Rate Limiting
```python
# Be considerate with requests
time.sleep(0.5)  # Between requests

# Batch queries when possible
queries = ['query1', 'query2']
for q in queries:
    results = client.search(q)
    time.sleep(1)
```

## Data Storage

Generated data saved to:
- `/data/knowledge/arxiv_neuroscience.json`

## Testing

Run the integration test:
```bash
python tools_utilities/arxiv_integration.py
```

## Response Format

### Atom Feed Structure
```xml
<feed>
  <title>ArXiv Query Results</title>
  <entry>
    <id>http://arxiv.org/abs/1234.5678</id>
    <title>Paper Title</title>
    <summary>Abstract text...</summary>
    <author><name>Author Name</name></author>
    <published>2025-01-20T00:00:00Z</published>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/pdf/1234.5678" type="application/pdf"/>
  </entry>
</feed>
```

## Bulk Data Access

For large-scale downloads:
- **Bulk Data**: https://info.arxiv.org/help/bulk_data.html
- **OAI-PMH**: https://info.arxiv.org/help/oa/
- **Kaggle Dataset**: Full arXiv dataset on Kaggle

## Example Workflow

```python
# Monitor new neuroscience papers
def monitor_neuroscience():
    client = ArXivClient()
    
    # Get papers from last week
    results = client.search_by_category(
        'q-bio.NC',
        days_back=7
    )
    
    # Filter by keywords
    keywords = ['neural', 'brain', 'cognition']
    relevant = []
    
    for paper in results['entries']:
        abstract = paper['abstract'].lower()
        if any(kw in abstract for kw in keywords):
            relevant.append({
                'title': paper['title'],
                'authors': paper['authors'][:3],
                'abstract': paper['abstract'][:200],
                'pdf': paper['pdf_url'],
                'date': paper['published']
            })
    
    return relevant
```

## References

### Documentation
- [arXiv API Manual](https://info.arxiv.org/help/api/user-manual.html)
- [API Basics](https://info.arxiv.org/help/api/basics.html)
- [Terms of Use](https://info.arxiv.org/help/api/tou.html)

### Support
- Mailing List: arxiv-api@googlegroups.com
- GitHub Issues: https://github.com/arXiv/arxiv-api

### Citation
When using arXiv content, cite papers individually and acknowledge arXiv:
```
arXiv.org e-Print archive, Cornell University
```

## Notes

- **No API Key Required**: Completely open access
- **Be Reasonable**: No hard rate limits but don't abuse
- **Max Results**: Can retrieve up to 30,000 results per query
- **Metadata Only**: API provides metadata and links, not full text

## Status

✅ **Integration Complete**: API configured, tested, and 20 neuroscience papers retrieved from 2+ million preprint archive.
