# OpenML REST API Integration

**Date**: 2025-01-20  
**Status**: ✔ Active

## Overview

The OpenML API integration provides Quark with access to a collaborative machine learning platform. OpenML offers 20,000+ curated datasets, standardized ML tasks, algorithm implementations (flows), and experiment results (runs) for reproducible research.

## Key Features

### API Capabilities
- **Public Access**: No authentication for reading data
- **20,000+ Datasets**: Curated ML datasets with metadata
- **Standardized Tasks**: Benchmarks for fair comparison
- **Algorithm Flows**: Reusable ML workflows
- **Experiment Tracking**: Reproducible runs with evaluations
- **Version Control**: For datasets and algorithms

### Available Services
1. **Dataset Access**: Browse and download ML datasets
2. **Task Management**: Standardized ML challenges
3. **Flow Repository**: Algorithm implementations
4. **Run Storage**: Experiment results and metrics
5. **Study Collections**: Benchmark suites
6. **Evaluations**: Performance metrics

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"openml": {
    "service": "OpenML REST API",
    "endpoints": {
        "base": "https://www.openml.org/api/v1",
        "json_base": "https://www.openml.org/api/v1/json",
        "dataset": ".../data",
        "task": ".../task",
        "flow": ".../flow"
    },
    "api_key": "optional_for_uploads",
    "authentication": "API key required for uploads only"
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/openml_integration.py`

### Usage Examples

```python
from tools_utilities.openml_integration import OpenMLClient

# Initialize client
client = OpenMLClient()

# List datasets
datasets = client.list_datasets(limit=100)

# Get specific dataset
iris = client.get_dataset(61)  # Iris dataset

# List ML tasks
tasks = client.list_tasks(task_type=1)  # Classification tasks

# Search for brain datasets
brain_datasets = client.search_brain_datasets()

# Get popular ML benchmarks
popular = client.get_popular_ml_datasets()
```

## Brain-Related Datasets Found

Successfully identified 6 brain/neuroscience datasets:

### Neuroimaging (1 dataset)
- **eeg-eye-state** (ID: 1471): EEG data for eye state detection

### Cognitive (2 datasets)
- **sleep-deprivation-and-cognitive-performance** (ID: 46754)
- **sleep-deprivation-and-cognitive-performance** (ID: 46897)

### Other Brain Datasets (3 datasets)
- **iq_brain_size** (ID: 483): Brain size and IQ correlation
- **Brainsize** (ID: 1093): Brain measurements dataset
- **Brain-cancer-gene-expression** (ID: 43657): Gene expression data

## Popular ML Datasets Retrieved

Successfully accessed classic ML benchmarks:
- Iris (ID: 61)
- Credit-G (ID: 31)
- Diabetes (ID: 37)
- Spambase (ID: 44)
- Vehicle (ID: 54)
- Bank Marketing (ID: 1461)

## ML Task Types

OpenML supports 8 types of ML tasks:
1. Supervised Classification
2. Supervised Regression
3. Learning Curve
4. Data Stream Classification
5. Clustering
6. ML Challenges
7. Survival Analysis
8. Subgroup Discovery

## Integration with Quark

### Use Cases for Brain Simulation
1. **ML Benchmarking**: Test brain-inspired algorithms
2. **Neural Dataset Access**: EEG, cognitive performance data
3. **AutoML Integration**: Automated model selection
4. **Reproducible Research**: Version-controlled experiments
5. **Meta-Learning**: Learn from past experiments

### Scientific Applications
- Algorithm comparison
- Dataset curation
- Benchmark creation
- Reproducible experiments
- Educational resources

## Python Package

Install the official Python client:
```bash
pip install openml
```

## Data Storage

Generated data saved to:
- `/data/knowledge/openml_brain_datasets.json`

## Testing

Run the integration test:
```bash
python tools_utilities/openml_integration.py
```

## Best Practices

- Cache results when possible
- Be considerate with requests
- Use official Python client for uploads
- Include proper citations in research

## References

### Documentation
- [OpenML Python Docs](https://openml.github.io/openml-python/main/)
- [GitHub Repository](https://github.com/openml/openml-python)

### Publication
- Vanschoren et al. (2013) - "OpenML: Networked science in machine learning"

### Support
- GitHub Issues: https://github.com/openml/OpenML/issues
- Website: https://www.openml.org

## Notes

- Read access is completely open
- API key only needed for uploading datasets/results
- Integrates with scikit-learn, MLR, Weka, etc.
- Supports ARFF, CSV, Parquet formats

## Status

✅ **Integration Complete**: API configured, tested, and 6 brain-related datasets identified.
