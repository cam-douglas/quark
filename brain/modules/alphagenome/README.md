# AlphaGenome Module

DNA sequence analysis and prediction module for biological simulation.

## Module Structure

- `types.py` - Core data types (OutputType enum, GenomicInterval, Variant, PredictionOutputs)
- `model_core.py` - Main AlphaGenomeModel class with prediction methods
- `mock_data.py` - Mock data generation for testing
- `models.py` - Main entry point and convenience functions
- `data.py` - Genome data utilities
- `visualization.py` - Plotting and visualization tools

## Usage

```python
from alphagenome.models import create, GenomicInterval, OutputType

# Create model
model = create(api_key="your_key")

# Define interval
interval = GenomicInterval("chr1", 1000000, 1010000)

# Make predictions
outputs = model.predict_interval(
    interval,
    requested_outputs=[OutputType.RNA_SEQ, OutputType.ATAC]
)

# Access results
expression = outputs.rna_seq
accessibility = outputs.atac
```

## File Size Compliance

All modules follow the project's file size limits:
- Each file is under 150 lines (target range)
- No file exceeds 300 lines (hard limit)
- Clear separation of concerns across modules
