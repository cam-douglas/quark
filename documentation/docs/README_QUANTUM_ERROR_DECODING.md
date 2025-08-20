# Quantum Error Correction Decoding Training Module

## Overview

This module implements neural network-based training for quantum error correction decoding, based on the Nature 2024 paper "Learning high-accuracy error decoding for quantum processors". The system trains neural networks to decode quantum error correction codes, specifically focusing on surface codes which are the leading approach for fault-tolerant quantum computation.

## Paper Reference

**Title**: "Learning high-accuracy error decoding for quantum processors"  
**Authors**: Johannes Bausch, Andrew W. Senior, Francisco J. H. Heras, et al.  
**Journal**: Nature, Volume 635, pages 834–840 (2024)  
**DOI**: [10.1038/s41586-024-08148-8](https://www.nature.com/articles/s41586-024-08148-8)

## Key Features

### 1. Surface Code Implementation
- **Code Distance**: Configurable surface code distance (3, 5, 7, 9, 11)
- **Stabilizer Measurements**: X and Z stabilizer measurement simulation
- **Error Injection**: Realistic quantum error simulation (bit flips, phase flips)
- **Syndrome Generation**: Error syndrome extraction for training data

### 2. Neural Network Decoder
- **Architecture**: Convolutional neural network with classification head
- **Input Processing**: 2D syndrome data (X and Z stabilizers)
- **Output**: Binary classification (error detected vs. no error)
- **Training**: Supervised learning with cross-entropy loss

### 3. Training Pipeline
- **Data Generation**: Synthetic training data from surface code simulation
- **Batch Processing**: Configurable batch sizes and learning rates
- **Progress Tracking**: Rich console output with progress bars
- **Result Saving**: Training results, models, and metrics

## Architecture

### SurfaceCode Class
```python
class SurfaceCode:
    """Simplified surface code implementation for quantum error correction."""
    
    def __init__(self, code_distance: int):
        # Initialize d×d grid of data qubits
        # Initialize (d-1)×(d-1) grid of stabilizer qubits
    
    def initialize_logical_state(self, logical_state: str):
        # Initialize logical qubit in |0⟩, |1⟩, |+⟩, or |-⟩ state
    
    def apply_errors(self, error_rate: float):
        # Apply random X and Z errors to data qubits
    
    def measure_stabilizers(self) -> np.ndarray:
        # Measure X and Z stabilizers, return syndrome
```

### QuantumErrorDecoder Class
```python
class QuantumErrorDecoder(nn.Module):
    """Neural network decoder for quantum error correction."""
    
    def __init__(self, config: QuantumErrorDecodingConfig):
        # Convolutional encoder for syndrome processing
        # Classification head for error detection
    
    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        # Process syndrome through network
        # Return binary classification output
```

### QuantumErrorDecodingTrainer Class
```python
class QuantumErrorDecodingTrainer:
    """Trainer for quantum error correction decoding."""
    
    def __init__(self, config: QuantumErrorDecodingConfig):
        # Initialize surface code, decoder, and optimizer
    
    def generate_training_data(self, num_samples: int):
        # Generate synthetic training data
    
    async def train_decoder(self):
        # Train the neural network decoder
```

## Configuration

### QuantumErrorDecodingConfig
```python
@dataclass
class QuantumErrorDecodingConfig:
    code_distance: int = 5              # Surface code distance
    batch_size: int = 32                # Training batch size
    learning_rate: float = 1e-4         # Learning rate
    num_epochs: int = 50                # Number of training epochs
    hidden_dim: int = 256               # Hidden layer dimension
    output_dir: str = "quantum_error_decoding_outputs"
    enable_brain_simulation: bool = True
```

## Usage

### Basic Training
```python
from quantum_error_decoding_training import (
    QuantumErrorDecodingConfig,
    QuantumErrorDecodingTrainer
)

# Configure training
config = QuantumErrorDecodingConfig(
    code_distance=5,
    num_epochs=100,
    batch_size=64
)

# Initialize trainer
trainer = QuantumErrorDecodingTrainer(config)

# Train decoder
results = await trainer.train_decoder()
```

### Custom Surface Code
```python
from quantum_error_decoding_training import SurfaceCode

# Create surface code with custom distance
code = SurfaceCode(code_distance=7)

# Initialize logical state
code.initialize_logical_state("1")

# Apply errors
code.apply_errors(error_rate=0.02)

# Measure stabilizers
syndrome = code.measure_stabilizers()
```

### Standalone Decoder
```python
from quantum_error_decoding_training import QuantumErrorDecoder, QuantumErrorDecodingConfig

# Create decoder
config = QuantumErrorDecodingConfig(code_distance=5)
decoder = QuantumErrorDecoder(config)

# Process syndrome
syndrome = torch.randn(1, 2, 4, 4)  # [batch, channels, height, width]
prediction = decoder(syndrome)
```

## Training Data

### Data Generation
The training data is generated synthetically using the surface code simulation:

1. **Initialization**: Surface code is initialized in a logical state (|0⟩, |1⟩, |+⟩, |-⟩)
2. **Error Injection**: Random X and Z errors are applied to data qubits
3. **Stabilizer Measurement**: X and Z stabilizers are measured to generate syndrome
4. **Label Creation**: Binary labels indicate whether errors were detected

### Data Format
- **Input**: Syndrome tensors of shape `[batch, 2, height, width]`
  - Channel 0: X stabilizer measurements
  - Channel 1: Z stabilizer measurements
  - Height/Width: `code_distance - 1`
- **Output**: Binary labels `[batch]` indicating error presence

## Performance Metrics

### Training Metrics
- **Loss**: Cross-entropy loss during training
- **Epoch Progress**: Training progress per epoch
- **Convergence**: Loss reduction over training epochs

### Decoding Metrics
- **Error Detection Accuracy**: Percentage of correctly detected errors
- **False Positive Rate**: Errors detected when none exist
- **False Negative Rate**: Errors missed when they exist

## Output Files

### Training Results
- `training_results.json`: Training history and loss values
- `quantum_error_decoder.pt`: Trained PyTorch model
- `README_QUANTUM_ERROR_DECODING.md`: This documentation

### File Structure
```
quantum_error_decoding_outputs/
├── training_results.json
├── quantum_error_decoder.pt
└── README_QUANTUM_ERROR_DECODING.md
```

## Testing

### Running Tests
```bash
# Run all tests
python test_quantum_error_decoding_training.py

# Run with pytest
pytest test_quantum_error_decoding_training.py -v
```

### Test Coverage
- **Configuration**: Default and custom configuration values
- **Surface Code**: Initialization, error injection, stabilizer measurement
- **Neural Decoder**: Architecture, forward pass, output validation
- **Training Pipeline**: Data generation, training, result saving
- **Integration**: Complete training pipeline validation

## Dependencies

### Required Packages
- `torch`: PyTorch for neural networks
- `numpy`: Numerical computing
- `rich`: Rich console output
- `pytest`: Testing framework (for tests)

### Optional Dependencies
- `qiskit`: Quantum computing simulation (future enhancement)
- `matplotlib`: Visualization (future enhancement)

## Future Enhancements

### Planned Features
1. **Advanced Error Models**: Cross-talk, leakage, measurement errors
2. **Multi-Distance Training**: Joint training across code distances
3. **Real Quantum Hardware**: Integration with actual quantum processors
4. **Advanced Architectures**: Transformer-based decoders
5. **Visualization**: Error pattern visualization and analysis

### Research Directions
1. **Noise Adaptation**: Learning from real quantum hardware noise
2. **Fault Tolerance**: Achieving fault-tolerant thresholds
3. **Scalability**: Handling larger surface codes
4. **Quantum Advantage**: Demonstrating quantum advantage in error correction

## Technical Details

### Quantum Error Correction
- **Surface Code**: 2D grid of data qubits with stabilizer measurements
- **Stabilizers**: X and Z operators that commute with logical operators
- **Syndrome**: Pattern of stabilizer measurement outcomes indicating errors
- **Decoding**: Process of inferring error locations from syndrome

### Neural Network Architecture
- **Encoder**: 3-layer convolutional network for syndrome processing
- **Pooling**: Global average pooling for spatial feature aggregation
- **Classifier**: 3-layer fully connected network for final classification
- **Activation**: ReLU activation functions with dropout regularization

### Training Process
1. **Data Generation**: Synthetic syndrome data from surface code simulation
2. **Forward Pass**: Syndrome processing through neural network
3. **Loss Calculation**: Cross-entropy loss between predictions and labels
4. **Backward Pass**: Gradient computation and parameter updates
5. **Iteration**: Repeat for specified number of epochs

## Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 Python style guide
2. **Documentation**: Add docstrings for all functions and classes
3. **Testing**: Write tests for new functionality
4. **Type Hints**: Use type hints for function parameters and returns

### Testing New Features
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Validate training performance
4. **Regression Tests**: Ensure existing functionality remains intact

## License

This module is licensed under the Apache-2.0 License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bausch2024learning,
  title={Learning high-accuracy error decoding for quantum processors},
  author={Bausch, Johannes and Senior, Andrew W and Heras, Francisco J H and others},
  journal={Nature},
  volume={635},
  pages={834--840},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## Contact

For questions, issues, or contributions, please contact the development team or open an issue in the repository.
