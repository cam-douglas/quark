# NEST Training Module

## Overview

The NEST Training Module integrates the [NEST Simulator](https://www.nest-simulator.org/) with the brain simulation framework for advanced spiking neural network simulations and computational neuroscience research.

## Features

- **Spiking Neural Network Simulations**: Create and simulate networks of spiking neurons
- **Multiple Neuron Models**: Support for various neuron types (iaf_cond_alpha, iaf_psc_alpha, etc.)
- **Synaptic Plasticity**: Implement learning and plasticity mechanisms
- **Performance Monitoring**: Track simulation metrics and performance
- **Brain Simulation Integration**: Connect with neural dynamics and cognitive science components
- **Scalable Architecture**: Support for threading and MPI parallelization

## Installation

### Prerequisites

1. **NEST Simulator**: Install NEST following the [official installation guide](https://www.nest-simulator.org/installation/)
2. **Python Dependencies**: Install required Python packages

```bash
pip install numpy pandas matplotlib seaborn rich pytest
```

### NEST Installation

#### Ubuntu/Debian
```bash
sudo apt-get install nest
```

#### macOS
```bash
brew install nest
```

#### From Source
```bash
git clone https://github.com/nest/nest-simulator.git
cd nest-simulator
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
sudo make install
```

## Usage

### Basic Usage

```python
from src.training.nest_training import NESTConfig, NESTTrainer
import asyncio

# Create configuration
config = NESTConfig(
    simulation_time=1000.0,  # ms
    num_neurons=1000,
    connectivity_density=0.1,
    output_dir="nest_outputs"
)

# Initialize trainer
trainer = NESTTrainer(config)

# Train network
async def main():
    results = await trainer.train_network("simple_network", epochs=10)
    print(f"Training completed with {len(results)} epochs")

asyncio.run(main())
```

### Advanced Configuration

```python
config = NESTConfig(
    # Simulation parameters
    simulation_time=2000.0,  # ms
    num_neurons=5000,
    connectivity_density=0.05,
    
    # Brain simulation integration
    enable_brain_simulation=True,
    
    # Output settings
    output_dir="advanced_nest_outputs"
)
```

## Configuration Options

### NESTConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulation_time` | float | 1000.0 | Simulation duration in milliseconds |
| `num_neurons` | int | 1000 | Number of neurons in the network |
| `connectivity_density` | float | 0.1 | Connection probability between neurons |
| `output_dir` | str | "nest_outputs" | Directory for saving results |
| `enable_brain_simulation` | bool | True | Enable brain simulation integration |

## Neuron Models

The module supports various NEST neuron models:

- **iaf_cond_alpha**: Integrate-and-fire with alpha-shaped conductance synapses
- **iaf_psc_alpha**: Integrate-and-fire with alpha-shaped current synapses
- **iaf_cond_exp**: Integrate-and-fire with exponential conductance synapses
- **iaf_psc_exp**: Integrate-and-fire with exponential current synapses
- **hh_cond_exp_traub**: Hodgkin-Huxley with Traub parameters
- **aeif_cond_alpha**: Adaptive exponential integrate-and-fire

## Synapse Models

Available synapse types:

- **static_synapse**: Fixed-weight synapses
- **tsodyks_synapse**: Synapses with short-term plasticity
- **stdp_synapse**: Spike-timing dependent plasticity
- **stdp_triplet_synapse**: STDP with triplet rule
- **ht_synapse**: Hodgkin-Huxley type synapses

## Output Files

The training module generates several output files:

- `nest_training_results.json`: Training results and metrics
- `nest_performance_metrics.json`: Performance statistics
- `nest_neural_state.json`: Neural state information
- `nest_models.json`: Model configurations and parameters

## Testing

Run the test suite to verify functionality:

```bash
cd src/training
python test_nest_training.py
```

Or run with pytest:

```bash
pytest test_nest_training.py -v
```

## Examples

### Simple Network Simulation

```python
# Create a simple network with 1000 neurons
config = NESTConfig(num_neurons=1000)
trainer = NESTTrainer(config)

# Create and simulate network
model_data = trainer.create_network_model("simple_network")
results = await trainer.run_simulation("simple_network", model_data)
```

### Training with Multiple Epochs

```python
# Train network over multiple epochs
results = await trainer.train_network("training_network", epochs=50)

# Analyze results
for epoch_result in results:
    metrics = epoch_result["results"]["metrics"]
    print(f"Epoch {epoch_result['epoch']}: {metrics['spike_rate']:.2f} Hz")
```

## Integration with Brain Simulation Framework

The NEST training module integrates with the brain simulation framework:

- **Neural Dynamics**: Process simulation results through neural dynamics components
- **Cognitive Science**: Apply cognitive principles to simulation outcomes
- **Machine Learning**: Optimize network parameters using ML techniques
- **Sleep Consolidation**: Implement memory consolidation mechanisms

## Performance Optimization

### Threading Support

Enable multi-threading for faster simulations:

```python
config = NESTConfig(
    enable_threading=True,
    num_threads=4
)
```

### MPI Support

For distributed computing:

```python
config = NESTConfig(
    enable_mpi=True
)
```

## Troubleshooting

### Common Issues

1. **NEST Not Found**: Ensure NEST is properly installed and in your PATH
2. **Import Errors**: Check that all dependencies are installed
3. **Memory Issues**: Reduce network size or simulation time for large networks
4. **Performance**: Use threading or MPI for better performance

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To contribute to the NEST training module:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure compatibility with NEST versions

## References

- [NEST Simulator Documentation](https://www.nest-simulator.org/)
- [NEST GitHub Repository](https://github.com/nest/nest-simulator)
- [NEST Installation Guide](https://www.nest-simulator.org/installation/)
- [NEST User Manual](https://nest-simulator.readthedocs.io/)

## License

This module is part of the Brain Simulation Framework and is licensed under Apache-2.0.

## Support

For issues and questions:

1. Check the NEST documentation
2. Review the test files for examples
3. Open an issue in the project repository
4. Contact the development team
