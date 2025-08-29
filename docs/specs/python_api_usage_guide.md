# Python API Usage Guide for NEST Brain Physics Simulator

## 🚀 Quick Start Example

```python
from src.physics_simulation.brain_physics import BrainPhysicsSimulator

# Create simulator
simulator = BrainPhysicsSimulator(simulation_time=2000.0, time_step=0.1)

# Setup brain model
brain_regions = ['cortex', 'hippocampus', 'thalamus']
cell_types = ['excitatory', 'inhibitory']
simulator.setup_brain_development_model(brain_regions, cell_types)

# Setup connectivity
simulator.setup_developmental_connectivity('hierarchical')

# Add external input
simulator.add_external_input('cortex', 'excitatory', input_strength=50.0)

# Run simulation
results = simulator.simulate_brain_growth(1000.0)

# Cleanup
simulator.cleanup()
```

## 🧠 Advanced Configuration

### Custom Region Sizes
```python
region_sizes = {
    'cortex': 2000,      # 2000 neurons
    'hippocampus': 1500, # 1500 neurons
    'thalamus': 1000     # 1000 neurons
}

simulator.setup_brain_development_model(
    brain_regions, cell_types, region_sizes
)
```

### Multiple External Inputs
```python
input_config = {
    'cortex': {'excitatory': 60.0, 'inhibitory': 20.0},
    'hippocampus': {'excitatory': 40.0},
    'thalamus': {'excitatory': 30.0}
}

for region, cell_configs in input_config.items():
    for cell_type, strength in cell_configs.items():
        simulator.add_external_input(region, cell_type, strength)
```

## 📊 Results Analysis

### Basic Activity Analysis
```python
# Get overall network statistics
total_spikes = results['total_network_spikes']
avg_firing_rate = results['average_firing_rate']
development_stage = results['development_stage']

print(f"Total spikes: {total_spikes:,}")
print(f"Average firing rate: {avg_firing_rate:.2f} Hz")
print(f"Development stage: {development_stage:.2f}")
```

### Region-Specific Analysis
```python
# Analyze activity by region
for region_name, activity_data in results['region_activity'].items():
    spike_times = activity_data['spike_times']
    spike_senders = activity_data['spike_senders']
    firing_rates = activity_data['firing_rates']
    total_spikes = activity_data['total_spikes']
    
    print(f"\n{region_name}:")
    print(f"  Total spikes: {total_spikes:,}")
    print(f"  Average firing rate: {np.mean(firing_rates):.2f} Hz")
    
    if len(spike_times) > 0:
        # Calculate inter-spike intervals
        isi = np.diff(spike_times)
        avg_isi = np.mean(isi)
        print(f"  Average ISI: {avg_isi:.2f} ms")
```

### Time-Series Analysis
```python
# Get time points for analysis
time_points = results['time_points']

# Analyze firing rate evolution
for region_name, activity_data in results['region_activity'].items():
    if len(activity_data['firing_rates']) > 0:
        firing_rates = activity_data['firing_rates']
        
        # Find peak activity
        peak_rate = np.max(firing_rates)
        peak_time_idx = np.argmax(firing_rates)
        peak_time = time_points[peak_time_idx * 100]  # Convert to ms
        
        print(f"{region_name}: Peak rate {peak_rate:.2f} Hz at {peak_time} ms")
```

## 🔗 Connectivity Patterns

### Available Patterns
- **random**: Random synaptic connections between all neurons
- **lateral**: Lateral inhibitory connections within regions
- **hierarchical**: Feedforward connections between regions

### Custom Connectivity
```python
# Manual connection setup
for region_name, region_data in simulator.brain_regions.items():
    for cell_type, neurons in region_data['neurons'].items():
        # Connect neurons within the same population
        if len(neurons) > 1:
            nest.Connect(neurons, neurons, 
                        {'rule': 'pairwise_bernoulli', 'p': 0.1})
        
        # Connect to other regions
        for other_region, other_data in simulator.brain_regions.items():
            if other_region != region_name:
                for other_cell_type, other_neurons in other_data['neurons'].items():
                    nest.Connect(neurons, other_neurons,
                                {'rule': 'pairwise_bernoulli', 'p': 0.05})
```

## ⚡ External Input Management

### Adding Inputs
```python
# Single input
simulator.add_external_input('cortex', 'excitatory', input_strength=50.0)

# Multiple inputs to same region
simulator.add_external_input('cortex', 'excitatory', input_strength=60.0)
simulator.add_external_input('cortex', 'inhibitory', input_strength=20.0)

# Inputs to multiple regions
regions = ['cortex', 'hippocampus', 'thalamus']
for region in regions:
    simulator.add_external_input(region, 'excitatory', input_strength=40.0)
```

### Input Types
- **DC Generator**: Constant current input
- **Spike Generator**: Preset spike patterns
- **Poisson Generator**: Random spike trains

## 📈 Development Metrics

### Get Current Status
```python
metrics = simulator.get_development_metrics()

print(f"Development stage: {metrics['development_stage']:.2f}")
print(f"Total neurons: {metrics['total_neurons']:,}")
print(f"Brain regions: {metrics['region_count']}")
print(f"Cell types: {metrics['cell_population_types']}")
```

### Monitor Development Progress
```python
# Run multiple simulation stages
stages = [500, 1000, 1500, 2000]  # ms
development_progress = []

for stage in stages:
    results = simulator.simulate_brain_growth(stage)
    development_progress.append(results['development_stage'])

# Plot development curve
import matplotlib.pyplot as plt
plt.plot(stages, development_progress)
plt.xlabel('Time (ms)')
plt.ylabel('Development Stage')
plt.title('Brain Development Progress')
plt.show()
```

## 🧹 Resource Management

### Proper Cleanup
```python
try:
    # Run simulation
    results = simulator.simulate_brain_growth(1000.0)
    
    # Analyze results
    # ... your analysis code ...
    
finally:
    # Always cleanup
    simulator.cleanup()
```

### Memory Management
```python
# For long simulations, consider resetting periodically
if simulation_time > 5000:  # 5 seconds
    simulator.cleanup()
    simulator = BrainPhysicsSimulator(simulation_time, time_step)
    # Re-setup model and connectivity
```

## 🔬 Research Applications

### Parameter Sweeps
```python
# Test different input strengths
input_strengths = [10, 30, 50, 70, 90]
results_by_strength = {}

for strength in input_strengths:
    simulator = BrainPhysicsSimulator()
    simulator.setup_brain_development_model(['cortex'], ['excitatory'])
    simulator.add_external_input('cortex', 'excitatory', strength)
    
    results = simulator.simulate_brain_growth(1000.0)
    results_by_strength[strength] = results['total_network_spikes']
    
    simulator.cleanup()

# Plot results
plt.plot(input_strengths, list(results_by_strength.values()))
plt.xlabel('Input Strength (pA)')
plt.ylabel('Total Spikes')
plt.title('Input-Output Relationship')
plt.show()
```

### Connectivity Comparison
```python
patterns = ['random', 'lateral', 'hierarchical']
results_by_pattern = {}

for pattern in patterns:
    simulator = BrainPhysicsSimulator()
    simulator.setup_brain_development_model(['cortex', 'hippocampus'], ['excitatory', 'inhibitory'])
    simulator.setup_developmental_connectivity(pattern)
    simulator.add_external_input('cortex', 'excitatory', 50.0)
    
    results = simulator.simulate_brain_growth(1000.0)
    results_by_pattern[pattern] = results['average_firing_rate']
    
    simulator.cleanup()

# Compare results
for pattern, rate in results_by_pattern.items():
    print(f"{pattern}: {rate:.2f} Hz")
```

## 🚨 Troubleshooting

### Common Issues
1. **No spikes detected**: Increase input strength or simulation duration
2. **Memory issues**: Reduce neuron counts or use cleanup()
3. **Import errors**: Ensure you're in the project root directory
4. **NEST errors**: Check NEST installation and version compatibility

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run simulation with detailed logging
simulator = BrainPhysicsSimulator()
# ... setup and run ...
```

---

**Generated on**: 2025-08-17 02:13:20
**Simulation Results**: 0 total spikes, 0.00 Hz avg rate
**Brain Regions**: 8 regions monitored
