#!/usr/bin/env python3
"""
NEST Brain Physics Simulator - Usage Example

This script demonstrates how to use the NEST-based brain physics simulator
for brain development and neural network research.

Usage:
    python nest_brain_simulation.py [--duration 1000] [--regions cortex,hippocampus,thalamus] [--connectivity hierarchical]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add conda base path to access NEST
conda_base = "/opt/homebrew/Caskroom/miniforge/base"
if conda_base not in sys.path:
    sys.path.insert(0, conda_base)

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from physics_simulation.brain_physics import BrainPhysicsSimulator
    print("✓ BrainPhysicsSimulator imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class BrainSimulationRunner:
    """Main class for running brain simulations"""
    
    def __init__(self, simulation_time=2000.0, time_step=0.1):
        """
        Initialize the simulation runner
        
        Args:
            simulation_time: Total simulation time in milliseconds
            time_step: Simulation time step in milliseconds
        """
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.simulator = None
        self.results = None
        
    def setup_simulation(self, brain_regions, cell_types, region_sizes=None):
        """
        Setup the brain development simulation
        
        Args:
            brain_regions: List of brain region names
            cell_types: List of cell type names
            region_sizes: Optional dict mapping regions to neuron counts
        """
        print(f"\n🧠 Setting up brain simulation...")
        print(f"  Regions: {', '.join(brain_regions)}")
        print(f"  Cell types: {', '.join(cell_types)}")
        
        # Create simulator
        self.simulator = BrainPhysicsSimulator(
            simulation_time=self.simulation_time,
            time_step=self.time_step
        )
        
        # Setup brain model
        success = self.simulator.setup_brain_development_model(
            brain_regions, cell_types, region_sizes
        )
        
        if not success:
            raise RuntimeError("Failed to setup brain development model")
        
        # Get initial metrics
        metrics = self.simulator.get_development_metrics()
        print(f"✓ Model ready: {metrics['total_neurons']:,} neurons in {metrics['region_count']} regions")
        
        return True
    
    def setup_connectivity(self, connectivity_pattern='hierarchical'):
        """Setup neural connectivity patterns"""
        print(f"🔗 Setting up {connectivity_pattern} connectivity...")
        self.simulator.setup_developmental_connectivity(connectivity_pattern)
        print("✓ Connectivity setup complete")
    
    def add_external_inputs(self, input_config):
        """
        Add external inputs to drive neural activity
        
        Args:
            input_config: Dict of {region: {cell_type: strength}}
        """
        print("⚡ Adding external inputs...")
        for region, cell_configs in input_config.items():
            for cell_type, strength in cell_configs.items():
                self.simulator.add_external_input(region, cell_type, strength)
                print(f"  ✓ {region}_{cell_type}: {strength} pA")
    
    def run_simulation(self, duration):
        """
        Run the brain development simulation
        
        Args:
            duration: Simulation duration in milliseconds
        """
        print(f"\n🚀 Running simulation for {duration}ms...")
        
        # Run simulation
        self.results = self.simulator.simulate_brain_growth(duration)
        
        if self.results:
            print("✓ Simulation completed successfully!")
            print(f"  - Total network spikes: {self.results['total_network_spikes']:,}")
            print(f"  - Average firing rate: {self.results['average_firing_rate']:.2f} Hz")
            print(f"  - Regions monitored: {len(self.results['region_activity'])}")
        else:
            print("❌ Simulation failed")
        
        return self.results
    
    def analyze_results(self):
        """Analyze and display simulation results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print(f"\n📊 Analysis Results:")
        print("=" * 40)
        
        # Activity by region
        print("Neural Activity by Brain Region:")
        for region_name, activity_data in self.results['region_activity'].items():
            print(f"  {region_name:20}: {activity_data['total_spikes']:>8,} spikes")
        
        # Network summary
        print(f"\nNetwork Summary:")
        print(f"  Total spikes: {self.results['total_network_spikes']:,}")
        print(f"  Average firing rate: {self.results['average_firing_rate']:.2f} Hz")
        print(f"  Development stage: {self.results['development_stage']:.2f}")
        
        # Firing rate analysis
        active_regions = [r for r, d in self.results['region_activity'].items() 
                         if d['total_spikes'] > 0]
        if active_regions:
            print(f"\nMost Active Regions:")
            for region in active_regions[:3]:  # Top 3
                data = self.results['region_activity'][region]
                print(f"  {region}: {data['total_spikes']:,} spikes")
    
    def create_visualizations(self, output_dir="outputs"):
        """Create and save visualization plots with Python API examples"""
        if not self.results:
            print("❌ No results to visualize")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📈 Creating visualizations in {output_path}")
        
        # Create enhanced plots
        self._plot_firing_rates_with_api_examples(output_path)
        self._plot_spike_raster_with_analysis(output_path)
        self._plot_activity_summary_with_metrics(output_path)
        self._plot_connectivity_diagram(output_path)
        self._create_api_usage_guide(output_path)
        
        print("✓ Enhanced visualizations and API guide saved successfully")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'simulator') and self.simulator:
            self.simulator.cleanup()
            print("✓ Resources cleaned up")
    
    def _plot_firing_rates_with_api_examples(self, output_path):
        """Plot firing rates over time with Python API usage examples"""
        plt.figure(figsize=(14, 8))
        
        # Main plot
        plt.subplot(2, 1, 1)
        for region_name, activity_data in self.results['region_activity'].items():
            if len(activity_data['firing_rates']) > 0:
                time_bins = np.arange(0, len(activity_data['firing_rates']) * 100, 100)
                plt.plot(time_bins, activity_data['firing_rates'], 
                        label=region_name, linewidth=2, alpha=0.8)
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('Neural Activity Over Time - Firing Rate Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add API usage example
        plt.subplot(2, 1, 2)
        plt.text(0.05, 0.9, 'Python API Usage Example:', fontsize=12, fontweight='bold',
                transform=plt.gca().transAxes)
        
        api_text = '''# Get firing rates for specific region
cortex_data = results['region_activity']['cortex_excitatory']
firing_rates = cortex_data['firing_rates']
spike_times = cortex_data['spike_times']

# Calculate average firing rate
avg_rate = np.mean(firing_rates)
print(f"Cortex excitatory avg rate: {avg_rate:.2f} Hz")

# Find peak activity time
peak_time = time_bins[np.argmax(firing_rates)]
print(f"Peak activity at: {peak_time} ms")'''
        
        plt.text(0.05, 0.7, api_text, fontsize=10, fontfamily='monospace',
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "firing_rates_with_api.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spike_raster_with_analysis(self, output_path):
        """Plot spike raster with analysis and API examples"""
        active_regions = [r for r, d in self.results['region_activity'].items() 
                         if d['total_spikes'] > 0]
        
        if not active_regions:
            # Create informative plot even with no spikes
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, 'No Spikes Detected\n\nPython API to investigate:', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
            
            api_text = '''# Check neuron parameters
simulator = BrainPhysicsSimulator()
# Add more external input
simulator.add_external_input('cortex', 'excitatory', 100.0)
# Run longer simulation
results = simulator.simulate_brain_growth(2000.0)'''
            
            plt.text(0.5, 0.3, api_text, ha='center', va='center', 
                    fontsize=10, fontfamily='monospace')
            plt.axis('off')
            
            plot_path = output_path / "spike_raster_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        plt.figure(figsize=(14, 10))
        
        # Spike raster plot
        plt.subplot(2, 1, 1)
        y_offset = 0
        for region_name in active_regions:
            activity_data = self.results['region_activity'][region_name]
            if len(activity_data['spike_times']) > 0:
                # Limit spikes for visualization
                spike_times = activity_data['spike_times'][:2000]
                spike_senders = activity_data['spike_senders'][:2000]
                
                plt.scatter(spike_times, spike_senders + y_offset, 
                           s=1, alpha=0.6, label=region_name)
                y_offset += 1000
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        plt.title('Spike Raster Plot - Neural Activity Patterns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # API usage examples
        plt.subplot(2, 1, 2)
        plt.text(0.05, 0.9, 'Spike Analysis API Examples:', fontsize=12, fontweight='bold',
                transform=plt.gca().transAxes)
        
        api_text = '''# Analyze spike patterns
for region, data in results['region_activity'].items():
    if data['total_spikes'] > 0:
        spike_times = data['spike_times']
        spike_senders = data['spike_senders']
        
        # Calculate inter-spike intervals
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            avg_isi = np.mean(isi)
            print(f"{region}: Avg ISI = {avg_isi:.2f} ms")
        
        # Find most active neurons
        unique_neurons, counts = np.unique(spike_senders, return_counts=True)
        most_active = unique_neurons[np.argmax(counts)]
        print(f"{region}: Most active neuron {most_active}")'''
        
        plt.text(0.05, 0.6, api_text, fontsize=9, fontfamily='monospace',
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "spike_raster_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_activity_summary_with_metrics(self, output_path):
        """Plot activity summary with detailed metrics and API examples"""
        plt.figure(figsize=(16, 10))
        
        # Activity summary bar chart
        plt.subplot(2, 2, 1)
        regions = list(self.results['region_activity'].keys())
        spike_counts = [self.results['region_activity'][r]['total_spikes'] for r in regions]
        
        bars = plt.bar(range(len(regions)), spike_counts, alpha=0.7, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.xlabel('Brain Region')
        plt.ylabel('Total Spikes')
        plt.title('Total Neural Activity by Region')
        plt.xticks(range(len(regions)), regions, rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, spike_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spike_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom')
        
        # Network metrics
        plt.subplot(2, 2, 2)
        metrics = [
            self.results['total_network_spikes'],
            self.results['average_firing_rate'],
            len(self.results['region_activity']),
            self.results['development_stage']
        ]
        metric_labels = ['Total Spikes', 'Avg Firing Rate (Hz)', 'Regions', 'Development Stage']
        
        bars = plt.bar(range(len(metrics)), metrics, alpha=0.7, color='green')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Network Summary Metrics')
        plt.xticks(range(len(metrics)), metric_labels, rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics)*0.01,
                    f'{value:.2f}' if isinstance(value, float) else f'{value:,}', 
                    ha='center', va='bottom')
        
        # API usage examples
        plt.subplot(2, 2, 3)
        plt.text(0.05, 0.9, 'Activity Analysis API:', fontsize=12, fontweight='bold',
                transform=plt.gca().transAxes)
        
        api_text = '''# Get activity statistics
total_spikes = results['total_network_spikes']
avg_rate = results['average_firing_rate']

# Find most active regions
active_regions = [(r, d['total_spikes']) 
                 for r, d in results['region_activity'].items()]
active_regions.sort(key=lambda x: x[1], reverse=True)

print("Most Active Regions:")
for region, spikes in active_regions[:3]:
    print(f"  {region}: {spikes:,} spikes")'''
        
        plt.text(0.05, 0.6, api_text, fontsize=9, fontfamily='monospace',
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        
        # Development analysis
        plt.subplot(2, 2, 4)
        plt.text(0.05, 0.9, 'Development Analysis API:', fontsize=12, fontweight='bold',
                transform=plt.gca().transAxes)
        
        dev_text = '''# Analyze development progress
dev_stage = results['development_stage']
time_points = results['time_points']

# Calculate development rate
if len(time_points) > 1:
    dev_rate = dev_stage / (time_points[-1] / 1000.0)
    print(f"Development rate: {dev_rate:.3f} per second")

# Check region maturation
for region, data in results['region_activity'].items():
    if data['total_spikes'] > 0:
        print(f"{region}: Active (mature)")
    else:
        print(f"{region}: Inactive (developing)")'''
        
        plt.text(0.05, 0.6, dev_text, fontsize=9, fontfamily='monospace',
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "activity_summary_with_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_connectivity_diagram(self, output_path):
        """Create a connectivity diagram showing network structure"""
        plt.figure(figsize=(12, 8))
        
        # Create a simple network diagram
        plt.subplot(1, 2, 1)
        
        # Define positions for brain regions
        region_positions = {
            'cortex': (0.2, 0.8),
            'hippocampus': (0.8, 0.8),
            'thalamus': (0.2, 0.2),
            'cerebellum': (0.8, 0.2)
        }
        
        # Plot regions - get from simulator if available
        brain_regions = []
        if hasattr(self, 'simulator') and self.simulator:
            brain_regions = list(self.simulator.brain_regions.keys())
        else:
            # Fallback to common regions
            brain_regions = ['cortex', 'hippocampus', 'thalamus', 'cerebellum']
        
        for region, pos in region_positions.items():
            if region in brain_regions:
                plt.scatter(pos[0], pos[1], s=200, alpha=0.7, label=region)
                plt.text(pos[0], pos[1], region, ha='center', va='center', fontweight='bold')
        
        # Add connections (simplified)
        plt.plot([0.2, 0.8], [0.8, 0.8], 'k--', alpha=0.3, label='Inter-region connections')
        plt.plot([0.2, 0.2], [0.8, 0.2], 'k--', alpha=0.3)
        plt.plot([0.8, 0.8], [0.8, 0.2], 'k--', alpha=0.3)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Brain Region Connectivity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # API usage for connectivity
        plt.subplot(1, 2, 2)
        plt.text(0.05, 0.9, 'Connectivity Setup API:', fontsize=12, fontweight='bold',
                transform=plt.gca().transAxes)
        
        conn_text = '''# Setup different connectivity patterns
simulator.setup_developmental_connectivity('hierarchical')
# Options: 'random', 'lateral', 'hierarchical'

# Custom connectivity setup
for region in brain_regions:
    # Connect excitatory to inhibitory
    if 'excitatory' in simulator.brain_regions[region]['neurons']:
        excitatory = simulator.brain_regions[region]['neurons']['excitatory']
        if 'inhibitory' in simulator.brain_regions[region]['neurons']:
            inhibitory = simulator.brain_regions[region]['neurons']['inhibitory']
            nest.Connect(excitatory, inhibitory, 
                        {'rule': 'pairwise_bernoulli', 'p': 0.3})'''
        
        plt.text(0.05, 0.6, conn_text, fontsize=9, fontfamily='monospace',
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "connectivity_diagram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_api_usage_guide(self, output_path):
        """Create a comprehensive API usage guide"""
        guide_path = output_path / "python_api_usage_guide.md"
        
        guide_content = f"""# Python API Usage Guide for NEST Brain Physics Simulator

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
region_sizes = {{
    'cortex': 2000,      # 2000 neurons
    'hippocampus': 1500, # 1500 neurons
    'thalamus': 1000     # 1000 neurons
}}

simulator.setup_brain_development_model(
    brain_regions, cell_types, region_sizes
)
```

### Multiple External Inputs
```python
input_config = {{
    'cortex': {{'excitatory': 60.0, 'inhibitory': 20.0}},
    'hippocampus': {{'excitatory': 40.0}},
    'thalamus': {{'excitatory': 30.0}}
}}

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

print(f"Total spikes: {{total_spikes:,}}")
print(f"Average firing rate: {{avg_firing_rate:.2f}} Hz")
print(f"Development stage: {{development_stage:.2f}}")
```

### Region-Specific Analysis
```python
# Analyze activity by region
for region_name, activity_data in results['region_activity'].items():
    spike_times = activity_data['spike_times']
    spike_senders = activity_data['spike_senders']
    firing_rates = activity_data['firing_rates']
    total_spikes = activity_data['total_spikes']
    
    print(f"\\n{{region_name}}:")
    print(f"  Total spikes: {{total_spikes:,}}")
    print(f"  Average firing rate: {{np.mean(firing_rates):.2f}} Hz")
    
    if len(spike_times) > 0:
        # Calculate inter-spike intervals
        isi = np.diff(spike_times)
        avg_isi = np.mean(isi)
        print(f"  Average ISI: {{avg_isi:.2f}} ms")
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
        
        print(f"{{region_name}}: Peak rate {{peak_rate:.2f}} Hz at {{peak_time}} ms")
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
                        {{'rule': 'pairwise_bernoulli', 'p': 0.1}})
        
        # Connect to other regions
        for other_region, other_data in simulator.brain_regions.items():
            if other_region != region_name:
                for other_cell_type, other_neurons in other_data['neurons'].items():
                    nest.Connect(neurons, other_neurons,
                                {{'rule': 'pairwise_bernoulli', 'p': 0.05}})
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

print(f"Development stage: {{metrics['development_stage']:.2f}}")
print(f"Total neurons: {{metrics['total_neurons']:,}}")
print(f"Brain regions: {{metrics['region_count']}}")
print(f"Cell types: {{metrics['cell_population_types']}}")
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
results_by_strength = {{}}

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
results_by_pattern = {{}}

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
    print(f"{{pattern}}: {{rate:.2f}} Hz")
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

**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Simulation Results**: {self.results.get('total_network_spikes', 0):,} total spikes, {self.results.get('average_firing_rate', 0.0):.2f} Hz avg rate
**Brain Regions**: {len(self.results.get('region_activity', {}))} regions monitored
"""
        
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"✓ API usage guide saved to {guide_path}")


def run_example_simulation():
    """Run a complete example simulation"""
    print("🧠 NEST Brain Physics Simulator - Example Simulation")
    print("=" * 60)
    
    # Configuration
    brain_regions = ['cortex', 'hippocampus', 'thalamus', 'cerebellum']
    cell_types = ['excitatory', 'inhibitory']
    region_sizes = {
        'cortex': 2000,      # Larger cortex
        'hippocampus': 1500, # Medium hippocampus
        'thalamus': 1000,    # Smaller thalamus
        'cerebellum': 1200   # Medium cerebellum
    }
    
    # Input configuration to drive activity
    input_config = {
        'cortex': {'excitatory': 60.0, 'inhibitory': 20.0},
        'hippocampus': {'excitatory': 40.0},
        'thalamus': {'excitatory': 30.0}
    }
    
    # Create and run simulation
    runner = BrainSimulationRunner(simulation_time=3000.0, time_step=0.1)
    
    try:
        # Setup
        runner.setup_simulation(brain_regions, cell_types, region_sizes)
        runner.setup_connectivity('hierarchical')
        runner.add_external_inputs(input_config)
        
        # Run simulation
        results = runner.run_simulation(2000.0)  # 2 second simulation
        
        if results:
            # Analyze results
            runner.analyze_results()
            
            # Create visualizations
            runner.create_visualizations()
        
        print("\n🎉 Example simulation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        runner.cleanup()


def run_custom_simulation(args):
    """Run a custom simulation based on command line arguments"""
    print("🧠 NEST Brain Physics Simulator - Custom Simulation")
    print("=" * 60)
    
    # Parse regions
    brain_regions = [r.strip() for r in args.regions.split(',')]
    cell_types = ['excitatory', 'inhibitory']
    
    # Create and run simulation
    runner = BrainSimulationRunner(simulation_time=2000.0, time_step=0.1)
    
    try:
        # Setup
        runner.setup_simulation(brain_regions, cell_types)
        runner.setup_connectivity(args.connectivity)
        
        # Add some default inputs
        input_config = {region: {'excitatory': 50.0} for region in brain_regions}
        runner.add_external_inputs(input_config)
        
        # Run simulation
        results = runner.run_simulation(args.duration)
        
        if results:
            runner.analyze_results()
            runner.create_visualizations()
        
        print("\n🎉 Custom simulation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        runner.cleanup()


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="NEST Brain Physics Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nest_brain_simulation.py                    # Run example simulation
  python nest_brain_simulation.py --duration 1500   # Custom duration
  python nest_brain_simulation.py --regions cortex,hippocampus --connectivity lateral
        """
    )
    
    parser.add_argument('--duration', type=int, default=1000,
                       help='Simulation duration in milliseconds (default: 1000)')
    parser.add_argument('--regions', type=str, 
                       default='cortex,hippocampus,thalamus',
                       help='Comma-separated brain regions (default: cortex,hippocampus,thalamus)')
    parser.add_argument('--connectivity', type=str, 
                       choices=['random', 'lateral', 'hierarchical'],
                       default='hierarchical',
                       help='Connectivity pattern (default: hierarchical)')
    parser.add_argument('--example', action='store_true',
                       help='Run the complete example simulation')
    
    args = parser.parse_args()
    
    if args.example:
        run_example_simulation()
    else:
        run_custom_simulation(args)


if __name__ == "__main__":
    main()
