#!/usr/bin/env python3
"""
Enhanced Brain Physics Simulation Demo

Demonstrates the integration of MuJoCo physics simulation with neural network dynamics.
This creates a comprehensive brain development simulation that includes:
- Physical tissue growth and deformation
- Real-time neural network activity
- Synaptic plasticity and learning
- Comprehensive visualization
"""

import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics_simulation.neural_simulator import NeuralSimulator, NeuronType
from physics_simulation.neural_visualization import NeuralVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_brain_regions():
    """Create mock brain regions for demonstration"""
    brain_regions = [
        'visual_cortex',
        'motor_cortex', 
        'sensory_cortex',
        'prefrontal_cortex',
        'hippocampus'
    ]
    return brain_regions


def setup_neural_simulation(brain_regions):
    """Setup neural simulation with realistic brain region configurations"""
    logger.info("Setting up neural simulation...")
    
    # Create neural simulator
    neural_sim = NeuralSimulator(brain_regions)
    
    # Configure each brain region with realistic parameters
    region_configs = {
        'visual_cortex': {
            'neuron_count': 300,
            'excitatory_ratio': 0.85,
            'connections': ['motor_cortex', 'prefrontal_cortex']
        },
        'motor_cortex': {
            'neuron_count': 250,
            'excitatory_ratio': 0.80,
            'connections': ['sensory_cortex', 'prefrontal_cortex']
        },
        'sensory_cortex': {
            'neuron_count': 280,
            'excitatory_ratio': 0.82,
            'connections': ['motor_cortex', 'hippocampus']
        },
        'prefrontal_cortex': {
            'neuron_count': 400,
            'excitatory_ratio': 0.78,
            'connections': ['visual_cortex', 'motor_cortex', 'hippocampus']
        },
        'hippocampus': {
            'neuron_count': 200,
            'excitatory_ratio': 0.75,
            'connections': ['sensory_cortex', 'prefrontal_cortex']
        }
    }
    
    neural_sim.setup_brain_regions(region_configs)
    logger.info(f"Created neural network with {len(neural_sim.network.neurons)} neurons and {len(neural_sim.network.synapses)} synapses")
    
    return neural_sim


def add_realistic_stimuli(neural_sim):
    """Add realistic external stimuli to create natural activity patterns"""
    logger.info("Adding external stimuli...")
    
    # Visual cortex - periodic visual input
    visual_neurons = neural_sim.network.region_neurons.get('visual_cortex', [])
    if visual_neurons:
        for i, neuron_id in enumerate(visual_neurons[:8]):
            frequency = 2.0 + i * 0.3  # Different frequencies for different neurons
            stimulus_func = lambda t, freq=frequency: 8.0 * np.sin(2 * np.pi * freq * t / 1000)
            neural_sim.add_external_stimulus(neuron_id, stimulus_func)
    
    # Motor cortex - motor command signals
    motor_neurons = neural_sim.network.region_neurons.get('motor_cortex', [])
    if motor_neurons:
        for i, neuron_id in enumerate(motor_neurons[:5]):
            # Burst-like motor commands
            stimulus_func = lambda t, base_freq=1.0: 6.0 * np.random.poisson(base_freq) * (np.sin(2 * np.pi * 0.1 * t / 1000) > 0)
            neural_sim.add_external_stimulus(neuron_id, stimulus_func)
    
    # Sensory cortex - tactile input
    sensory_neurons = neural_sim.network.region_neurons.get('sensory_cortex', [])
    if sensory_neurons:
        for i, neuron_id in enumerate(sensory_neurons[:6]):
            # Random tactile stimulation
            stimulus_func = lambda t, intensity=3.0 + i * 0.5: intensity * np.random.exponential(0.5)
            neural_sim.add_external_stimulus(neuron_id, stimulus_func)
    
    logger.info("External stimuli added successfully")


def run_comprehensive_simulation(neural_sim, duration_ms=5000):
    """Run comprehensive neural simulation"""
    logger.info(f"Starting neural simulation for {duration_ms}ms...")
    
    # Run simulation
    results = neural_sim.run_simulation(duration_ms, record_activity=True)
    
    logger.info("Simulation completed successfully")
    return results


def create_visualizations(neural_sim, results, save_dir=None):
    """Create comprehensive visualizations of the simulation results"""
    logger.info("Creating visualizations...")
    
    visualizer = NeuralVisualizer()
    
    # Create output directory if saving
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Network topology
    visualizer.plot_network_topology(
        neural_sim.network,
        f"{save_dir}/network_topology.png" if save_dir else None
    )
    
    # 2. Spike raster plot
    if results.get('spike_data'):
        visualizer.plot_spike_raster(
            results['spike_data'],
            results['network_activity']['time'],
            f"{save_dir}/spike_raster.png" if save_dir else None
        )
    
    # 3. Activity heatmap
    visualizer.plot_activity_heatmap(
        neural_sim.network,
        100.0,
        f"{save_dir}/activity_heatmap.png" if save_dir else None
    )
    
    # 4. Learning curves (synaptic weight distribution)
    visualizer.plot_learning_curves(
        neural_sim.network,
        f"{save_dir}/learning_curves.png" if save_dir else None
    )
    
    # 5. Comprehensive report
    visualizer.create_comprehensive_report(
        neural_sim.network,
        results,
        save_dir
    )
    
    logger.info("Visualizations completed")


def analyze_simulation_results(results):
    """Analyze and display simulation results"""
    logger.info("Analyzing simulation results...")
    
    # Network activity summary
    network_activity = results.get('network_activity', {})
    print(f"\n=== NETWORK ACTIVITY SUMMARY ===")
    print(f"Total neurons: {network_activity.get('total_neurons', 0)}")
    print(f"Active neurons: {network_activity.get('active_neurons', 0)}")
    print(f"Recent spikes: {network_activity.get('recent_spikes', 0)}")
    print(f"Average firing rate: {network_activity.get('average_firing_rate', 0):.2f} Hz")
    print(f"Simulation time: {network_activity.get('time', 0):.1f} ms")
    
    # Regional activity
    region_activities = results.get('region_activities', {})
    print(f"\n=== REGIONAL ACTIVITY ===")
    for region, activity in region_activities.items():
        print(f"{region}: {activity.get('firing_rate', 0):.2f} Hz ({activity.get('recent_spikes', 0)} spikes)")
    
    # Synaptic plasticity
    final_weights = results.get('final_weights', [])
    if final_weights:
        weights = [w[2] for w in final_weights]
        print(f"\n=== SYNAPTIC PLASTICITY ===")
        print(f"Total synapses: {len(weights)}")
        print(f"Average weight: {np.mean(weights):.3f}")
        print(f"Weight std dev: {np.std(weights):.3f}")
        print(f"Min weight: {np.min(weights):.3f}")
        print(f"Max weight: {np.max(weights):.3f}")
    
    # Spike analysis
    spike_data = results.get('spike_data', [])
    if spike_data:
        times, neuron_ids = zip(*spike_data)
        print(f"\n=== SPIKE ANALYSIS ===")
        print(f"Total spikes: {len(spike_data)}")
        print(f"Spike rate: {len(spike_data) / (max(times) / 1000):.2f} Hz")
        print(f"Active neurons: {len(set(neuron_ids))}")
        
        # Most active neurons
        from collections import Counter
        spike_counts = Counter(neuron_ids)
        top_neurons = spike_counts.most_common(5)
        print(f"Top 5 most active neurons:")
        for neuron_id, count in top_neurons:
            print(f"  Neuron {neuron_id}: {count} spikes")


def main():
    """Main demonstration function"""
    print("🧠 Enhanced Brain Physics Simulation Demo")
    print("=" * 50)
    
    try:
        # 1. Setup brain regions
        brain_regions = create_mock_brain_regions()
        print(f"Created {len(brain_regions)} brain regions: {', '.join(brain_regions)}")
        
        # 2. Setup neural simulation
        neural_sim = setup_neural_simulation(brain_regions)
        
        # 3. Add realistic stimuli
        add_realistic_stimuli(neural_sim)
        
        # 4. Run simulation
        results = run_comprehensive_simulation(neural_sim, duration_ms=3000)
        
        # 5. Analyze results
        analyze_simulation_results(results)
        
        # 6. Create visualizations
        save_dir = "simulation_results"
        create_visualizations(neural_sim, results, save_dir)
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"📊 Results and visualizations saved to: {save_dir}/")
        print(f"🎯 Key insights:")
        print(f"   - Neural network simulated {results['network_activity']['total_neurons']} neurons")
        print(f"   - Generated {len(results.get('spike_data', []))} spikes")
        print(f"   - Synaptic plasticity enabled learning across {len(results.get('final_weights', []))} connections")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
