#!/usr/bin/env python3
"""
Debug neural dynamics to understand why firing rates are low
"""

import sys
import os
# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'src'))

from core.neural_components import SpikingNeuron, NeuralPopulation

def test_single_neuron():
    """Test single neuron with different inputs"""
    print("🧠 Testing Single Neuron...")
    
    neuron = SpikingNeuron(0, "regular_spiking")
    
    # Test with different input strengths
    for input_strength in [5, 10, 20, 30, 50, 100]:
        print(f"\n  Input strength: {input_strength}")
        
        spike_count = 0
        for i in range(100):
            if neuron.step(input_strength, 1.0, i):
                spike_count += 1
        
        print(f"    Spikes: {spike_count}")
        print(f"    Firing rate: {neuron.get_firing_rate():.2f} Hz")
        print(f"    Membrane potential: {neuron.get_membrane_potential():.2f} mV")

def test_population():
    """Test neural population"""
    print("\n🧠 Testing Neural Population...")
    
    population = NeuralPopulation("test", 10, "regular_spiking", 0.3)
    
    # Test with different input strengths
    for input_strength in [10, 20, 30, 50]:
        print(f"\n  Input strength: {input_strength}")
        
        # Run multiple steps to allow synaptic interactions
        total_spikes = 0
        for step in range(100):
            external_inputs = [input_strength] * 10
            spike_events = population.step(external_inputs, 1.0, step)
            total_spikes += sum(spike_events)
        
        print(f"    Total spikes over 100 steps: {total_spikes}")
        print(f"    Average firing rate: {population.get_population_firing_rate():.2f} Hz")
        print(f"    Individual firing rates: {[f'{r:.2f}' for r in population.get_neuron_firing_rates()]}")

if __name__ == "__main__":
    test_single_neuron()
    test_population()
