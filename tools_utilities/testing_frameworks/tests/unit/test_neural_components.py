#!/usr/bin/env python3
"""
🧠 Unit Tests for Neural Components
Tests individual neural components with simulation validation

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Unit testing of neural components
**Validation Level:** Component-level functional verification
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.neural_components import (
    SpikingNeuron, HebbianSynapse, STDP, NeuralPopulation,
    calculate_synchrony, calculate_oscillation_power, SpikeEvent
)

class TestSpikingNeuron:
    """Test spiking neuron functionality"""
    
    def test_neuron_initialization(self):
        """Test neuron initialization with different types"""
        # Test regular spiking neuron
        neuron = SpikingNeuron(0, "regular_spiking")
        assert neuron.neuron_id == 0
        assert neuron.neuron_type == "regular_spiking"
        assert neuron.v == -65.0  # Initial membrane potential
        assert neuron.u == 0.0    # Initial recovery variable
        
        # Test fast spiking neuron
        fast_neuron = SpikingNeuron(1, "fast_spiking")
        assert fast_neuron.a == 0.1  # Different parameter for fast spiking
        
        # Test bursting neuron
        burst_neuron = SpikingNeuron(2, "bursting")
        assert burst_neuron.c == -50.0  # Different reset potential
    
    def test_neuron_spiking(self):
        """Test neuron spiking behavior"""
        neuron = SpikingNeuron(0, "regular_spiking")
        
        # Test with strong input (should spike)
        spike_count = 0
        for i in range(100):
            if neuron.step(50.0, 1.0, i):
                spike_count += 1
        
        # Should have spiked multiple times
        assert spike_count > 0
        assert neuron.get_firing_rate() > 0.0
    
    def test_neuron_no_spiking(self):
        """Test neuron behavior with weak input"""
        neuron = SpikingNeuron(0, "regular_spiking")
        
        # Test with weak input (should not spike)
        spike_count = 0
        for i in range(100):
            if neuron.step(1.0, 1.0, i):
                spike_count += 1
        
        # Should not spike with weak input
        assert spike_count == 0
        assert neuron.get_firing_rate() == 0.0
    
    def test_membrane_potential_dynamics(self):
        """Test membrane potential changes"""
        neuron = SpikingNeuron(0, "regular_spiking")
        initial_v = neuron.get_membrane_potential()
        
        # Step neuron with moderate input
        neuron.step(10.0, 1.0, 0.0)
        new_v = neuron.get_membrane_potential()
        
        # Membrane potential should change
        assert new_v != initial_v
    
    def test_spike_history(self):
        """Test spike history recording"""
        neuron = SpikingNeuron(0, "regular_spiking")
        
        # Generate some spikes
        for i in range(100):
            neuron.step(50.0, 1.0, i)
        
        # Should have recorded spikes
        assert len(neuron.spike_history) > 0
        
        # Check spike event structure
        if neuron.spike_history:
            spike = neuron.spike_history[0]
            assert isinstance(spike, SpikeEvent)
            assert spike.neuron_id == 0
            assert spike.timestamp >= 0
            assert spike.strength == 1.0

class TestHebbianSynapse:
    """Test Hebbian synapse functionality"""
    
    def test_synapse_initialization(self):
        """Test synapse initialization"""
        synapse = HebbianSynapse(0, 1, initial_weight=0.5)
        assert synapse.pre_neuron_id == 0
        assert synapse.post_neuron_id == 1
        assert synapse.weight == 0.5
        assert synapse.learning_rate == 0.01
    
    def test_hebbian_learning(self):
        """Test Hebbian learning rule"""
        synapse = HebbianSynapse(0, 1, initial_weight=0.5)
        
        # Test LTP (both neurons active)
        old_weight = synapse.get_weight()
        synapse.update_hebbian(True, True, 10.0, 8.0)
        new_weight = synapse.get_weight()
        
        # Weight should increase
        assert new_weight > old_weight
    
    def test_anti_hebbian_learning(self):
        """Test anti-Hebbian learning"""
        synapse = HebbianSynapse(0, 1, initial_weight=0.5)
        
        # Test anti-Hebbian (only pre-synaptic active)
        old_weight = synapse.get_weight()
        synapse.update_hebbian(True, False, 10.0, 0.0)
        new_weight = synapse.get_weight()
        
        # Weight should decrease
        assert new_weight < old_weight
    
    def test_weight_bounds(self):
        """Test weight bounds enforcement"""
        synapse = HebbianSynapse(0, 1, initial_weight=0.5)
        
        # Try to increase weight beyond bounds
        for _ in range(1000):
            synapse.update_hebbian(True, True, 10.0, 10.0)
        
        # Weight should be at maximum
        assert synapse.get_weight() <= synapse.max_weight
        
        # Try to decrease weight below bounds
        synapse.weight = 0.0
        synapse.update_hebbian(True, False, 10.0, 0.0)
        
        # Weight should not go below minimum
        assert synapse.get_weight() >= synapse.min_weight
    
    def test_weight_history(self):
        """Test weight history tracking"""
        synapse = HebbianSynapse(0, 1, initial_weight=0.5)
        
        # Make some weight changes
        for _ in range(10):
            synapse.update_hebbian(True, True, 5.0, 5.0)
        
        # Should have weight history
        history = synapse.get_weight_history()
        assert len(history) > 1
        assert all(isinstance(w, (int, float)) for w in history)

class TestSTDP:
    """Test STDP functionality"""
    
    def test_stdp_initialization(self):
        """Test STDP initialization"""
        stdp = STDP()
        assert stdp.tau_plus == 20.0
        assert stdp.tau_minus == 20.0
        assert stdp.A_plus == 0.01
        assert stdp.A_minus == 0.01
    
    def test_ltp_calculation(self):
        """Test LTP weight change calculation"""
        stdp = STDP()
        
        # Post-synaptic spike after pre-synaptic (LTP)
        dw = stdp.calculate_weight_change(10.0, 15.0)
        
        # Should be positive (LTP)
        assert dw > 0
        assert dw <= stdp.A_plus  # Should not exceed maximum
    
    def test_ltd_calculation(self):
        """Test LTD weight change calculation"""
        stdp = STDP()
        
        # Pre-synaptic spike after post-synaptic (LTD)
        dw = stdp.calculate_weight_change(15.0, 10.0)
        
        # Should be negative (LTD)
        assert dw < 0
        assert abs(dw) <= stdp.A_minus  # Should not exceed maximum
    
    def test_simultaneous_spikes(self):
        """Test simultaneous spike handling"""
        stdp = STDP()
        
        # Simultaneous spikes
        dw = stdp.calculate_weight_change(10.0, 10.0)
        
        # Should be zero
        assert dw == 0.0
    
    def test_timing_dependence(self):
        """Test timing dependence of STDP"""
        stdp = STDP()
        
        # Test different time differences
        dw1 = stdp.calculate_weight_change(10.0, 11.0)  # 1ms difference
        dw2 = stdp.calculate_weight_change(10.0, 20.0)  # 10ms difference
        
        # Closer timing should have larger effect
        assert dw1 > dw2

class TestNeuralPopulation:
    """Test neural population functionality"""
    
    def test_population_initialization(self):
        """Test population initialization"""
        population = NeuralPopulation("test", 10, "regular_spiking", 0.3)
        
        assert population.population_id == "test"
        assert population.num_neurons == 10
        assert population.neuron_type == "regular_spiking"
        assert population.connectivity == 0.3
        assert len(population.neurons) == 10
        assert len(population.synapses) > 0
    
    def test_population_step(self):
        """Test population stepping"""
        population = NeuralPopulation("test", 5, "regular_spiking", 0.5)
        
        # Step with external inputs
        external_inputs = [10.0] * 5
        spike_events = population.step(external_inputs, 1.0, 0.0)
        
        # Should return spike events for each neuron
        assert len(spike_events) == 5
        assert all(isinstance(event, bool) for event in spike_events)
    
    def test_population_statistics(self):
        """Test population statistics"""
        population = NeuralPopulation("test", 10, "regular_spiking", 0.3)
        
        # Run multiple steps to build up statistics
        for step in range(50):
            external_inputs = [5.0] * 10
            population.step(external_inputs, 1.0, step)
        
        # Check statistics
        firing_rate = population.get_population_firing_rate()
        assert firing_rate >= 0.0
        
        neuron_rates = population.get_neuron_firing_rates()
        assert len(neuron_rates) == 10
        assert all(rate >= 0.0 for rate in neuron_rates)
        
        weights = population.get_synaptic_weights()
        assert len(weights) > 0
        
        potentials = population.get_membrane_potentials()
        assert len(potentials) == 10
    
    def test_synaptic_connectivity(self):
        """Test synaptic connectivity patterns"""
        population = NeuralPopulation("test", 10, "regular_spiking", 0.5)
        
        # Check connectivity
        synapses = population.synapses
        assert len(synapses) > 0
        
        # Check synapse structure
        for (pre, post), synapse in synapses.items():
            assert pre != post  # No self-connections
            assert 0 <= pre < 10
            assert 0 <= post < 10
            assert isinstance(synapse, HebbianSynapse)

class TestNeuralAnalysis:
    """Test neural analysis functions"""
    
    def test_synchrony_calculation(self):
        """Test neural synchrony calculation"""
        # Test with no spikes
        synchrony = calculate_synchrony([], window_size=100.0)
        assert synchrony == 0.0
        
        # Test with single spike
        synchrony = calculate_synchrony([50.0], window_size=100.0)
        assert synchrony == 0.0
        
        # Test with multiple spikes
        spike_times = [10.0, 20.0, 30.0, 40.0, 50.0]
        synchrony = calculate_synchrony(spike_times, window_size=100.0)
        assert 0.0 <= synchrony <= 1.0
    
    def test_oscillation_power(self):
        """Test oscillation power calculation"""
        # Test with no spikes
        power = calculate_oscillation_power([], 10.0)
        assert power == 0.0
        
        # Test with spikes
        spike_times = [10.0, 20.0, 30.0, 40.0, 50.0]
        power = calculate_oscillation_power(spike_times, 10.0)
        assert power >= 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
