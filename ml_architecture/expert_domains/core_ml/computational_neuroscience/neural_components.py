#!/usr/bin/env python3
"""
🧠 Neural Components for Pillar 1: Basic Neural Dynamics
Implements spiking neurons, Hebbian plasticity, and STDP mechanisms

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Core neural dynamics for brain simulation
**Validation Level:** Functional behavior verification
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, field

@dataclass
class SpikeEvent:
    """Represents a spike event with timing information"""
    neuron_id: int
    timestamp: float
    strength: float = 1.0

class SpikingNeuron:
    """Izhikevich spiking neuron model for realistic neural dynamics"""
    
    def __init__(self, neuron_id: int, neuron_type: str = "regular_spiking", 
                 v0: float = -65.0, u0: float = 0.0):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.v = v0  # Membrane potential (mV)
        self.u = u0  # Recovery variable
        
        # Izhikevich parameters based on neuron type
        if neuron_type == "regular_spiking":
            self.a = 0.02
            self.b = 0.2
            self.c = -65.0
            self.d = 8.0  # Increased for more spiking
        elif neuron_type == "fast_spiking":
            self.a = 0.1
            self.b = 0.2
            self.c = -65.0
            self.d = 2.0
        elif neuron_type == "bursting":
            self.a = 0.02
            self.b = 0.2
            self.c = -50.0
            self.d = 2.0
        else:  # Default to regular spiking
            self.a = 0.02
            self.b = 0.2
            self.c = -65.0
            self.d = 8.0  # Increased for more spiking
        
        self.spike_history = []
        self.firing_rate = 0.0
        self.last_spike_time = 0.0
        
    def step(self, I_input: float, dt: float = 1.0, current_time: float = 0.0) -> bool:
        """
        Step the neuron forward in time
        
        Args:
            I_input: Input current (nA)
            dt: Time step (ms)
            current_time: Current simulation time (ms)
            
        Returns:
            bool: True if neuron spiked, False otherwise
        """
        # Izhikevich model equations
        dv = (0.04 * self.v**2 + 5*self.v + 140 - self.u + I_input) * dt
        du = self.a * (self.b * self.v - self.u) * dt
        
        self.v += dv
        self.u += du
        
        # Spike detection and reset
        if self.v >= 30:
            self.v = self.c
            self.u += self.d
            
            # Record spike
            spike_event = SpikeEvent(
                neuron_id=self.neuron_id,
                timestamp=current_time,
                strength=1.0
            )
            self.spike_history.append(spike_event)
            self.last_spike_time = current_time
            
            # Update firing rate (simple moving average)
            if len(self.spike_history) > 100:
                self.spike_history = self.spike_history[-100:]
            
            # Calculate firing rate based on recent spikes
            if len(self.spike_history) > 0:
                # Use last 1000ms or all available spikes if less than 1000ms
                time_window = min(1000.0, current_time)
                if time_window > 0:
                    recent_spikes = [s for s in self.spike_history if current_time - s.timestamp < time_window]
                    self.firing_rate = len(recent_spikes) / (time_window / 1000.0)  # Hz
                else:
                    self.firing_rate = 0.0
            else:
                self.firing_rate = 0.0
            
            return True
        
        return False
    
    def get_firing_rate(self) -> float:
        """Get current firing rate in Hz"""
        return self.firing_rate
    
    def get_membrane_potential(self) -> float:
        """Get current membrane potential"""
        return self.v
    
    def get_recovery_variable(self) -> float:
        """Get current recovery variable"""
        return self.u

class HebbianSynapse:
    """Hebbian synapse with weight plasticity"""
    
    def __init__(self, pre_neuron_id: int, post_neuron_id: int, 
                 initial_weight: float = 0.5, learning_rate: float = 0.01):
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = initial_weight
        self.learning_rate = learning_rate
        self.weight_history = [initial_weight]
        
        # Bounds for synaptic weights
        self.min_weight = 0.0
        self.max_weight = 2.0
        
    def update_hebbian(self, pre_spike: bool, post_spike: bool, 
                      pre_firing_rate: float, post_firing_rate: float) -> float:
        """
        Update synaptic weight using Hebbian learning rule
        
        Args:
            pre_spike: Whether pre-synaptic neuron spiked
            post_spike: Whether post-synaptic neuron spiked
            pre_firing_rate: Pre-synaptic firing rate
            post_firing_rate: Post-synaptic firing rate
            
        Returns:
            float: Weight change
        """
        # Classic Hebbian rule: strengthen if both neurons are active
        if pre_spike and post_spike:
            dw = self.learning_rate * pre_firing_rate * post_firing_rate
        elif pre_spike and not post_spike:
            # Anti-Hebbian: weaken if only pre-synaptic fires
            dw = -self.learning_rate * pre_firing_rate * 0.1
        elif not pre_spike and post_spike:
            # Anti-Hebbian: weaken if only post-synaptic fires
            dw = -self.learning_rate * post_firing_rate * 0.1
        else:
            dw = 0.0
        
        # Apply weight change
        old_weight = self.weight
        self.weight += dw
        
        # Enforce bounds
        self.weight = max(self.min_weight, min(self.max_weight, self.weight))
        
        # Record weight history
        self.weight_history.append(self.weight)
        if len(self.weight_history) > 1000:
            self.weight_history = self.weight_history[-1000:]
        
        return self.weight - old_weight
    
    def get_weight(self) -> float:
        """Get current synaptic weight"""
        return self.weight
    
    def get_weight_history(self) -> List[float]:
        """Get weight history for analysis"""
        return self.weight_history.copy()

class STDP:
    """Spike-Timing Dependent Plasticity implementation"""
    
    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0,
                 A_plus: float = 0.01, A_minus: float = 0.01):
        self.tau_plus = tau_plus   # Time constant for LTP (ms)
        self.tau_minus = tau_minus # Time constant for LTD (ms)
        self.A_plus = A_plus       # Amplitude for LTP
        self.A_minus = A_minus     # Amplitude for LTD
        
    def calculate_weight_change(self, pre_spike_time: float, post_spike_time: float) -> float:
        """
        Calculate weight change based on spike timing
        
        Args:
            pre_spike_time: Time of pre-synaptic spike
            post_spike_time: Time of post-synaptic spike
            
        Returns:
            float: Weight change
        """
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Post-synaptic spike after pre-synaptic (LTP)
            dw = self.A_plus * np.exp(-dt / self.tau_plus)
        elif dt < 0:  # Pre-synaptic spike after post-synaptic (LTD)
            dw = -self.A_minus * np.exp(dt / self.tau_minus)
        else:  # Simultaneous spikes
            dw = 0.0
            
        return dw

class NeuralPopulation:
    """A population of spiking neurons with synaptic connections"""
    
    def __init__(self, population_id: str, num_neurons: int, 
                 neuron_type: str = "regular_spiking", connectivity: float = 0.1):
        self.population_id = population_id
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type
        self.connectivity = connectivity
        
        # Create neurons
        self.neurons = []
        for i in range(num_neurons):
            neuron = SpikingNeuron(i, neuron_type)
            self.neurons.append(neuron)
        
        # Create synaptic connections
        self.synapses = {}
        self._create_connections()
        
        # STDP mechanism
        self.stdp = STDP()
        
        # Population statistics
        self.population_firing_rate = 0.0
        self.spike_times = []
        
    def _create_connections(self):
        """Create random synaptic connections within the population"""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and random.random() < self.connectivity:
                    synapse = HebbianSynapse(i, j)
                    self.synapses[(i, j)] = synapse
    
    def step(self, external_inputs: List[float], dt: float = 1.0, 
             current_time: float = 0.0) -> List[bool]:
        """
        Step the entire population forward in time
        
        Args:
            external_inputs: External input currents for each neuron
            dt: Time step (ms)
            current_time: Current simulation time (ms)
            
        Returns:
            List[bool]: Spike events for each neuron
        """
        # Ensure external_inputs matches neuron count
        if len(external_inputs) != self.num_neurons:
            external_inputs = external_inputs + [0.0] * (self.num_neurons - len(external_inputs))
        
        # Step all neurons
        spike_events = []
        for i, neuron in enumerate(self.neurons):
            # Calculate synaptic input from other neurons
            synaptic_input = 0.0
            for j, other_neuron in enumerate(self.neurons):
                if i != j and (j, i) in self.synapses:
                    synapse = self.synapses[(j, i)]
                    # Simple synaptic input (could be more sophisticated)
                    if other_neuron.get_firing_rate() > 0:
                        synaptic_input += synapse.get_weight() * other_neuron.get_firing_rate()
            
            # Total input
            total_input = external_inputs[i] + synaptic_input
            
            # Step neuron
            spiked = neuron.step(total_input, dt, current_time)
            spike_events.append(spiked)
            
            if spiked:
                self.spike_times.append(current_time)
        
        # Update synaptic weights using STDP
        self._update_synapses(current_time)
        
        # Update population statistics
        self._update_population_stats()
        
        return spike_events
    
    def _update_synapses(self, current_time: float):
        """Update synaptic weights using STDP"""
        for (pre_id, post_id), synapse in self.synapses.items():
            pre_neuron = self.neurons[pre_id]
            post_neuron = self.neurons[post_id]
            
            # Get recent spike times
            pre_spikes = [s.timestamp for s in pre_neuron.spike_history 
                         if current_time - s.timestamp < 100]
            post_spikes = [s.timestamp for s in post_neuron.spike_history 
                          if current_time - s.timestamp < 100]
            
            # Apply STDP for recent spikes
            for pre_time in pre_spikes:
                for post_time in post_spikes:
                    dw = self.stdp.calculate_weight_change(pre_time, post_time)
                    synapse.weight += dw
                    synapse.weight = max(0.0, min(2.0, synapse.weight))
    
    def _update_population_stats(self):
        """Update population-level statistics"""
        # Calculate average firing rate
        total_firing_rate = sum(neuron.get_firing_rate() for neuron in self.neurons)
        self.population_firing_rate = total_firing_rate / self.num_neurons
        
        # Keep only recent spike times
        if len(self.spike_times) > 1000:
            self.spike_times = self.spike_times[-1000:]
    
    def get_population_firing_rate(self) -> float:
        """Get average firing rate of the population"""
        return self.population_firing_rate
    
    def get_neuron_firing_rates(self) -> List[float]:
        """Get firing rates of all neurons"""
        return [neuron.get_firing_rate() for neuron in self.neurons]
    
    def get_synaptic_weights(self) -> Dict[Tuple[int, int], float]:
        """Get all synaptic weights"""
        return {(pre, post): synapse.get_weight() 
                for (pre, post), synapse in self.synapses.items()}
    
    def get_membrane_potentials(self) -> List[float]:
        """Get membrane potentials of all neurons"""
        return [neuron.get_membrane_potential() for neuron in self.neurons]

# Utility functions for neural analysis
def calculate_synchrony(spike_times: List[float], window_size: float = 100.0) -> float:
    """
    Calculate neural synchrony using spike timing
    
    Args:
        spike_times: List of spike times
        window_size: Time window for synchrony calculation (ms)
        
    Returns:
        float: Synchrony measure (0-1)
    """
    if len(spike_times) < 2:
        return 0.0
    
    # Count spikes in time windows
    windows = {}
    for spike_time in spike_times:
        window_id = int(spike_time / window_size)
        windows[window_id] = windows.get(window_id, 0) + 1
    
    # Calculate coefficient of variation
    spike_counts = list(windows.values())
    if len(spike_counts) < 2:
        return 0.0
    
    mean_count = np.mean(spike_counts)
    std_count = np.std(spike_counts)
    
    if mean_count == 0:
        return 0.0
    
    cv = std_count / mean_count
    synchrony = 1.0 / (1.0 + cv)  # Higher CV = lower synchrony
    
    return synchrony

def calculate_oscillation_power(spike_times: List[float], frequency: float, 
                               sampling_rate: float = 1000.0) -> float:
    """
    Calculate power at a specific frequency using FFT
    
    Args:
        spike_times: List of spike times
        frequency: Frequency of interest (Hz)
        sampling_rate: Sampling rate (Hz)
        
    Returns:
        float: Power at the specified frequency
    """
    if len(spike_times) < 10:
        return 0.0
    
    # Create spike train
    max_time = max(spike_times)
    num_samples = int(max_time * sampling_rate / 1000.0)
    spike_train = np.zeros(num_samples)
    
    for spike_time in spike_times:
        sample_idx = int(spike_time * sampling_rate / 1000.0)
        if sample_idx < num_samples:
            spike_train[sample_idx] = 1.0
    
    # Calculate FFT
    fft = np.fft.fft(spike_train)
    freqs = np.fft.fftfreq(len(spike_train), 1.0/sampling_rate)
    
    # Find power at target frequency
    target_idx = np.argmin(np.abs(freqs - frequency))
    power = np.abs(fft[target_idx])**2
    
    return power
