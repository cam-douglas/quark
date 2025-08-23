#!/usr/bin/env python3
"""
🧠 Izhikevich Neuron Model Implementation
Implements biologically plausible neuron models using the Izhikevich equations
for realistic neural dynamics and firing patterns.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class NeuronType(Enum):
    """Types of neurons based on Izhikevich classification"""
    REGULAR_SPIKING = "RS"      # Regular spiking (cortical excitatory)
    INTRINSICALLY_BURSTING = "IB"  # Intrinsically bursting
    CHATTERING = "CH"           # Chattering
    FAST_SPIKING = "FS"         # Fast spiking (cortical inhibitory)
    LOW_THRESHOLD_SPIKING = "LTS"  # Low-threshold spiking
    THALAMO_CORTICAL = "TC"     # Thalamo-cortical

@dataclass
class NeuronParameters:
    """Parameters for Izhikevich neuron model"""
    a: float  # Recovery time scale
    b: float  # Sensitivity of recovery variable
    c: float  # Post-spike reset value of v
    d: float  # Post-spike reset value of u
    v_rest: float = -65.0  # Resting membrane potential
    v_threshold: float = 30.0  # Spike threshold
    dt: float = 0.1  # Time step (ms)

@dataclass
class NeuronState:
    """Current state of a neuron"""
    v: float  # Membrane potential (mV)
    u: float  # Recovery variable
    last_spike_time: float = 0.0
    spike_count: int = 0
    is_refractory: bool = False

class IzhikevichNeuron:
    """Single Izhikevich neuron implementation"""
    
    # Predefined parameter sets for different neuron types
    NEURON_PARAMS = {
        NeuronType.REGULAR_SPIKING: NeuronParameters(a=0.02, b=0.2, c=-65.0, d=2.0),
        NeuronType.INTRINSICALLY_BURSTING: NeuronParameters(a=0.02, b=0.2, c=-55.0, d=4.0),
        NeuronType.CHATTERING: NeuronParameters(a=0.02, b=0.2, c=-50.0, d=2.0),
        NeuronType.FAST_SPIKING: NeuronParameters(a=0.1, b=0.2, c=-65.0, d=2.0),
        NeuronType.LOW_THRESHOLD_SPIKING: NeuronParameters(a=0.02, b=0.25, c=-65.0, d=2.0),
        NeuronType.THALAMO_CORTICAL: NeuronParameters(a=0.02, b=0.25, c=-65.0, d=0.05)
    }
    
    def __init__(self, neuron_type: NeuronType = NeuronType.REGULAR_SPIKING, 
                 neuron_id: str = None):
        self.neuron_type = neuron_type
        self.neuron_id = neuron_id or f"neuron_{int(time.time() * 1000)}"
        
        # Get parameters for this neuron type
        self.params = self.NEURON_PARAMS[neuron_type]
        
        # Initialize state
        self.state = NeuronState(
            v=self.params.v_rest,
            u=self.params.b * self.params.v_rest
        )
        
        # Spike history
        self.spike_times: List[float] = []
        self.membrane_potential_history: List[float] = []
        self.recovery_variable_history: List[float] = []
        
        # Input current
        self.input_current = 0.0
        
        print(f"🧠 Created {neuron_type.value} neuron: {self.neuron_id}")
    
    def step(self, dt: float = None) -> bool:
        """Step the neuron forward in time, returns True if spike occurred"""
        if dt is None:
            dt = self.params.dt
        
        # Check if neuron is in refractory period
        if self.state.is_refractory:
            self.state.is_refractory = False
            return False
        
        # Izhikevich equations
        # dv/dt = 0.04v² + 5v + 140 - u + I
        # du/dt = a(bv - u)
        
        # Update membrane potential
        dv = (0.04 * self.state.v**2 + 5*self.state.v + 140 - self.state.u + self.input_current) * dt
        self.state.v += dv
        
        # Update recovery variable
        du = self.params.a * (self.params.b * self.state.v - self.state.u) * dt
        self.state.u += du
        
        # Check for spike
        if self.state.v >= self.params.v_threshold:
            # Spike occurred
            self.state.v = self.params.c
            self.state.u += self.params.d
            self.state.spike_count += 1
            self.state.last_spike_time = time.time()
            self.state.is_refractory = True
            
            # Record spike
            self.spike_times.append(time.time())
            
            return True
        
        # Record history
        self.membrane_potential_history.append(self.state.v)
        self.recovery_variable_history.append(self.state.u)
        
        return False
    
    def add_input_current(self, current: float):
        """Add input current to the neuron"""
        self.input_current += current
    
    def reset_input_current(self):
        """Reset input current to zero"""
        self.input_current = 0.0
    
    def get_firing_rate(self, time_window: float = 1000.0) -> float:
        """Calculate firing rate in Hz over the specified time window"""
        current_time = time.time()
        recent_spikes = [t for t in self.spike_times if current_time - t <= time_window/1000.0]
        return len(recent_spikes) / (time_window / 1000.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current neuron status"""
        return {
            "neuron_id": self.neuron_id,
            "neuron_type": self.neuron_type.value,
            "membrane_potential": self.state.v,
            "recovery_variable": self.state.u,
            "input_current": self.input_current,
            "spike_count": self.state.spike_count,
            "firing_rate": self.get_firing_rate(),
            "is_refractory": self.state.is_refractory
        }

class IzhikevichNetwork:
    """Network of Izhikevich neurons with synaptic connections"""
    
    def __init__(self, num_neurons: int = 100):
        self.num_neurons = num_neurons
        self.neurons: List[IzhikevichNeuron] = []
        self.connections: Dict[Tuple[int, int], float] = {}  # (pre, post) -> weight
        self.synaptic_delays: Dict[Tuple[int, int], float] = {}  # (pre, post) -> delay
        
        # Network statistics
        self.network_stats = {
            "total_spikes": 0,
            "mean_firing_rate": 0.0,
            "synchronization_index": 0.0
        }
        
        print(f"🧠 Created Izhikevich network with {num_neurons} neurons")
    
    def add_neuron(self, neuron_type: NeuronType = NeuronType.REGULAR_SPIKING) -> int:
        """Add a neuron to the network, returns neuron index"""
        neuron = IzhikevichNeuron(neuron_type)
        self.neurons.append(neuron)
        return len(self.neurons) - 1
    
    def add_connection(self, pre_neuron: int, post_neuron: int, 
                      weight: float, delay: float = 1.0):
        """Add a synaptic connection between neurons"""
        if 0 <= pre_neuron < len(self.neurons) and 0 <= post_neuron < len(self.neurons):
            self.connections[(pre_neuron, post_neuron)] = weight
            self.synaptic_delays[(pre_neuron, post_neuron)] = delay
            print(f"🔗 Added connection {pre_neuron} -> {post_neuron} (weight: {weight:.2f})")
    
    def step(self, dt: float = 0.1) -> List[int]:
        """Step the entire network forward, returns list of neurons that spiked"""
        spiked_neurons = []
        
        # Step all neurons
        for i, neuron in enumerate(self.neurons):
            if neuron.step(dt):
                spiked_neurons.append(i)
                self.network_stats["total_spikes"] += 1
        
        # Apply synaptic connections
        for pre_neuron in spiked_neurons:
            for (pre, post), weight in self.connections.items():
                if pre == pre_neuron:
                    # Apply synaptic input with delay
                    self.neurons[post].add_input_current(weight)
        
        # Update network statistics
        self._update_network_stats()
        
        return spiked_neurons
    
    def _update_network_stats(self):
        """Update network statistics"""
        # Calculate mean firing rate
        firing_rates = [neuron.get_firing_rate() for neuron in self.neurons]
        self.network_stats["mean_firing_rate"] = np.mean(firing_rates)
        
        # Calculate synchronization index (simplified)
        if len(self.neurons) > 1:
            membrane_potentials = [neuron.state.v for neuron in self.neurons]
            self.network_stats["synchronization_index"] = np.std(membrane_potentials) / 100.0
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status"""
        return {
            "num_neurons": len(self.neurons),
            "num_connections": len(self.connections),
            "network_stats": self.network_stats.copy(),
            "neuron_types": [neuron.neuron_type.value for neuron in self.neurons]
        }
    
    def reset_network(self):
        """Reset the entire network"""
        for neuron in self.neurons:
            neuron.state = NeuronState(
                v=neuron.params.v_rest,
                u=neuron.params.b * neuron.params.v_rest
            )
            neuron.spike_times.clear()
            neuron.membrane_potential_history.clear()
            neuron.recovery_variable_history.clear()
            neuron.reset_input_current()
        
        self.network_stats = {
            "total_spikes": 0,
            "mean_firing_rate": 0.0,
            "synchronization_index": 0.0
        }

def main():
    """Demonstrate Izhikevich neuron model"""
    print("🧠 QUARK Izhikevich Neuron Model - Phase 2")
    print("=" * 50)
    
    # Create a small network
    network = IzhikevichNetwork(num_neurons=10)
    
    # Add different types of neurons
    for i in range(5):
        network.add_neuron(NeuronType.REGULAR_SPIKING)  # Excitatory
    
    for i in range(3):
        network.add_neuron(NeuronType.FAST_SPIKING)  # Inhibitory
    
    for i in range(2):
        network.add_neuron(NeuronType.LOW_THRESHOLD_SPIKING)  # Specialized
    
    # Add some connections
    for i in range(8):
        for j in range(i+1, min(i+3, 10)):
            weight = 0.5 if i < 5 else -0.3  # Excitatory vs inhibitory
            network.add_connection(i, j, weight)
    
    # Run simulation
    print("\n🔄 Running network simulation...")
    simulation_steps = 100
    
    for step in range(simulation_steps):
        spiked = network.step()
        if spiked:
            print(f"   Step {step}: Neurons {spiked} spiked")
    
    # Show results
    status = network.get_network_status()
    print(f"\n📊 Network Results:")
    print(f"   Total spikes: {status['network_stats']['total_spikes']}")
    print(f"   Mean firing rate: {status['network_stats']['mean_firing_rate']:.2f} Hz")
    print(f"   Synchronization: {status['network_stats']['synchronization_index']:.3f}")
    
    # Show individual neuron status
    print(f"\n🧠 Individual Neuron Status:")
    for i, neuron in enumerate(network.neurons):
        neuron_status = neuron.get_status()
        print(f"   Neuron {i} ({neuron_status['neuron_type']}): "
              f"V={neuron_status['membrane_potential']:.1f}mV, "
              f"Rate={neuron_status['firing_rate']:.1f}Hz")
    
    print("\n✅ Izhikevich neuron model demonstration completed!")
    return network

if __name__ == "__main__":
    try:
        network = main()
    except Exception as e:
        print(f"❌ Izhikevich model failed: {e}")
        import traceback
        traceback.print_exc()
