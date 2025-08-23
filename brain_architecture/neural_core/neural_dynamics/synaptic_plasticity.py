#!/usr/bin/env python3
"""
🔗 Synaptic Plasticity Mechanisms
Implements various forms of synaptic plasticity including STDP, 
Hebbian learning, and homeostatic plasticity for adaptive neural networks.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PlasticityType(Enum):
    """Types of synaptic plasticity"""
    STDP = "STDP"  # Spike-Timing Dependent Plasticity
    HEBBIAN = "HEBBIAN"  # Hebbian learning
    HOMEOSTATIC = "HOMEOSTATIC"  # Homeostatic plasticity
    BCM = "BCM"  # Bienenstock-Cooper-Munro rule

@dataclass
class STDPParameters:
    """Parameters for STDP learning rule"""
    tau_plus: float = 20.0  # Time constant for LTP (ms)
    tau_minus: float = 20.0  # Time constant for LTD (ms)
    A_plus: float = 0.1  # Amplitude for LTP
    A_minus: float = -0.1  # Amplitude for LTD
    max_weight: float = 2.0  # Maximum synaptic weight
    min_weight: float = 0.0  # Minimum synaptic weight

@dataclass
class SynapseState:
    """State of a synaptic connection"""
    weight: float = 1.0
    last_pre_spike: float = -1000.0  # Time of last presynaptic spike
    last_post_spike: float = -1000.0  # Time of last postsynaptic spike
    eligibility_trace: float = 0.0  # Eligibility trace for learning
    activity_history: List[float] = None  # Recent activity levels

class STDPPlasticity:
    """Spike-Timing Dependent Plasticity implementation"""
    
    def __init__(self, params: STDPParameters = None):
        self.params = params or STDPParameters()
        self.synapses: Dict[Tuple[int, int], SynapseState] = {}
        
        print("🔗 STDP Plasticity system initialized")
    
    def add_synapse(self, pre_neuron: int, post_neuron: int, initial_weight: float = 1.0):
        """Add a new synaptic connection"""
        synapse_key = (pre_neuron, post_neuron)
        self.synapses[synapse_key] = SynapseState(weight=initial_weight)
        print(f"🔗 Added synapse {pre_neuron} -> {post_neuron} (weight: {initial_weight:.2f})")
    
    def pre_spike(self, pre_neuron: int, post_neuron: int, spike_time: float):
        """Handle presynaptic spike"""
        synapse_key = (pre_neuron, post_neuron)
        if synapse_key in self.synapses:
            synapse = self.synapses[synapse_key]
            
            # Update last presynaptic spike time
            synapse.last_pre_spike = spike_time
            
            # Calculate time difference with last postsynaptic spike
            dt = synapse.last_pre_spike - synapse.last_post_spike
            
            if dt > 0:  # Pre-spike before post-spike (LTP)
                # Calculate weight change
                dw = self.params.A_plus * np.exp(-dt / self.params.tau_plus)
                synapse.weight = np.clip(synapse.weight + dw, 
                                       self.params.min_weight, 
                                       self.params.max_weight)
                
                print(f"   LTP: {pre_neuron} -> {post_neuron} (+{dw:.3f})")
    
    def post_spike(self, pre_neuron: int, post_neuron: int, spike_time: float):
        """Handle postsynaptic spike"""
        synapse_key = (pre_neuron, post_neuron)
        if synapse_key in self.synapses:
            synapse = self.synapses[synapse_key]
            
            # Update last postsynaptic spike time
            synapse.last_post_spike = spike_time
            
            # Calculate time difference with last presynaptic spike
            dt = synapse.last_post_spike - synapse.last_pre_spike
            
            if dt > 0:  # Post-spike before pre-spike (LTD)
                # Calculate weight change
                dw = self.params.A_minus * np.exp(-dt / self.params.tau_minus)
                synapse.weight = np.clip(synapse.weight + dw, 
                                       self.params.min_weight, 
                                       self.params.max_weight)
                
                print(f"   LTD: {pre_neuron} -> {post_neuron} ({dw:.3f})")
    
    def get_synapse_weight(self, pre_neuron: int, post_neuron: int) -> float:
        """Get current synaptic weight"""
        synapse_key = (pre_neuron, post_neuron)
        return self.synapses.get(synapse_key, SynapseState()).weight
    
    def get_all_weights(self) -> Dict[Tuple[int, int], float]:
        """Get all synaptic weights"""
        return {key: synapse.weight for key, synapse in self.synapses.items()}

class HebbianPlasticity:
    """Hebbian learning rule implementation"""
    
    def __init__(self, learning_rate: float = 0.01, max_weight: float = 2.0):
        self.learning_rate = learning_rate
        self.max_weight = max_weight
        self.synapses: Dict[Tuple[int, int], SynapseState] = {}
        
        print("🔗 Hebbian Plasticity system initialized")
    
    def add_synapse(self, pre_neuron: int, post_neuron: int, initial_weight: float = 1.0):
        """Add a new synaptic connection"""
        synapse_key = (pre_neuron, post_neuron)
        self.synapses[synapse_key] = SynapseState(weight=initial_weight)
    
    def update_weights(self, pre_activity: Dict[int, float], post_activity: Dict[int, float]):
        """Update synaptic weights based on pre and post synaptic activity"""
        for (pre, post), synapse in self.synapses.items():
            if pre in pre_activity and post in post_activity:
                # Hebbian rule: dw = η * pre_activity * post_activity
                dw = self.learning_rate * pre_activity[pre] * post_activity[post]
                synapse.weight = np.clip(synapse.weight + dw, 0.0, self.max_weight)
                
                if abs(dw) > 0.001:  # Only print significant changes
                    print(f"   Hebbian: {pre} -> {post} ({dw:+.3f})")

class HomeostaticPlasticity:
    """Homeostatic plasticity for maintaining network stability"""
    
    def __init__(self, target_rate: float = 10.0, learning_rate: float = 0.001):
        self.target_rate = target_rate  # Target firing rate (Hz)
        self.learning_rate = learning_rate
        self.neuron_rates: Dict[int, float] = {}
        self.synapses: Dict[Tuple[int, int], SynapseState] = {}
        
        print("🔗 Homeostatic Plasticity system initialized")
    
    def add_synapse(self, pre_neuron: int, post_neuron: int, initial_weight: float = 1.0):
        """Add a new synaptic connection"""
        synapse_key = (pre_neuron, post_neuron)
        self.synapses[synapse_key] = SynapseState(weight=initial_weight)
    
    def update_rates(self, neuron_rates: Dict[int, float]):
        """Update current firing rates"""
        self.neuron_rates.update(neuron_rates)
    
    def apply_homeostasis(self):
        """Apply homeostatic plasticity to maintain target firing rates"""
        for (pre, post), synapse in self.synapses.items():
            if post in self.neuron_rates:
                current_rate = self.neuron_rates[post]
                rate_error = self.target_rate - current_rate
                
                # Adjust weight based on rate error
                dw = self.learning_rate * rate_error
                synapse.weight = np.clip(synapse.weight + dw, 0.0, 2.0)
                
                if abs(dw) > 0.001:  # Only print significant changes
                    print(f"   Homeostasis: {pre} -> {post} ({dw:+.3f})")

class PlasticityManager:
    """Manages multiple plasticity mechanisms"""
    
    def __init__(self):
        self.stdp = STDPPlasticity()
        self.hebbian = HebbianPlasticity()
        self.homeostasis = HomeostaticPlasticity()
        
        # Track which plasticity types are active
        self.active_plasticity = {
            PlasticityType.STDP: True,
            PlasticityType.HEBBIAN: False,
            PlasticityType.HOMEOSTATIC: True
        }
        
        print("🔗 Plasticity Manager initialized")
    
    def add_synapse(self, pre_neuron: int, post_neuron: int, initial_weight: float = 1.0):
        """Add synapse to all active plasticity mechanisms"""
        if self.active_plasticity[PlasticityType.STDP]:
            self.stdp.add_synapse(pre_neuron, post_neuron, initial_weight)
        
        if self.active_plasticity[PlasticityType.HEBBIAN]:
            self.hebbian.add_synapse(pre_neuron, post_neuron, initial_weight)
        
        if self.active_plasticity[PlasticityType.HOMEOSTATIC]:
            self.homeostasis.add_synapse(pre_neuron, post_neuron, initial_weight)
    
    def handle_pre_spike(self, pre_neuron: int, post_neuron: int, spike_time: float):
        """Handle presynaptic spike for STDP"""
        if self.active_plasticity[PlasticityType.STDP]:
            self.stdp.pre_spike(pre_neuron, post_neuron, spike_time)
    
    def handle_post_spike(self, pre_neuron: int, post_neuron: int, spike_time: float):
        """Handle postsynaptic spike for STDP"""
        if self.active_plasticity[PlasticityType.STDP]:
            self.stdp.post_spike(pre_neuron, post_neuron, spike_time)
    
    def update_hebbian(self, pre_activity: Dict[int, float], post_activity: Dict[int, float]):
        """Update Hebbian plasticity"""
        if self.active_plasticity[PlasticityType.HEBBIAN]:
            self.hebbian.update_weights(pre_activity, post_activity)
    
    def update_homeostasis(self, neuron_rates: Dict[int, float]):
        """Update homeostatic plasticity"""
        if self.active_plasticity[PlasticityType.HOMEOSTATIC]:
            self.homeostasis.update_rates(neuron_rates)
            self.homeostasis.apply_homeostasis()
    
    def get_synapse_weight(self, pre_neuron: int, post_neuron: int) -> float:
        """Get current synaptic weight (from STDP as primary mechanism)"""
        return self.stdp.get_synapse_weight(pre_neuron, post_neuron)
    
    def get_plasticity_stats(self) -> Dict[str, Any]:
        """Get statistics about plasticity mechanisms"""
        stdp_weights = self.stdp.get_all_weights()
        
        return {
            "active_mechanisms": [pt.value for pt, active in self.active_plasticity.items() if active],
            "num_synapses": len(stdp_weights),
            "mean_weight": np.mean(list(stdp_weights.values())) if stdp_weights else 0.0,
            "weight_std": np.std(list(stdp_weights.values())) if stdp_weights else 0.0,
            "stdp_weights": stdp_weights
        }

def main():
    """Demonstrate synaptic plasticity mechanisms"""
    print("🔗 QUARK Synaptic Plasticity - Phase 2")
    print("=" * 50)
    
    # Create plasticity manager
    plasticity = PlasticityManager()
    
    # Add some synapses
    for i in range(5):
        for j in range(i+1, 5):
            plasticity.add_synapse(i, j, initial_weight=1.0)
    
    # Simulate some spike events
    print("\n🔄 Simulating spike events...")
    current_time = time.time()
    
    # Pre-spike events
    plasticity.handle_pre_spike(0, 1, current_time)
    time.sleep(0.01)  # Small delay
    
    # Post-spike events (LTD)
    plasticity.handle_post_spike(0, 1, current_time + 0.01)
    
    # Another pre-spike (LTP)
    time.sleep(0.01)
    plasticity.handle_pre_spike(1, 2, current_time + 0.02)
    plasticity.handle_post_spike(1, 2, current_time + 0.03)
    
    # Update Hebbian plasticity
    pre_activity = {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.2, 4: 0.1}
    post_activity = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}
    plasticity.update_hebbian(pre_activity, post_activity)
    
    # Update homeostatic plasticity
    neuron_rates = {0: 8.0, 1: 12.0, 2: 15.0, 3: 6.0, 4: 18.0}
    plasticity.update_homeostasis(neuron_rates)
    
    # Show results
    stats = plasticity.get_plasticity_stats()
    print(f"\n📊 Plasticity Statistics:")
    print(f"   Active mechanisms: {', '.join(stats['active_mechanisms'])}")
    print(f"   Number of synapses: {stats['num_synapses']}")
    print(f"   Mean weight: {stats['mean_weight']:.3f}")
    print(f"   Weight std: {stats['weight_std']:.3f}")
    
    print(f"\n🔗 Synaptic Weights:")
    for (pre, post), weight in stats['stdp_weights'].items():
        print(f"   {pre} -> {post}: {weight:.3f}")
    
    print("\n✅ Synaptic plasticity demonstration completed!")
    return plasticity

if __name__ == "__main__":
    try:
        plasticity = main()
    except Exception as e:
        print(f"❌ Synaptic plasticity failed: {e}")
        import traceback
        traceback.print_exc()
