#!/usr/bin/env python3
"""
🧠 Integrated Neural Dynamics System - Phase 2
Combines Izhikevich neurons, synaptic plasticity, and attention mechanisms
for comprehensive biologically plausible neural processing.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import Phase 2 components
from izhikevich_neurons import IzhikevichNetwork, NeuronType
from synaptic_plasticity import PlasticityManager
from attention_mechanisms import AttentionManager

@dataclass
class NeuralDynamicsConfig:
    """Configuration for integrated neural dynamics"""
    num_neurons: int = 50
    spatial_resolution: Tuple[int, int] = (50, 50)
    simulation_steps: int = 100
    time_step: float = 0.1
    enable_plasticity: bool = True
    enable_attention: bool = True

class IntegratedNeuralDynamics:
    """Integrated neural dynamics system combining all Phase 2 components"""
    
    def __init__(self, config: NeuralDynamicsConfig = None):
        self.config = config or NeuralDynamicsConfig()
        
        # Initialize components
        self.neural_network = IzhikevichNetwork(self.config.num_neurons)
        self.plasticity_manager = PlasticityManager()
        self.attention_manager = AttentionManager(self.config.spatial_resolution)
        
        # Integration state
        self.simulation_time = 0.0
        self.neuron_locations: Dict[int, Tuple[int, int]] = {}
        self.neuron_features: Dict[int, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_spikes": 0,
            "mean_firing_rate": 0.0,
            "plasticity_events": 0,
            "attention_modulations": 0,
            "synchronization_index": 0.0
        }
        
        print("🧠 Integrated Neural Dynamics System initialized")
        self._setup_network()
    
    def _setup_network(self):
        """Setup the neural network with different neuron types"""
        # Add different types of neurons
        num_excitatory = int(self.config.num_neurons * 0.8)
        num_inhibitory = int(self.config.num_neurons * 0.15)
        num_specialized = self.config.num_neurons - num_excitatory - num_inhibitory
        
        # Add excitatory neurons (Regular Spiking)
        for i in range(num_excitatory):
            neuron_idx = self.neural_network.add_neuron(NeuronType.REGULAR_SPIKING)
            self._assign_neuron_location(neuron_idx)
            self._assign_neuron_features(neuron_idx, "excitatory")
        
        # Add inhibitory neurons (Fast Spiking)
        for i in range(num_inhibitory):
            neuron_idx = self.neural_network.add_neuron(NeuronType.FAST_SPIKING)
            self._assign_neuron_location(neuron_idx)
            self._assign_neuron_features(neuron_idx, "inhibitory")
        
        # Add specialized neurons (Low Threshold Spiking)
        for i in range(num_specialized):
            neuron_idx = self.neural_network.add_neuron(NeuronType.LOW_THRESHOLD_SPIKING)
            self._assign_neuron_location(neuron_idx)
            self._assign_neuron_features(neuron_idx, "specialized")
        
        # Add synaptic connections
        self._setup_connections()
        
        print(f"   Created {num_excitatory} excitatory, {num_inhibitory} inhibitory, {num_specialized} specialized neurons")
    
    def _assign_neuron_location(self, neuron_idx: int):
        """Assign spatial location to neuron"""
        x = np.random.randint(0, self.config.spatial_resolution[1])
        y = np.random.randint(0, self.config.spatial_resolution[0])
        self.neuron_locations[neuron_idx] = (x, y)
    
    def _assign_neuron_features(self, neuron_idx: int, neuron_type: str):
        """Assign features to neuron"""
        features = {
            "type": neuron_type,
            "color": np.random.choice(["red", "blue", "green", "yellow"]),
            "orientation": np.random.randint(0, 360),
            "motion": np.random.choice(["fast", "medium", "slow"]),
            "sensitivity": np.random.uniform(0.5, 1.5)
        }
        self.neuron_features[neuron_idx] = features
    
    def _setup_connections(self):
        """Setup synaptic connections between neurons"""
        num_connections = 0
        
        for i in range(len(self.neural_network.neurons)):
            # Connect to nearby neurons
            for j in range(i + 1, len(self.neural_network.neurons)):
                if np.random.random() < 0.3:  # 30% connection probability
                    # Determine connection weight based on neuron types
                    pre_type = self.neuron_features[i]["type"]
                    post_type = self.neuron_features[j]["type"]
                    
                    if pre_type == "excitatory" and post_type == "excitatory":
                        weight = np.random.uniform(0.3, 0.8)
                    elif pre_type == "inhibitory":
                        weight = np.random.uniform(-0.6, -0.2)
                    else:
                        weight = np.random.uniform(-0.3, 0.5)
                    
                    # Add connection to network
                    self.neural_network.add_connection(i, j, weight)
                    
                    # Add connection to plasticity manager
                    self.plasticity_manager.add_synapse(i, j, weight)
                    
                    num_connections += 1
        
        print(f"   Created {num_connections} synaptic connections")
    
    def step(self) -> Dict[str, Any]:
        """Step the integrated system forward"""
        # Step neural network
        spiked_neurons = self.neural_network.step(self.config.time_step)
        
        # Update plasticity based on spikes
        if self.config.enable_plasticity and spiked_neurons:
            current_time = time.time()
            
            for pre_neuron in spiked_neurons:
                for (pre, post), weight in self.neural_network.connections.items():
                    if pre == pre_neuron:
                        # Handle STDP
                        self.plasticity_manager.handle_pre_spike(pre, post, current_time)
                        
                        # Check if post neuron also spiked (for post-spike STDP)
                        if post in spiked_neurons:
                            self.plasticity_manager.handle_post_spike(pre, post, current_time)
            
            # Update homeostatic plasticity
            neuron_rates = {i: neuron.get_firing_rate() for i, neuron in enumerate(self.neural_network.neurons)}
            self.plasticity_manager.update_homeostasis(neuron_rates)
            
            self.performance_metrics["plasticity_events"] += len(spiked_neurons)
        
        # Update attention mechanisms
        if self.config.enable_attention:
            self._update_attention()
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Update simulation time
        self.simulation_time += self.config.time_step
        
        return {
            "spiked_neurons": spiked_neurons,
            "simulation_time": self.simulation_time,
            "performance_metrics": self.performance_metrics.copy()
        }
    
    def _update_attention(self):
        """Update attention mechanisms based on current neural activity"""
        # Add attention focus based on active neurons
        for neuron_idx in range(len(self.neural_network.neurons)):
            neuron = self.neural_network.neurons[neuron_idx]
            firing_rate = neuron.get_firing_rate()
            
            if firing_rate > 5.0:  # High activity neurons
                location = self.neuron_locations[neuron_idx]
                features = self.neuron_features[neuron_idx]
                
                # Add spatial attention focus
                self.attention_manager.add_spatial_focus(
                    location[0], location[1], 
                    radius=5.0, strength=firing_rate / 10.0
                )
                
                # Add feature attention focus
                self.attention_manager.add_feature_focus(
                    "color", features["color"], strength=0.5
                )
                
                self.performance_metrics["attention_modulations"] += 1
        
        # Update attention systems
        self.attention_manager.update_attention_systems()
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        # Update firing rate
        firing_rates = [neuron.get_firing_rate() for neuron in self.neural_network.neurons]
        self.performance_metrics["mean_firing_rate"] = np.mean(firing_rates)
        
        # Update synchronization index
        membrane_potentials = [neuron.state.v for neuron in self.neural_network.neurons]
        self.performance_metrics["synchronization_index"] = np.std(membrane_potentials) / 100.0
    
    def apply_stimulus(self, stimulus_location: Tuple[int, int], 
                      stimulus_features: Dict[str, Any], strength: float = 1.0):
        """Apply external stimulus to the network"""
        # Find neurons near stimulus location
        target_neurons = []
        
        for neuron_idx, location in self.neuron_locations.items():
            distance = np.sqrt((location[0] - stimulus_location[0])**2 + 
                             (location[1] - stimulus_location[1])**2)
            
            if distance < 10:  # Within 10 units
                # Check feature similarity
                features = self.neuron_features[neuron_idx]
                feature_match = 0
                
                for key, value in stimulus_features.items():
                    if key in features and features[key] == value:
                        feature_match += 1
                
                # Calculate stimulus strength based on distance and feature match
                stimulus_strength = strength * (1.0 - distance/10.0) * (feature_match / len(stimulus_features))
                
                if stimulus_strength > 0.1:
                    target_neurons.append((neuron_idx, stimulus_strength))
        
        # Apply stimulus to target neurons
        for neuron_idx, stimulus_strength in target_neurons:
            self.neural_network.neurons[neuron_idx].add_input_current(stimulus_strength)
        
        print(f"🎯 Applied stimulus to {len(target_neurons)} neurons")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        network_status = self.neural_network.get_network_status()
        plasticity_stats = self.plasticity_manager.get_plasticity_stats()
        attention_stats = self.attention_manager.get_attention_stats()
        
        return {
            "neural_network": network_status,
            "plasticity": plasticity_stats,
            "attention": attention_stats,
            "performance_metrics": self.performance_metrics.copy(),
            "simulation_time": self.simulation_time,
            "config": {
                "num_neurons": self.config.num_neurons,
                "enable_plasticity": self.config.enable_plasticity,
                "enable_attention": self.config.enable_attention
            }
        }
    
    def run_simulation(self, steps: int = None) -> Dict[str, Any]:
        """Run a complete simulation"""
        steps = steps or self.config.simulation_steps
        
        print(f"🔄 Running integrated neural dynamics simulation for {steps} steps...")
        
        # Apply initial stimulus
        self.apply_stimulus((25, 25), {"color": "red", "motion": "fast"}, strength=2.0)
        
        # Run simulation
        for step in range(steps):
            result = self.step()
            
            if step % 20 == 0:  # Print every 20 steps
                print(f"   Step {step}: {len(result['spiked_neurons'])} neurons spiked")
        
        # Final status
        final_status = self.get_system_status()
        
        print(f"\n📊 Simulation Results:")
        print(f"   Total spikes: {final_status['performance_metrics']['total_spikes']}")
        print(f"   Mean firing rate: {final_status['performance_metrics']['mean_firing_rate']:.2f} Hz")
        print(f"   Plasticity events: {final_status['performance_metrics']['plasticity_events']}")
        print(f"   Attention modulations: {final_status['performance_metrics']['attention_modulations']}")
        print(f"   Synchronization: {final_status['performance_metrics']['synchronization_index']:.3f}")
        
        return final_status

def main():
    """Demonstrate integrated neural dynamics system"""
    print("🧠 QUARK Integrated Neural Dynamics - Phase 2")
    print("=" * 55)
    
    # Create configuration
    config = NeuralDynamicsConfig(
        num_neurons=30,
        spatial_resolution=(40, 40),
        simulation_steps=50,
        enable_plasticity=True,
        enable_attention=True
    )
    
    # Create integrated system
    system = IntegratedNeuralDynamics(config)
    
    # Run simulation
    results = system.run_simulation()
    
    # Show detailed results
    print(f"\n🧠 Neural Network Status:")
    print(f"   Neurons: {results['neural_network']['num_neurons']}")
    print(f"   Connections: {results['neural_network']['num_connections']}")
    print(f"   Neuron types: {results['neural_network']['neuron_types'][:5]}...")
    
    print(f"\n🔗 Plasticity Status:")
    print(f"   Active mechanisms: {', '.join(results['plasticity']['active_mechanisms'])}")
    print(f"   Synapses: {results['plasticity']['num_synapses']}")
    print(f"   Mean weight: {results['plasticity']['mean_weight']:.3f}")
    
    print(f"\n👁️ Attention Status:")
    print(f"   Spatial focus: {results['attention']['spatial']['focus_point']}")
    print(f"   Feature filters: {results['attention']['feature']['num_filters']}")
    print(f"   Integration weights: {results['attention']['integration_weights']}")
    
    print("\n✅ Integrated neural dynamics demonstration completed!")
    return system

if __name__ == "__main__":
    try:
        system = main()
    except Exception as e:
        print(f"❌ Integrated neural dynamics failed: {e}")
        import traceback
        traceback.print_exc()
