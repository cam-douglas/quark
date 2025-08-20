"""
Neural Network Simulator for Brain Physics

Simulates neural networks including:
- Neuron dynamics and firing patterns
- Synaptic plasticity and learning
- Network connectivity and topology
- Integration with physical brain development
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from enum import Enum

logger = logging.getLogger(__name__)


class NeuronType(Enum):
    """Types of neurons in the simulation"""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    INTERNEURON = "interneuron"
    SENSORY = "sensory"
    MOTOR = "motor"


@dataclass
class Neuron:
    """Individual neuron representation"""
    id: int
    neuron_type: NeuronType
    position: np.ndarray  # 3D position in brain space
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    refractory_period: float = 2.0  # ms
    refractory_counter: float = 0.0
    last_fire_time: float = 0.0
    spike_history: List[float] = None
    
    def __post_init__(self):
        if self.spike_history is None:
            self.spike_history = []
    
    def update(self, dt: float, input_current: float = 0.0):
        """Update neuron state"""
        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            return False
        
        # Leaky integrate-and-fire model
        tau = 20.0  # membrane time constant (ms)
        self.membrane_potential += (input_current - self.membrane_potential) * dt / tau
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = -70.0
            self.refractory_counter = self.refractory_period
            self.last_fire_time = 0.0
            self.spike_history.append(0.0)
            return True
        
        self.last_fire_time += dt
        return False


@dataclass
class Synapse:
    """Synaptic connection between neurons"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float = 1.0
    delay: float = 1.0  # ms
    plasticity: bool = True
    learning_rate: float = 0.01
    last_activity: float = 0.0


class NeuralNetwork:
    """Neural network simulation"""
    
    def __init__(self, brain_regions: List[str]):
        self.brain_regions = brain_regions
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: List[Synapse] = []
        self.region_neurons: Dict[str, List[int]] = {region: [] for region in brain_regions}
        self.time: float = 0.0
        self.dt: float = 0.1  # ms
        self.spike_times: List[Tuple[float, int]] = []
        
    def add_neuron(self, neuron: Neuron, region: str):
        """Add a neuron to the network"""
        self.neurons[neuron.id] = neuron
        if region in self.region_neurons:
            self.region_neurons[region].append(neuron.id)
        logger.info(f"Added neuron {neuron.id} to region {region}")
    
    def add_synapse(self, synapse: Synapse):
        """Add a synaptic connection"""
        self.synapses.append(synapse)
        logger.info(f"Added synapse from {synapse.pre_neuron_id} to {synapse.post_neuron_id}")
    
    def create_region_network(self, region: str, neuron_count: int, 
                             excitatory_ratio: float = 0.8):
        """Create a network of neurons in a specific brain region"""
        if region not in self.brain_regions:
            raise ValueError(f"Unknown brain region: {region}")
        
        start_id = len(self.neurons)
        
        for i in range(neuron_count):
            neuron_type = (NeuronType.EXCITATORY if np.random.random() < excitatory_ratio 
                          else NeuronType.INHIBITORY)
            
            # Random position within region (simplified)
            position = np.random.randn(3) * 10.0
            
            neuron = Neuron(
                id=start_id + i,
                neuron_type=neuron_type,
                position=position
            )
            
            self.add_neuron(neuron, region)
        
        # Create random connectivity within region
        self._create_random_connectivity(region, connection_probability=0.1)
        logger.info(f"Created network in region {region} with {neuron_count} neurons")
    
    def _create_random_connectivity(self, region: str, connection_probability: float):
        """Create random synaptic connections within a region"""
        region_neuron_ids = self.region_neurons[region]
        
        for pre_id in region_neuron_ids:
            for post_id in region_neuron_ids:
                if pre_id != post_id and np.random.random() < connection_probability:
                    weight = np.random.normal(1.0, 0.3)
                    delay = np.random.uniform(0.5, 2.0)
                    
                    synapse = Synapse(
                        pre_neuron_id=pre_id,
                        post_neuron_id=post_id,
                        weight=weight,
                        delay=delay
                    )
                    self.add_synapse(synapse)
    
    def step(self, external_inputs: Optional[Dict[int, float]] = None):
        """Simulate one time step"""
        if external_inputs is None:
            external_inputs = {}
        
        # Update all neurons
        for neuron_id, neuron in self.neurons.items():
            input_current = external_inputs.get(neuron_id, 0.0)
            
            # Add synaptic inputs
            for synapse in self.synapses:
                if synapse.post_neuron_id == neuron_id:
                    # Check if pre-synaptic neuron fired recently
                    pre_neuron = self.neurons[synapse.pre_neuron_id]
                    if (self.time - pre_neuron.last_fire_time) <= synapse.delay:
                        input_current += synapse.weight * 10.0  # Synaptic current
            
            # Update neuron
            if neuron.update(self.dt, input_current):
                self.spike_times.append((self.time, neuron_id))
        
        # Update synaptic plasticity
        self._update_synaptic_plasticity()
        
        self.time += self.dt
    
    def _update_synaptic_plasticity(self):
        """Update synaptic weights based on activity"""
        for synapse in self.synapses:
            if not synapse.plasticity:
                continue
            
            pre_neuron = self.neurons[synapse.pre_neuron_id]
            post_neuron = self.neurons[synapse.post_neuron_id]
            
            # Simple STDP-like plasticity
            if pre_neuron.last_fire_time < 5.0 and post_neuron.last_fire_time < 5.0:
                # Both neurons recently active - strengthen connection
                synapse.weight += synapse.learning_rate * 0.1
            elif pre_neuron.last_fire_time > 50.0 or post_neuron.last_fire_time > 50.0:
                # Neurons inactive - weaken connection
                synapse.weight -= synapse.learning_rate * 0.05
            
            # Keep weights in reasonable bounds
            synapse.weight = np.clip(synapse.weight, 0.0, 5.0)
    
    def get_network_activity(self) -> Dict:
        """Get current network activity metrics"""
        active_neurons = sum(1 for n in self.neurons.values() 
                           if n.refractory_counter <= 0)
        
        recent_spikes = [spike for spike in self.spike_times 
                        if self.time - spike[0] < 100.0]
        
        return {
            'total_neurons': len(self.neurons),
            'active_neurons': active_neurons,
            'recent_spikes': len(recent_spikes),
            'average_firing_rate': len(recent_spikes) / 100.0 if recent_spikes else 0.0,
            'time': self.time
        }
    
    def get_region_activity(self, region: str) -> Dict:
        """Get activity metrics for a specific brain region"""
        if region not in self.region_neurons:
            return {}
        
        region_neuron_ids = self.region_neurons[region]
        region_neurons = [self.neurons[nid] for nid in region_neuron_ids]
        
        active_count = sum(1 for n in region_neurons if n.refractory_counter <= 0)
        recent_spikes = [spike for spike in self.spike_times 
                        if spike[1] in region_neuron_ids and 
                        self.time - spike[0] < 100.0]
        
        return {
            'region': region,
            'neuron_count': len(region_neurons),
            'active_neurons': active_count,
            'recent_spikes': len(recent_spikes),
            'firing_rate': len(recent_spikes) / 100.0 if recent_spikes else 0.0
        }


class NeuralSimulator:
    """Main neural simulation controller"""
    
    def __init__(self, brain_regions: List[str]):
        self.network = NeuralNetwork(brain_regions)
        self.simulation_running = False
        self.external_stimuli = {}
        
    def setup_brain_regions(self, region_configs: Dict[str, Dict]):
        """Setup neural networks for different brain regions"""
        for region, config in region_configs.items():
            neuron_count = config.get('neuron_count', 100)
            excitatory_ratio = config.get('excitatory_ratio', 0.8)
            
            self.network.create_region_network(region, neuron_count, excitatory_ratio)
            
            # Add inter-region connections
            if 'connections' in config:
                self._create_inter_region_connections(region, config['connections'])
        
        logger.info("Brain region neural networks setup complete")
    
    def _create_inter_region_connections(self, source_region: str, 
                                       target_regions: List[str]):
        """Create connections between different brain regions"""
        source_neurons = self.network.region_neurons[source_region]
        
        for target_region in target_regions:
            if target_region in self.network.region_neurons:
                target_neurons = self.network.region_neurons[target_region]
                
                # Create sparse inter-region connections
                connection_prob = 0.05  # Lower probability for inter-region
                
                for source_id in source_neurons:
                    for target_id in target_neurons:
                        if np.random.random() < connection_prob:
                            weight = np.random.normal(0.8, 0.2)
                            delay = np.random.uniform(1.0, 3.0)
                            
                            synapse = Synapse(
                                pre_neuron_id=source_id,
                                post_neuron_id=target_id,
                                weight=weight,
                                delay=delay,
                                plasticity=True
                            )
                            self.network.add_synapse(synapse)
    
    def add_external_stimulus(self, neuron_id: int, stimulus_function: Callable[[float], float]):
        """Add external stimulus to a specific neuron"""
        self.external_stimuli[neuron_id] = stimulus_function
        logger.info(f"Added external stimulus to neuron {neuron_id}")
    
    def run_simulation(self, duration: float, record_activity: bool = True):
        """Run neural simulation for specified duration"""
        self.simulation_running = True
        steps = int(duration / self.network.dt)
        
        logger.info(f"Starting neural simulation for {duration}ms ({steps} steps)")
        
        for step in range(steps):
            # Calculate external inputs
            external_inputs = {}
            for neuron_id, stimulus_func in self.external_stimuli.items():
                external_inputs[neuron_id] = stimulus_func(self.network.time)
            
            # Step the network
            self.network.step(external_inputs)
            
            if step % 100 == 0:
                activity = self.network.get_network_activity()
                logger.info(f"Step {step}: {activity['recent_spikes']} recent spikes")
        
        self.simulation_running = False
        logger.info(f"Neural simulation completed: {duration}ms")
        
        if record_activity:
            return self.get_simulation_results()
    
    def get_simulation_results(self) -> Dict:
        """Get comprehensive simulation results"""
        results = {
            'network_activity': self.network.get_network_activity(),
            'region_activities': {},
            'spike_data': self.network.spike_times.copy(),
            'final_weights': [(s.pre_neuron_id, s.post_neuron_id, s.weight) 
                             for s in self.network.synapses]
        }
        
        for region in self.network.brain_regions:
            results['region_activities'][region] = self.network.get_region_activity(region)
        
        return results
