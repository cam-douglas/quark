# brain_modules/basal_ganglia/enhanced_architecture.py

"""
Purpose: Enhanced biological models for Basal Ganglia with spiking neurons and pathway separation.
Inputs: Neural inputs, dopamine signals
Outputs: Spiking responses, pathway activations
Dependencies: numpy
"""

import numpy as np
from typing import List, Tuple, Optional
import time

class SpikingNeuron:
    """Spiking neuron model for biological plausibility"""
    def __init__(self, threshold: float = 1.0, decay: float = 0.9, refractory_period: int = 3):
        self.threshold = threshold
        self.decay = decay
        self.refractory_period = refractory_period
        self.membrane_potential = 0.0
        self.spike_history = []
        self.refractory_counter = 0
        self.last_spike_time = -1

    def update(self, input_current: float, current_time: int) -> bool:
        """Update neuron state and return if spiked"""
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.spike_history.append(0)
            return False
            
        # Update membrane potential
        self.membrane_potential = self.decay * self.membrane_potential + input_current
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            self.last_spike_time = current_time
            self.spike_history.append(1)
            return True
            
        self.spike_history.append(0)
        return False

    def get_firing_rate(self, window_size: int = 10) -> float:
        """Calculate firing rate over recent window"""
        if len(self.spike_history) < window_size:
            return 0.0
        return sum(self.spike_history[-window_size:]) / window_size

class EnhancedStriatum:
    """Striatum with direct/indirect pathway modeling"""
    def __init__(self, num_neurons: int = 100):
        self.num_neurons = num_neurons
        self.direct_pathway = [SpikingNeuron() for _ in range(num_neurons//2)]
        self.indirect_pathway = [SpikingNeuron() for _ in range(num_neurons//2)]
        self.d1_receptors = np.ones(num_neurons//2)  # Direct pathway receptors
        self.d2_receptors = np.ones(num_neurons//2)  # Indirect pathway receptors
        self.current_time = 0

    def process_input(self, input_signal: np.ndarray, dopamine_level: float) -> Tuple[float, float]:
        """Process input through direct and indirect pathways with dopamine modulation"""
        # Ensure input signal matches neuron count
        if len(input_signal) != self.num_neurons:
            input_signal = np.resize(input_signal, self.num_neurons)
        
        # Modulate receptors based on dopamine
        self.d1_receptors *= (1 + 0.1 * dopamine_level)  # D1 receptors excited by dopamine
        self.d2_receptors *= (1 - 0.1 * dopamine_level)  # D2 receptors inhibited by dopamine
        
        # Process through direct pathway (D1 receptors)
        direct_outputs = []
        for i, neuron in enumerate(self.direct_pathway):
            modulated_input = input_signal[i] * self.d1_receptors[i]
            spiked = neuron.update(modulated_input, self.current_time)
            direct_outputs.append(spiked)
        
        # Process through indirect pathway (D2 receptors)
        indirect_outputs = []
        for i, neuron in enumerate(self.indirect_pathway):
            modulated_input = input_signal[i + len(self.direct_pathway)] * self.d2_receptors[i]
            spiked = neuron.update(modulated_input, self.current_time)
            indirect_outputs.append(spiked)
        
        self.current_time += 1
        
        return sum(direct_outputs), sum(indirect_outputs)

    def get_pathway_activity(self) -> Tuple[float, float]:
        """Get current activity levels of both pathways"""
        direct_activity = np.mean([n.get_firing_rate() for n in self.direct_pathway])
        indirect_activity = np.mean([n.get_firing_rate() for n in self.indirect_pathway])
        return direct_activity, indirect_activity

class EnhancedDopamineSystem:
    """Extended dopamine system with phasic/tonic responses"""
    def __init__(self, baseline_dopamine=0.5, rpe_sensitivity=0.2):
        self.baseline_dopamine = baseline_dopamine
        self.rpe_sensitivity = rpe_sensitivity
        self.tonic_level = baseline_dopamine
        self.phasic_response = 0.0
        self.current_dopamine_level = baseline_dopamine
        self.response_history = []
        self.time_history = []

    def calculate_reward_prediction_error(self, predicted_reward: float, actual_reward: float) -> float:
        """Calculate reward prediction error"""
        return actual_reward - predicted_reward

    def update_dopamine_level(self, rpe: float, current_time: int) -> float:
        """Update with phasic/tonic response separation"""
        # Phasic response (immediate change)
        self.phasic_response = self.rpe_sensitivity * rpe
        
        # Tonic response (slow baseline adjustment)
        tonic_change = 0.01 * np.sign(rpe) * abs(rpe)
        self.tonic_level += tonic_change
        self.tonic_level = max(0.1, min(0.9, self.tonic_level))
        
        # Current level combines both responses
        self.current_dopamine_level = self.tonic_level + self.phasic_response
        self.current_dopamine_level = max(0, min(1, self.current_dopamine_level))
        
        # Record history
        self.response_history.append((self.tonic_level, self.phasic_response, self.current_dopamine_level))
        self.time_history.append(current_time)
        
        return self.current_dopamine_level

    def process_reward(self, predicted_reward: float, actual_reward: float, current_time: int) -> Tuple[float, float]:
        """Process reward and update dopamine levels"""
        rpe = self.calculate_reward_prediction_error(predicted_reward, actual_reward)
        dopamine_level = self.update_dopamine_level(rpe, current_time)
        return dopamine_level, rpe

    def get_phasic_tonic_ratio(self) -> float:
        """Get ratio of phasic to tonic response"""
        if len(self.response_history) < 2:
            return 0.0
        recent_phasic = abs(self.response_history[-1][1])
        recent_tonic = self.response_history[-1][0]
        return recent_phasic / (recent_tonic + 1e-6)

class EnhancedBasalGanglia:
    """Enhanced Basal Ganglia with biological components"""
    def __init__(self, num_neurons: int = 100):
        self.striatum = EnhancedStriatum(num_neurons)
        self.dopamine_system = EnhancedDopamineSystem()
        self.current_time = 0
        self.activity_history = []

    def process_action_selection(self, input_signal: np.ndarray, predicted_reward: float, actual_reward: float) -> Tuple[int, float]:
        """Process action selection with enhanced biological modeling"""
        # Update dopamine based on reward
        dopamine_level, rpe = self.dopamine_system.process_reward(predicted_reward, actual_reward, self.current_time)
        
        # Process through striatum
        direct_activity, indirect_activity = self.striatum.process_input(input_signal, dopamine_level)
        
        # Action selection based on pathway balance
        # Direct pathway promotes action, indirect pathway inhibits action
        action_strength = direct_activity - indirect_activity
        
        # Record activity
        self.activity_history.append({
            'time': self.current_time,
            'direct_activity': direct_activity,
            'indirect_activity': indirect_activity,
            'dopamine_level': dopamine_level,
            'rpe': rpe,
            'action_strength': action_strength
        })
        
        self.current_time += 1
        
        # Return action decision (simplified - in practice this would be more complex)
        action = 1 if action_strength > 0 else 0
        return action, action_strength

    def get_system_state(self) -> dict:
        """Get current state of the enhanced system"""
        direct_activity, indirect_activity = self.striatum.get_pathway_activity()
        phasic_tonic_ratio = self.dopamine_system.get_phasic_tonic_ratio()
        
        return {
            'direct_pathway_activity': direct_activity,
            'indirect_pathway_activity': indirect_activity,
            'dopamine_level': self.dopamine_system.current_dopamine_level,
            'phasic_tonic_ratio': phasic_tonic_ratio,
            'current_time': self.current_time
        }

if __name__ == '__main__':
    # Test the enhanced system
    print("Testing Enhanced Basal Ganglia System")
    
    # Create system
    enhanced_bg = EnhancedBasalGanglia(num_neurons=50)
    
    # Simulate some inputs
    for i in range(100):
        input_signal = np.random.rand(50)
        predicted_reward = 0.3
        actual_reward = 0.8 if i % 10 == 0 else 0.2  # Periodic high rewards
        
        action, strength = enhanced_bg.process_action_selection(input_signal, predicted_reward, actual_reward)
        
        if i % 20 == 0:
            state = enhanced_bg.get_system_state()
            print(f"Time {i}: Action={action}, Strength={strength:.3f}, "
                  f"Dopamine={state['dopamine_level']:.3f}, "
                  f"Direct={state['direct_pathway_activity']:.3f}, "
                  f"Indirect={state['indirect_pathway_activity']:.3f}")
    
    print("Enhanced Basal Ganglia test completed.")
