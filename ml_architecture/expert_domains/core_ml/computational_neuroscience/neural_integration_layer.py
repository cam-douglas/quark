#!/usr/bin/env python3
"""
🧠 Neural Integration Layer for Pillar 1: Basic Neural Dynamics
Connects brain modules to actual neural populations and implements biological neural dynamics

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Wire brain modules to neural components and implement cortical-subcortical loops
**Validation Level:** Biological neural dynamics verification
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, field
from enum import Enum

from ...........................................................neural_components import (
    SpikingNeuron, HebbianSynapse, STDP, NeuralPopulation as NeuralPopulationComponent,
    calculate_synchrony, calculate_oscillation_power
)

class MessageType(Enum):
    """Message types for neural integration"""
    SPIKE = "spike"
    MODULATION = "modulation"
    COMMAND = "command"
    TELEMETRY = "telemetry"
    PLASTICITY = "plasticity"

@dataclass
class NeuralMessage:
    """Neural message with biological properties"""
    message_type: MessageType
    source_population: str
    target_population: str
    spike_count: int = 0
    firing_rate: float = 0.0
    modulation_strength: float = 0.0
    timestamp: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)

class NeuralPopulation:
    """Biological neural population with realistic dynamics"""
    
    def __init__(self, name: str, neuron_count: int, neuron_type: str = "regular_spiking"):
        self.name = name
        self.neuron_count = neuron_count
        self.neuron_type = neuron_type
        
        # Create neural population using NeuralPopulation from neural_components
        self.population = NeuralPopulationComponent(
            population_id=name,
            num_neurons=neuron_count,
            neuron_type=neuron_type,
            connectivity=0.1
        )
        
        # Population state
        self.firing_rates = np.zeros(neuron_count)
        self.spike_times = []
        self.synchrony = 0.0
        self.oscillation_power = 0.0
        
    def step(self, dt: float, input_current: float = 0.0) -> Dict[str, Any]:
        """Step neural population forward in time"""
        
        # Convert single input to list for each neuron
        input_list = [input_current] * self.neuron_count
        
        # Step the population
        spike_events = self.population.step(input_list, dt)
        
        # Extract spikes - spike_events is a list of booleans
        spikes = [(i, self.time) for i, spiked in enumerate(spike_events) if spiked]
        self.spike_times.extend([self.time for spiked in spike_events if spiked])
        
        # Update population metrics
        self.firing_rates = np.array([self.population.neurons[i].get_firing_rate() 
                                     for i in range(self.neuron_count)])
        self.synchrony = calculate_synchrony(self.spike_times)
        self.oscillation_power = calculate_oscillation_power(self.spike_times, 10.0)  # 10 Hz
        
        return {
            "spikes": spikes,
            "firing_rate": self.population.population_firing_rate,
            "synchrony": self.synchrony,
            "oscillation_power": self.oscillation_power,
            "active_neurons": len(spikes)
        }
    
    @property
    def neurons(self):
        """Access to neurons for compatibility"""
        return self.population.neurons

class CorticalSubcorticalLoop:
    """Implements PFC→BG→Thalamus→PFC feedback loop"""
    
    def __init__(self):
        # Create neural populations
        self.pfc_population = NeuralPopulation("PFC", 1000, "regular_spiking")
        self.bg_population = NeuralPopulation("BG", 500, "fast_spiking") 
        self.thalamus_population = NeuralPopulation("Thalamus", 800, "bursting")
        
        # Create connections with realistic weights
        # For population-level connections, we'll use representative neuron IDs
        self.pfc_to_bg = HebbianSynapse(
            pre_neuron_id=0,  # Representative PFC neuron
            post_neuron_id=1000,  # Representative BG neuron
            initial_weight=0.1,
            learning_rate=0.01
        )
        
        self.bg_to_thalamus = HebbianSynapse(
            pre_neuron_id=1000,  # Representative BG neuron
            post_neuron_id=1500,  # Representative Thalamus neuron
            initial_weight=0.15,
            learning_rate=0.008
        )
        
        self.thalamus_to_pfc = HebbianSynapse(
            pre_neuron_id=1500,  # Representative Thalamus neuron
            post_neuron_id=0,  # Representative PFC neuron
            initial_weight=0.12,
            learning_rate=0.012
        )
        
        # STDP mechanisms
        self.pfc_bg_stdp = STDP(
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.1,
            A_minus=0.1
        )
        
        self.bg_thalamus_stdp = STDP(
            tau_plus=15.0,
            tau_minus=15.0,
            A_plus=0.08,
            A_minus=0.08
        )
        
        self.thalamus_pfc_stdp = STDP(
            tau_plus=25.0,
            tau_minus=25.0,
            A_plus=0.12,
            A_minus=0.12
        )
        
        # Loop state
        self.loop_activity = []
        self.feedback_strength = 0.0
        
    def step(self, dt: float, external_input: float = 0.0) -> Dict[str, Any]:
        """Step the cortical-subcortical loop forward"""
        
        # PFC receives external input + thalamic feedback
        thalamic_feedback = self.thalamus_to_pfc.weight * external_input  # Simplified feedback
        pfc_input = external_input + thalamic_feedback
        
        # Step PFC population
        pfc_output = self.pfc_population.step(dt, pfc_input)
        
        # PFC → BG (simplified connection)
        bg_input = self.pfc_to_bg.weight * pfc_output["firing_rate"]
        bg_output = self.bg_population.step(dt, bg_input)
        
        # BG → Thalamus (simplified connection)
        thalamus_input = self.bg_to_thalamus.weight * bg_output["firing_rate"]
        thalamus_output = self.thalamus_population.step(dt, thalamus_input)
        
        # Thalamus → PFC (feedback)
        thalamus_feedback = self.thalamus_to_pfc.weight * thalamus_output["firing_rate"]
        
        # Apply Hebbian learning to synapses
        self.pfc_to_bg.update_hebbian(
            pre_spike=len(pfc_output["spikes"]) > 0,
            post_spike=len(bg_output["spikes"]) > 0,
            pre_firing_rate=pfc_output["firing_rate"],
            post_firing_rate=bg_output["firing_rate"]
        )
        
        self.bg_to_thalamus.update_hebbian(
            pre_spike=len(bg_output["spikes"]) > 0,
            post_spike=len(thalamus_output["spikes"]) > 0,
            pre_firing_rate=bg_output["firing_rate"],
            post_firing_rate=thalamus_output["firing_rate"]
        )
        
        self.thalamus_to_pfc.update_hebbian(
            pre_spike=len(thalamus_output["spikes"]) > 0,
            post_spike=len(pfc_output["spikes"]) > 0,
            pre_firing_rate=thalamus_output["firing_rate"],
            post_firing_rate=pfc_output["firing_rate"]
        )
        
        # Calculate loop metrics
        self.feedback_strength = self.thalamus_to_pfc.weight
        self.loop_activity.append({
            "pfc_rate": pfc_output["firing_rate"],
            "bg_rate": bg_output["firing_rate"], 
            "thalamus_rate": thalamus_output["firing_rate"],
            "feedback_strength": self.feedback_strength
        })
        
        return {
            "pfc": pfc_output,
            "bg": bg_output,
            "thalamus": thalamus_output,
            "feedback_strength": self.feedback_strength,
            "loop_stability": self._calculate_loop_stability()
        }
    
    def _calculate_loop_stability(self) -> float:
        """Calculate stability of the cortical-subcortical loop"""
        if len(self.loop_activity) < 10:
            return 0.0
            
        recent_activity = self.loop_activity[-10:]
        rates = [a["pfc_rate"] for a in recent_activity]
        
        # Calculate coefficient of variation (lower = more stable)
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        
        if mean_rate == 0:
            return 0.0
            
        cv = std_rate / mean_rate
        stability = max(0.0, 1.0 - cv)  # Higher stability = lower CV
        
        return stability

class NeuralIntegrationLayer:
    """Main integration layer connecting brain modules to neural dynamics"""
    
    def __init__(self, stage: str = "F"):
        self.stage = stage
        self.time = 0.0
        self.dt = 0.001  # 1ms timestep
        
        # Create cortical-subcortical loop
        self.cortical_loop = CorticalSubcorticalLoop()
        
        # Population mappings
        self.populations = {
            "pfc": self.cortical_loop.pfc_population,
            "basal_ganglia": self.cortical_loop.bg_population,
            "thalamus": self.cortical_loop.thalamus_population
        }
        
        # Message conversion buffers
        self.message_buffer = []
        self.neural_buffer = []
        
        # Integration metrics
        self.integration_metrics = {
            "message_to_spike_conversion": 0.0,
            "spike_to_message_conversion": 0.0,
            "loop_activity": 0.0,
            "plasticity_events": 0
        }
        
    def convert_message_to_spike(self, message: Dict[str, Any]) -> NeuralMessage:
        """Convert brain module message to neural spike pattern"""
        
        message_type = message.get("kind", "Command")
        source = message.get("src", "unknown")
        target = message.get("dst", "unknown")
        
        # Convert message priority to spike count
        priority = message.get("priority", 0)
        spike_count = max(1, int(priority * 10))
        
        # Convert payload to neural parameters
        payload = message.get("payload", {})
        firing_rate = payload.get("firing_rate", 0.1)
        modulation = payload.get("modulation", 0.0)
        
        neural_msg = NeuralMessage(
            message_type=MessageType.COMMAND,
            source_population=source,
            target_population=target,
            spike_count=spike_count,
            firing_rate=firing_rate,
            modulation_strength=modulation,
            timestamp=self.time,
            payload=payload
        )
        
        self.integration_metrics["message_to_spike_conversion"] += 1
        return neural_msg
    
    def convert_spike_to_message(self, neural_output: Dict[str, Any], 
                               population_name: str) -> Dict[str, Any]:
        """Convert neural population output to brain module message"""
        
        message = {
            "kind": "Telemetry",
            "src": population_name,
            "dst": "architecture_agent",
            "priority": int(neural_output["firing_rate"] * 10),
            "payload": {
                "firing_rate": neural_output["firing_rate"],
                "synchrony": neural_output["synchrony"],
                "oscillation_power": neural_output["oscillation_power"],
                "active_neurons": neural_output["active_neurons"],
                "spike_count": len(neural_output["spikes"])
            }
        }
        
        self.integration_metrics["spike_to_message_conversion"] += 1
        return message
    
    def step(self, brain_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Main integration step - process brain messages through neural dynamics"""
        
        # Convert brain messages to neural inputs
        neural_inputs = []
        for msg in brain_messages:
            neural_msg = self.convert_message_to_spike(msg)
            neural_inputs.append(neural_msg)
        
        # Step cortical-subcortical loop
        external_input = sum([n.firing_rate for n in neural_inputs])
        loop_output = self.cortical_loop.step(self.dt, external_input)
        
        # Convert neural outputs back to brain messages
        brain_outputs = []
        for pop_name, pop_output in loop_output.items():
            if pop_name in ["pfc", "bg", "thalamus"]:
                brain_msg = self.convert_spike_to_message(pop_output, pop_name)
                brain_outputs.append(brain_msg)
        
        # Update integration metrics
        self.integration_metrics["loop_activity"] = loop_output["loop_stability"]
        self.integration_metrics["plasticity_events"] += len(loop_output["pfc"]["spikes"])
        
        # Update time
        self.time += self.dt
        
        return brain_outputs, self.integration_metrics
    
    def get_biological_metrics(self) -> Dict[str, Any]:
        """Get biological validation metrics"""
        return {
            "firing_rates": {
                "pfc": self.cortical_loop.pfc_population.firing_rates.mean(),
                "bg": self.cortical_loop.bg_population.firing_rates.mean(),
                "thalamus": self.cortical_loop.thalamus_population.firing_rates.mean()
            },
            "synchrony": {
                "pfc": self.cortical_loop.pfc_population.synchrony,
                "bg": self.cortical_loop.bg_population.synchrony,
                "thalamus": self.cortical_loop.thalamus_population.synchrony
            },
            "oscillation_power": {
                "pfc": self.cortical_loop.pfc_population.oscillation_power,
                "bg": self.cortical_loop.bg_population.oscillation_power,
                "thalamus": self.cortical_loop.thalamus_population.oscillation_power
            },
            "loop_stability": self.cortical_loop._calculate_loop_stability(),
            "feedback_strength": self.cortical_loop.feedback_strength,
            "integration_metrics": self.integration_metrics
        }
