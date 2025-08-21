#!/usr/bin/env python3
"""
🧠 Thalamus - Information Relay Module
Handles information routing, attention modulation, and sensory integration
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class SensoryInput:
    """Represents sensory input from various modalities"""
    modality: str  # visual, auditory, tactile, etc.
    content: Any
    intensity: float
    priority: float
    timestamp: float
    source_id: str

@dataclass
class AttentionSignal:
    """Attention modulation signal"""
    target_modality: str
    strength: float
    duration: float
    source: str

class InformationRelay:
    """Thalamus information relay and attention system"""
    
    def __init__(self):
        self.sensory_inputs: Dict[str, List[SensoryInput]] = defaultdict(list)
        self.attention_signals: List[AttentionSignal] = []
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        
        # Attention modulation
        self.attention_focus = "general"
        self.attention_strength = 1.0
        
        # Neural representation
        self.relay_neurons = np.random.rand(200, 30)  # 200 relay neurons, 30 features each
        
        # Initialize routing for common modalities
        self._initialize_routing()
    
    def _initialize_routing(self):
        """Initialize default routing table"""
        self.routing_table["visual"] = ["prefrontal_cortex", "working_memory", "conscious_agent"]
        self.routing_table["auditory"] = ["prefrontal_cortex", "working_memory", "conscious_agent"]
        self.routing_table["tactile"] = ["motor_control", "conscious_agent"]
        self.routing_table["internal"] = ["default_mode_network", "conscious_agent"]
    
    def receive_sensory_input(self, input_data: SensoryInput):
        """Receive sensory input from various modalities"""
        self.sensory_inputs[input_data.modality].append(input_data)
        
        # Maintain only recent inputs (last 100 per modality)
        if len(self.sensory_inputs[input_data.modality]) > 100:
            self.sensory_inputs[input_data.modality] = self.sensory_inputs[input_data.modality][-100:]
        
        # Update neural representations
        self._update_relay_neurons(input_data)
    
    def _update_relay_neurons(self, input_data: SensoryInput):
        """Update neural representation based on sensory input"""
        # Find neurons associated with this modality
        modality_idx = hash(input_data.modality) % 50  # Simple hash for modality
        
        for i in range(50):  # Update 50 neurons for this modality
            neuron_idx = modality_idx * 50 + i
            if neuron_idx < len(self.relay_neurons):
                # Update based on input intensity and priority
                activation = (input_data.intensity + input_data.priority) / 2.0
                self.relay_neurons[neuron_idx] += np.random.normal(0, 0.01, 30) * activation
                self.relay_neurons[neuron_idx] = np.clip(self.relay_neurons[neuron_idx], 0.0, 1.0)
    
    def route_information(self, modality: str, target: str) -> List[SensoryInput]:
        """Route sensory information to target brain regions"""
        if modality not in self.sensory_inputs:
            return []
        
        # Get recent inputs for this modality
        recent_inputs = self.sensory_inputs[modality][-10:]  # Last 10 inputs
        
        # Apply attention modulation
        modulated_inputs = self._apply_attention_modulation(recent_inputs, modality)
        
        return modulated_inputs
    
    def _apply_attention_modulation(self, inputs: List[SensoryInput], modality: str) -> List[SensoryInput]:
        """Apply attention modulation to sensory inputs"""
        # Check if there are attention signals for this modality
        modality_attention = 1.0
        for signal in self.attention_signals:
            if signal.target_modality == modality:
                modality_attention = max(modality_attention, signal.strength)
        
        # Apply attention modulation
        modulated_inputs = []
        for input_data in inputs:
            # Create modulated copy
            modulated_input = SensoryInput(
                modality=input_data.modality,
                content=input_data.content,
                intensity=input_data.intensity * modality_attention,
                priority=input_data.priority * modality_attention,
                timestamp=input_data.timestamp,
                source_id=input_data.source_id
            )
            modulated_inputs.append(modulated_input)
        
        return modulated_inputs
    
    def set_attention_focus(self, modality: str, strength: float, duration: float, source: str):
        """Set attention focus on specific modality"""
        attention_signal = AttentionSignal(
            target_modality=modality,
            strength=strength,
            duration=duration,
            source=source
        )
        
        self.attention_signals.append(attention_signal)
        self.attention_focus = modality
        self.attention_strength = strength
        
        # Clean up expired attention signals
        self._cleanup_expired_signals()
    
    def _cleanup_expired_signals(self):
        """Remove expired attention signals"""
        current_time = 0.0  # Will be set by external time system
        self.attention_signals = [
            signal for signal in self.attention_signals
            if signal.duration > 0  # Simple cleanup for now
        ]
    
    def get_routing_info(self, modality: str) -> Dict[str, Any]:
        """Get routing information for a specific modality"""
        if modality not in self.routing_table:
            return {"targets": [], "attention": 0.0, "input_count": 0}
        
        # Calculate attention strength for this modality
        attention_strength = 0.0
        for signal in self.attention_signals:
            if signal.target_modality == modality:
                attention_strength = max(attention_strength, signal.strength)
        
        return {
            "targets": self.routing_table[modality],
            "attention": attention_strength,
            "input_count": len(self.sensory_inputs.get(modality, [])),
            "recent_inputs": [
                {
                    "content": str(input_data.content)[:50],
                    "intensity": input_data.intensity,
                    "priority": input_data.priority
                }
                for input_data in self.sensory_inputs.get(modality, [])[-5:]  # Last 5 inputs
            ]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current thalamus status"""
        total_inputs = sum(len(inputs) for inputs in self.sensory_inputs.values())
        
        return {
            "attention_focus": self.attention_focus,
            "attention_strength": self.attention_strength,
            "total_sensory_inputs": total_inputs,
            "active_modalities": list(self.sensory_inputs.keys()),
            "attention_signals": len(self.attention_signals),
            "routing_table_size": len(self.routing_table)
        }
    
    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Step thalamus system forward"""
        # Process new sensory inputs
        if "sensory_inputs" in inputs:
            for input_data in inputs["sensory_inputs"]:
                sensory_input = SensoryInput(
                    modality=input_data["modality"],
                    content=input_data["content"],
                    intensity=input_data.get("intensity", 0.5),
                    priority=input_data.get("priority", 0.5),
                    timestamp=input_data.get("timestamp", 0.0),
                    source_id=input_data.get("source_id", "unknown")
                )
                self.receive_sensory_input(sensory_input)
        
        # Process attention commands
        if "attention_commands" in inputs:
            for command in inputs["attention_commands"]:
                self.set_attention_focus(
                    command["modality"],
                    command.get("strength", 1.0),
                    command.get("duration", 1.0),
                    command.get("source", "system")
                )
        
        # Process routing requests
        routing_results = {}
        if "routing_requests" in inputs:
            for request in inputs["routing_requests"]:
                modality = request["modality"]
                target = request.get("target", "default")
                routed_inputs = self.route_information(modality, target)
                routing_results[modality] = len(routed_inputs)
        
        return {
            "status": self.get_status(),
            "routing_results": routing_results,
            "attention_focus": self.attention_focus
        }
