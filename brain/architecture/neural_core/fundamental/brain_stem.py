"""Brain Stem
The most fundamental component of the brain, responsible for regulating autonomic
functions and acting as the primary signal relay between the brain and the body.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from typing import Dict, Any

class BrainStem:
    """
    Manages the overall arousal state of the agent and serves as the central
    hub for incoming sensory information and outgoing motor commands.
    """
    def __init__(self):
        """
        Initializes the BrainStem.
        """
        # Arousal level: 0.0 (unconscious/sleep) to 1.0 (hyper-alert)
        self.arousal_level = 0.5 # Start at a baseline level
        self.arousal_decay_rate = 0.01
        self.last_stimulus_type = "none"

        print("🧠 Brain Stem initialized.")

    def step(self, sensory_input: Dict[str, Any], motor_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single time step.

        1. Receives all sensory input and modulates arousal.
        2. Receives final motor commands to be relayed.
        """
        # Decay arousal back to baseline
        self.arousal_level *= (1.0 - self.arousal_decay_rate)

        # Example: A high error signal from the limbic system could cause a spike in arousal
        error_signal = sensory_input.get("limbic_system", {}).get("error_signal", 0.0)
        if error_signal > 0.8:
            self.arousal_level = 1.0
            self.last_stimulus_type = "high_error"

        # The BrainStem would eventually have more complex logic to route data.
        # For now, it mainly manages the arousal state.

        return {
            "arousal_level": self.arousal_level,
            "relayed_motor_command": motor_output.get("action")
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the BrainStem.
        """
        return {
            "arousal_level": self.arousal_level,
            "last_stimulus_type": self.last_stimulus_type
        }
