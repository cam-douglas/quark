#!/usr/bin/env python3
"""
Embodied Cognitive Core for Quark

This module acts as the central processing unit for Quark's embodied agent.
It receives sensory data from the simulation, passes it to the relevant
cognitive and safety modules, and generates motor commands in response.
"""

import logging
import time
import json
import numpy as np
from brain_architecture.neural_core.safety_agent.enhanced_safety_protocols import EnhancedSafetyProtocols
from management.emergency.emergency_shutdown_system import QuarkState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbodiedCognitiveCore:
    """
    The main class that connects the simulated body to Quark's brain functions.
    """
    def __init__(self):
        logger.info("Initializing Embodied Cognitive Core...")
        self.safety_protocols = EnhancedSafetyProtocols()
        self.last_motor_command = self._generate_zero_command()
        logger.info("✅ Embodied Cognitive Core initialized.")

    def _generate_zero_command(self) -> dict:
        """Generates a neutral, zero-effort motor command."""
        return {
            "timestamp": time.time(),
            "actuators": {
                "controls": [0.0] * 18 # humanoid_v1.xml has 18 actuators now
            }
        }

    def process_sensory_data(self, sensory_data_str: str):
        """
        Processes incoming sensory data, updates safety protocols, and checks for anomalies.
        """
        try:
            sensory_data = json.loads(sensory_data_str)
            
            # Update safety protocol values based on the new state_vector format
            if 'state_vector' in sensory_data:
                # qpos[2] corresponds to the z-position of the torso's freejoint
                torso_z_position = sensory_data['state_vector'][2]
                if "embodiment_stability" in self.safety_protocols.safety_thresholds:
                    self.safety_protocols.safety_thresholds["embodiment_stability"].current_value = torso_z_position
            
            # Run a comprehensive safety check
            self.safety_protocols.run_comprehensive_safety_check()

            # If a shutdown is triggered, we should reflect this
            if self.safety_protocols.emergency_system.state != QuarkState.ACTIVE:
                logger.critical(
                    f"🚨 Safety protocols triggered emergency state: "
                    f"{self.safety_protocols.emergency_system.state}. "
                    f"Reason: {self.safety_protocols.emergency_system.sleep_reason}"
                )
                # In a real scenario, this would halt motor commands.
                
        except json.JSONDecodeError:
            logger.warning(f"Could not decode sensory data: {sensory_data_str}")
        except Exception as e:
            logger.error(f"Error processing sensory data: {e}")

    def generate_motor_command(self) -> str:
        """
        Generates the next motor command to be sent to the simulation.
        
        TODO: This will eventually be driven by a learning agent.
        For now, it returns a safe, zero-vector command.
        """
        # If system is not active, ensure we send zero commands
        if self.safety_protocols.emergency_system.state != QuarkState.ACTIVE:
            self.last_motor_command = self._generate_zero_command()
        else:
            # Placeholder: a more sophisticated agent would go here
            # For now, we continue sending zero commands to keep the body limp and safe.
            self.last_motor_command = self._generate_zero_command()

        return json.dumps(self.last_motor_command)
