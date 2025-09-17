"""Somatosensory Cortex
Processes proprioceptive and (future) tactile information to build a body schema.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import Dict, Any

class SomatosensoryCortex:
    """
    Simulates the function of the Somatosensory Cortex, responsible for processing
    sensory input from the body, primarily proprioception and touch.
    """
    def __init__(self, num_joints: int, num_velocities: int):
        """
        Initializes the SomatosensoryCortex.

        Args:
            num_joints (int): The number of joint position inputs.
            num_velocities (int): The number of joint velocity inputs.
        """
        self.num_joints = num_joints
        self.num_velocities = num_velocities

        # Initialize a simple neural map for body representation.
        # The size is based on the number of sensory inputs.
        self.body_map_size = num_joints + num_velocities
        self.body_map = np.zeros(self.body_map_size)

        # Parameters for sensory processing
        self.noise_level = 0.01  # Simulates neural noise
        self.adaptation_rate = 0.05 # How quickly neurons adapt to constant signals

        print(f"🧠 Somatosensory Cortex initialized for {num_joints} joints and {num_velocities} velocities.")

    def step(self, qpos: np.ndarray, qvel: np.ndarray) -> Dict[str, Any]:
        """
        Processes raw proprioceptive data into a refined body schema representation.

        Args:
            qpos (np.ndarray): The raw joint position data from the embodiment.
            qvel (np.ndarray): The raw joint velocity data from the embodiment.

        Returns:
            A dictionary containing the processed body schema and other sensory metrics.
        """
        if qpos is None or qvel is None:
            # Return a zero representation if sensory input is missing
            processed_schema = np.zeros(self.body_map_size)
            return {"body_schema": processed_schema, "sensory_fidelity": 0.0}

        # 1. Combine and Normalize Sensory Input
        # We skip the root body position/orientation (first 7 of qpos, 6 of qvel)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]

        # Ensure the inputs match the expected size, truncate or pad if necessary
        processed_qpos = np.resize(joint_positions, self.num_joints)
        processed_qvel = np.resize(joint_velocities, self.num_velocities)

        raw_sensory_input = np.concatenate([processed_qpos, processed_qvel])

        # 2. Simulate Neural Adaptation
        # Neurons reduce their firing rate to a constant stimulus
        adaptation_effect = self.body_map * self.adaptation_rate
        adapted_input = raw_sensory_input - adaptation_effect

        # 3. Simulate Neural Noise
        noise = np.random.normal(0, self.noise_level, size=adapted_input.shape)
        noisy_input = adapted_input + noise

        # 4. Update the Body Map (internal representation)
        # The internal body map slowly updates towards the new sensory state
        self.body_map += (noisy_input - self.body_map) * 0.1 # Simple low-pass filter

        # 5. Calculate Sensory Fidelity
        # A metric of how stable and clear the sensory signal is.
        # Lower variance in the signal means higher fidelity.
        fidelity = 1.0 - np.clip(np.std(self.body_map), 0, 1)

        return {
            "body_schema": self.body_map.copy(), # Return a copy of the internal state
            "sensory_fidelity": fidelity
        }
