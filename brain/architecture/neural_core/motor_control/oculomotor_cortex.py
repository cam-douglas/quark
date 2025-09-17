"""Oculomotor Cortex
Controls eye movements (camera gaze) to focus on salient features.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import Dict, Any

class OculomotorCortex:
    """
    Simulates the Motor Function Area for eye movement and orientation.
    Controls the direction of the camera's gaze.
    """
    def __init__(self, update_frequency: int = 20):
        """
        Initializes the OculomotorCortex.
        
        Args:
            update_frequency (int): How many steps to wait before a potential gaze shift.
        """
        self.gaze_target = np.array([0.0, 0.0])  # (pan, tilt)
        self.update_frequency = update_frequency
        self.steps_since_last_update = 0

        print("🧠 Oculomotor Cortex initialized.")

    def step(self, salience_map: np.ndarray = None) -> Dict[str, Any]:
        """
        Determines the next gaze target.

        Args:
            salience_map: A 2D array indicating visually interesting areas.

        Returns:
            A dictionary containing the desired gaze direction.
        """
        self.steps_since_last_update += 1

        if self.steps_since_last_update >= self.update_frequency:
            self.steps_since_last_update = 0

            if salience_map is not None and salience_map.size > 0:
                # Find the most salient point and target it
                coords = np.unravel_index(np.argmax(salience_map), salience_map.shape)
                # Normalize coordinates to be a pan/tilt target between -1 and 1
                height, width = salience_map.shape
                pan = (coords[1] / width - 0.5) * 2.0
                tilt = (coords[0] / height - 0.5) * 2.0
                self.gaze_target = np.array([pan, tilt])
            else:
                # Default behavior: random saccade
                self.gaze_target = np.random.uniform(-0.5, 0.5, size=2)

        return {"gaze_target": self.gaze_target}
