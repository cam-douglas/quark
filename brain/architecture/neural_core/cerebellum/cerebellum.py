"""Cerebellum Module - Phase 3 Prototype
Provides fine-tuning and coordination for motor and cognitive actions.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import numpy as np
from collections import deque

class Cerebellum:
    """
    A more sophisticated cerebellum model that provides motor command smoothing
    and predictive error correction.
    """
    def __init__(self, num_actuators: int, smoothing_window: int = 5):
        """
        Initializes the Cerebellum.
        Args:
            num_actuators: The number of motor control signals.
            smoothing_window: The number of past commands to average for smoothing.
        """
        self.num_actuators = num_actuators

        # For motor command smoothing
        self.smoothing_window = smoothing_window
        self.command_history = deque(maxlen=smoothing_window)

        # For predictive error correction (simulating a forward model)
        # This is a learned adjustment applied to motor commands.
        self.predictive_correction = np.zeros(num_actuators)
        self.correction_learning_rate = 0.01

        print(f"🧠 Cerebellum initialized for {num_actuators} actuators (Smoothing window: {smoothing_window}).")

    def refine_motor_command(self, motor_command: np.ndarray) -> np.ndarray:
        """
        Refines a raw motor command from the Motor Cortex by smoothing it and
        applying a learned predictive correction.

        Args:
            motor_command: The raw control array (ctrl) from the Motor Cortex.

        Returns:
            The refined, coordinated control array.
        """
        if motor_command is None:
            return np.zeros(self.num_actuators)

        # 1. Apply the learned predictive correction
        corrected_command = motor_command + self.predictive_correction

        # 2. Add the new command to history for smoothing
        self.command_history.append(corrected_command)

        # 3. Smooth the command using a moving average
        if not self.command_history:
            return np.zeros(self.num_actuators) # Should not happen if command is not None

        smoothed_command = np.mean(list(self.command_history), axis=0)

        return smoothed_command

    def update_from_sensory_feedback(self, command_error: float):
        """
        Updates the predictive model based on sensory feedback (error).
        If the last action resulted in a large error (e.g., falling, high pose error),
        the cerebellum adjusts its predictive correction to compensate.

        Args:
            command_error: A scalar value representing the outcome error of the last action.
                           (e.g., 1.0 for a fall, or a scaled pose error).
        """
        # This is a simplified learning rule. If there was a large error,
        # it adjusts the predictive correction in a random direction to encourage
        # exploration of a better motor strategy. A more advanced model would
        # use the specific direction of the error to make a more targeted correction.

        if command_error > 0.5: # Only update on significant errors
            # Generate a small, random adjustment
            adjustment = np.random.randn(self.num_actuators) * command_error * self.correction_learning_rate
            self.predictive_correction -= adjustment # Adjust to counteract the error

            # Keep the correction within reasonable bounds
            self.predictive_correction = np.clip(self.predictive_correction, -0.1, 0.1)
