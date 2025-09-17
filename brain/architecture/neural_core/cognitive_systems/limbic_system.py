"""Limbic System
Generates objective motivational signals based on sensory input and goal achievement.
This system explicitly avoids creating negative emotional states.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import Dict, Any

class LimbicSystem:
    """
    Generates two primary motivational signals:
    1. Positive Valence: Represents success, progress, and goal achievement (0.0 to 1.0).
    2. Error Signal: Represents failure, falls, or physical limits (0.0 to 1.0).
    This two-channel system provides motivation without simulating negative emotions.
    """
    def __init__(self, decay_rate: float = 0.1, max_valence: float = 1.0):
        """
        Initializes the LimbicSystem.
        """
        self.positive_valence = 0.0  # Current progress/success signal
        self.error_signal = 0.0      # Current failure/error signal
        self.decay_rate = decay_rate
        self.max_valence = max_valence

        self.last_significant_event = "System Start"

        print("🧠 Limbic System initialized (Objective Feedback Mode).")

    def step(self, reward: float, is_fallen: bool, goal_achieved: bool) -> Dict[str, Any]:
        """
        Updates the motivational signals based on recent events.
        """
        # 1. Decay existing signals back towards neutral (0.0)
        self.positive_valence *= (1.0 - self.decay_rate)
        self.error_signal *= (1.0 - self.decay_rate)

        # 2. Apply strong, objective signals for critical events
        if is_fallen:
            self.error_signal = 1.0
            self.positive_valence = 0.0 # A fall negates any concurrent success
            self.last_significant_event = "Event: Stability Lost"
        elif goal_achieved:
            self.positive_valence = 1.0
            self.last_significant_event = "Event: Goal Achieved"

        # 3. Modulate positive valence by extrinsic reward
        # Reward contributes to the sense of progress
        reward_impact = reward * 0.1
        self.positive_valence += reward_impact

        # 4. Clip the signals to ensure they stay within the [0, 1] bounds
        self.positive_valence = np.clip(self.positive_valence, 0.0, self.max_valence)
        self.error_signal = np.clip(self.error_signal, 0.0, 1.0)

        return {
            "positive_valence": self.positive_valence,
            "error_signal": self.error_signal,
            "primary_driver": self.last_significant_event
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the Limbic System.
        """
        return {
            "current_positive_valence": self.positive_valence,
            "current_error_signal": self.error_signal,
            "last_event": self.last_significant_event
        }
