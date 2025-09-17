"""World Model - Phase 3 Prototype
A simple predictive model of the environment to detect novelty.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import numpy as np
from typing import Dict, Any

class SimpleWorldModel:
    """
    A basic world model that learns to predict the next state based on
    the current state and action. The prediction error is used as a
    novelty signal.
    """
    def __init__(self, num_states: int, num_actions: int):
        """
        Initializes the world model.
        Args:
            num_states: The number of possible states in the environment.
            num_actions: The number of possible actions.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        # A simple transition table to learn the environment's dynamics
        # T[state, action] -> predicted_next_state
        self.transition_model = np.zeros((num_states, num_actions))
        self.learning_rate = 0.1

    def predict_next_state(self, state: int, action: int) -> int:
        """
        Predicts the next state given the current state and action.
        """
        return int(self.transition_model[state, action])

    def update(self, state: int, action: int, next_state: int) -> Dict[str, Any]:
        """
        Updates the internal model based on a new transition and calculates
        the prediction error (novelty).

        Args:
            state: The starting state.
            action: The action taken.
            next_state: The actual state that resulted.

        Returns:
            A dictionary containing the prediction error and the next state.
        """
        predicted_next_state = self.predict_next_state(state, action)
        error = next_state - predicted_next_state

        # Update the model
        self.transition_model[state, action] += self.learning_rate * error

        return {
            "prediction_error": float(np.abs(error)),
            "next_state": next_state
        }
