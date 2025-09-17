"""Basic Salience Network - Phase 2 Prototype
Detects novel or important stimuli to guide attention.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import numpy as np
from typing import Dict

class BasicSalienceNetwork:
    """
    A simple salience network that calculates the salience of stimuli
    based on novelty and intensity, producing attention weights.
    """
    def __init__(self, num_sources: int):
        """
        Initializes the salience network.
        Args:
            num_sources: The number of sensory input sources to monitor.
        """
        self.num_sources = num_sources
        self.last_inputs: Dict[int, float] = {i: 0.0 for i in range(num_sources)}
        self.attention_weights = np.ones(num_sources) / num_sources

    def step(self, inputs: Dict[int, float]) -> np.ndarray:
        """
        Processes one time step of salience calculation.
        Args:
            inputs: A dictionary mapping source_id to a single float value
                    representing the stimulus intensity.
        Returns:
            A numpy array of attention weights, normalized to sum to 1.
        """
        salience_scores = np.zeros(self.num_sources)

        for source_id, intensity in inputs.items():
            if source_id < self.num_sources:
                # Novelty is the absolute difference from the last input
                novelty = abs(intensity - self.last_inputs[source_id])

                # Salience is a combination of novelty and intensity
                salience = novelty + intensity
                salience_scores[source_id] = salience

                # Update last input for the next step
                self.last_inputs[source_id] = intensity

        # Normalize salience scores to create attention weights
        total_salience = np.sum(salience_scores)
        if total_salience > 0:
            self.attention_weights = salience_scores / total_salience
        else:
            # If no salience, maintain equal attention
            self.attention_weights = np.ones(self.num_sources) / self.num_sources

        return self.attention_weights

    def get_attention_weights(self) -> np.ndarray:
        """Returns the current attention weights."""
        return self.attention_weights
