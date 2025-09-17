#!/usr/bin/env python3
"""Proto-Cortex Layer Sheet – Self-Organizing Map (SOM)
Performs unsupervised learning on sensory input streams to create a topological map.
This is a foundational step for forming internal representations of the world.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import Tuple

class LayerSheet:
    """
    A Self-Organizing Map (SOM) that learns to represent high-dimensional
    sensory data in a low-dimensional topological map.
    """
    def __init__(self, map_size: Tuple[int, int] = (10, 10), input_dim: int = 139,
                 learning_rate: float = 0.5, sigma: float = None, rng: np.random.Generator = None):
        """
        Initializes the SOM.
        Args:
            map_size: The (width, height) of the 2D neuron grid.
            input_dim: The dimensionality of the sensory input vectors.
            learning_rate: The initial learning rate for weight updates.
            sigma: The initial radius of the neighborhood function. Defaults to map_width / 2.0.
            rng: A numpy random number generator.
        """
        self.map_width, self.map_height = map_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma or (self.map_width / 2.0)
        self.rng = rng or np.random.default_rng()

        # Initialize the weights randomly
        self.weights = self.rng.random((self.map_width, self.map_height, self.input_dim))

        # Grid of neuron coordinates
        self.neuron_grid = np.array([[i, j] for i in range(self.map_width) for j in range(self.map_height)])

        self.activity = np.zeros(map_size)

        print(f"🧠 Proto-Cortex (SOM) initialized with a {map_size} map for {input_dim}-D input.")

    def _find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """Finds the Best Matching Unit (BMU) for a given input vector."""
        # Calculate the Euclidean distance between the input and all neuron weights
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        # Find the coordinates of the neuron with the minimum distance
        bmu_coords = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        return bmu_coords

    def _update_weights(self, bmu_coords: Tuple[int, int], input_vector: np.ndarray, current_iteration: int, max_iterations: int):
        """Updates the weights of the BMU and its neighbors."""
        # Decay learning rate and sigma over time
        lr = self.learning_rate * (1.0 - (current_iteration / max_iterations))
        sig = self.sigma * (1.0 - (current_iteration / max_iterations))

        # Calculate the distance of each neuron from the BMU
        bmu_loc = np.array([bmu_coords])
        distances_from_bmu = np.linalg.norm(self.neuron_grid.reshape(self.map_width, self.map_height, 2) - bmu_loc, axis=2)

        # Calculate the neighborhood function (Gaussian)
        neighborhood_func = np.exp(-distances_from_bmu**2 / (2 * sig**2))

        # Calculate the weight update
        delta = neighborhood_func[:, :, np.newaxis] * lr * (input_vector - self.weights)

        # Apply the update
        self.weights += delta

    def step(self, input_vector: np.ndarray, current_iteration: int, max_iterations: int):
        """
        Processes one sensory input vector and trains the map.
        Args:
            input_vector: The sensory data to process.
            current_iteration: The current training step, for decaying learning rate/sigma.
            max_iterations: The total number of training steps.
        Returns:
            The coordinates of the activated neuron (BMU).
        """
        if input_vector.shape[0] != self.input_dim:
            # Pad or truncate the input vector to match the expected dimension
            input_vector = np.resize(input_vector, self.input_dim)

        # 1. Find the Best Matching Unit (BMU)
        bmu_coords = self._find_bmu(input_vector)

        # 2. Update the weights of the BMU and its neighbors
        self._update_weights(bmu_coords, input_vector, current_iteration, max_iterations)

        # Update the activity map for visualization/downstream processing
        self.activity = np.exp(-np.linalg.norm(self.neuron_grid.reshape(self.map_width, self.map_height, 2) - np.array([bmu_coords]), axis=2)**2 / (2 * 1**2))

        return {"bmu_coordinates": bmu_coords, "activity_map": self.activity}

    def mean_activity(self) -> float:
        """Returns the mean activation of the map."""
        return float(self.activity.mean())
