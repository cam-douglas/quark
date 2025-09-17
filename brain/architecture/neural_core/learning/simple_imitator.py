

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from collections import defaultdict

class SimpleImitator:
    """
    A simplified Q-Learning agent specifically for the imitation task.
    It learns a policy to minimize pose error.
    """
    def __init__(self, num_actions: int, learning_rate=0.3, discount_factor=0.9, exploration_rate=1.0):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = 0.9995  # Slower decay to maintain exploration longer
        self.epsilon_min = 0.15     # Higher minimum to ensure continued exploration
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.error_bins = [1.0, 2.0, 2.3, 2.6, 3.0]  # Bins adjusted to actual error ranges observed (2.2-2.8)

    def _discretize_error(self, error: float) -> int:
        """Converts continuous pose error into a discrete state."""
        if error is None:
            return len(self.error_bins)
        return np.digitize(error, bins=self.error_bins)

    def choose_action(self, pose_error: float) -> int:
        """Choose an action using an epsilon-greedy policy with exploration bonus."""
        state = self._discretize_error(pose_error)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Add exploration bonus for states with low visit counts
        q_values_with_bonus = self.q_table[state].copy()

        # Simple exploration bonus: encourage actions that haven't been tried much
        for action in range(self.num_actions):
            if self.q_table[state][action] == 0:  # Unvisited action
                q_values_with_bonus[action] += 0.1  # Small bonus

        return np.argmax(q_values_with_bonus)

    def learn(self, last_error: float, action: int, reward: float, current_error: float, done: bool):
        """Update the Q-table based on the transition."""
        old_state = self._discretize_error(last_error)
        new_state = self._discretize_error(current_error)

        old_q = self.q_table[old_state][action]
        next_max = np.max(self.q_table[new_state]) if not done else 0

        new_q = old_q + self.lr * (reward + self.gamma * next_max - old_q)
        self.q_table[old_state][action] = new_q

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
