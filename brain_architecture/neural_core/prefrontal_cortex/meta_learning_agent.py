# brain_architecture/neural_core/prefrontal_cortex/meta_learning_agent.py
"""
Purpose: Implements a meta-learning agent that learns to select the best
         cognitive strategy (e.g., which agent to use) for a given task.
Inputs: State (task properties), Reward (performance feedback)
Outputs: Action (selected agent/strategy)
Dependencies: numpy
"""

import numpy as np

class MetaLearningAgent:
    """
    A Q-learning agent that operates at a higher level of abstraction, learning
    to select between different sub-agents or cognitive strategies.
    """
    def __init__(self, num_states: int, num_strategies: int):
        # Parameters optimized for high-level, stable learning
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay_rate = 0.99
        self.min_exploration_rate = 0.1

        self.num_states = num_states
        self.num_strategies = num_strategies
        self.q_table = np.zeros((num_states, num_strategies))

    def choose_strategy(self, state: int) -> int:
        """
        Chooses a strategy (i.e., which agent to deploy) based on the
        current task state using an epsilon-greedy policy.
        """
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.num_strategies)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state: int, strategy: int, reward: float, next_state: int):
        """
        Updates the Q-table based on the performance of the chosen strategy.
        """
        old_value = self.q_table[state, strategy]
        next_max = np.max(self.q_table[next_state, :])

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, strategy] = new_value

    def decay_exploration(self):
        """Reduces the exploration rate over time."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
