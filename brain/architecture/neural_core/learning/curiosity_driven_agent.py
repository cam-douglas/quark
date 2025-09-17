"""Curiosity-Driven Reinforcement Learning Agent
An agent that is intrinsically motivated to explore novel states and actions.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from collections import defaultdict
import logging
from typing import Tuple
from brain.architecture.neural_core.learning.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)

class CuriosityDrivenAgent:
    """
    A Q-learning agent that uses a LongTermMemory module to generate an
    intrinsic curiosity reward, encouraging exploration of novel experiences.
    """
    def __init__(
        self,
        num_actions: int,
        long_term_memory: LongTermMemory,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        initial_exploration_rate: float = 1.0,
        exploration_decay: float = 0.999,
        min_exploration_rate: float = 0.01,
        curiosity_weight: float = 0.5  # How much curiosity influences decisions
    ):
        self.num_actions = num_actions
        self.long_term_memory = long_term_memory
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.curiosity_weight = curiosity_weight

        # Q-table stores the expected future rewards for state-action pairs
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        print(f"🔬 Curiosity-Driven Agent initialized with curiosity weight: {self.curiosity_weight}")

    def _discretize_state(self, qpos: np.ndarray) -> Tuple:
        """
        Discretizes the state, focusing only on joint positions relevant to balance.
        """
        # We only need the joint positions (qpos), rounded to one decimal place.
        # Vision input is ignored for the balancing task.
        discretized_qpos = tuple(np.round(qpos, 1))
        return discretized_qpos

    def choose_action(self, qpos: np.ndarray) -> int:
        """
        Chooses an action based on the proprioceptive state.
        """
        state_tuple = self._discretize_state(qpos)

        # Add a curiosity bonus based on how novel the current state is.
        # This encourages the agent to explore different body configurations.
        curiosity_bonus = self.curiosity_weight * self.long_term_memory.get_novelty_score(state_tuple)

        q_values = self.q_table.get(state_tuple, np.zeros(self.num_actions))
        augmented_q_values = q_values + curiosity_bonus

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(augmented_q_values)

    def reset_q_table(self):
        """
        Resets the Q-table to "unlearn" previously established habits.
        This is crucial when the agent's goal or environment changes,
        allowing it to re-explore and find new optimal strategies.
        The long-term memory of visited states is preserved for curiosity.
        """
        old_size = len(self.q_table)
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
        logger.info(f"Q-table has been reset. Wiped {old_size} learned states. Ready to learn with a new purpose.")

    def learn(self, last_qpos: np.ndarray, last_action: int, reward: float, current_qpos: np.ndarray):
        """
        Updates the Q-table based on an experience (proprioceptive state).
        """
        last_state = self._discretize_state(last_qpos)
        current_state = self._discretize_state(current_qpos)

        old_q_value = self.q_table.get(last_state, np.zeros(self.num_actions))[last_action]
        future_q_value = np.max(self.q_table.get(current_state, np.zeros(self.num_actions)))

        new_q_value = old_q_value + self.alpha * (reward + self.gamma * future_q_value - old_q_value)

        # Update the Q-table
        if last_state not in self.q_table:
            self.q_table[last_state] = np.zeros(self.num_actions)
        self.q_table[last_state][last_action] = new_q_value

    def update_exploration_rate(self):
        """Decays the exploration rate to shift from exploration to exploitation over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def step(self, qpos: np.ndarray) -> int:
        """
        The main entry point for the agent to take a step.
        """
        state_tuple = self._discretize_state(qpos)
        action = self.choose_action(qpos)

        self.long_term_memory.record_experience(state_tuple, action)

        return action
