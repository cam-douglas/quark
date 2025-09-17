

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
# brain_modules/basal_ganglia/rl_agent.py

"""
Purpose: Implements a Reinforcement Learning agent for action selection.
Inputs: State, Reward
Outputs: Action
Dependencies: numpy
"""

import numpy as np

class QLearningAgent:
    """
    A simple Q-learning agent that learns to select actions
    based on rewards in a given environment.
    """
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_rate=0.99, min_exploration_rate=0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy strategy.
        """
        if np.random.random() < self.exploration_rate:
            # Explore: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploit: choose the best action from Q-table
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the agent's experience.
        """
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])

        # Q-learning formula
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value

    def decay_exploration(self):
        """
        Decays the exploration rate to reduce random actions over time.
        """
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

    def inject_knowledge(self, knowledge: "KnowledgeObject"):
        """
        Injects procedural knowledge from a dataset to pre-populate the Q-table.
        This is a form of imitation learning.
        """
        # We expect the content to be a dataset of (state, action, reward) tuples
        dataset = knowledge.content.get("dataset", [])

        for state, action, reward in dataset:
            if state < self.num_states and action < self.num_actions:
                # We update the Q-table with the "expert" knowledge.
                # We'll give it a strong update, but not absolute, to allow for further learning.
                current_value = self.q_table[state, action]
                self.q_table[state, action] = max(current_value, reward) # Set to the expert reward if it's better
