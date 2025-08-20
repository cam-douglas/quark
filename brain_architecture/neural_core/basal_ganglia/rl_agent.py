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

if __name__ == '__main__':
    # Example usage
    num_states = 10
    num_actions = 4
    agent = QLearningAgent(num_states, num_actions)

    # Simulate some learning steps
    state = 0
    for i in range(1000):
        action = agent.choose_action(state)
        reward = np.random.rand() # Simulate a random reward
        next_state = (state + np.random.choice(num_actions) -1) % num_states # Simulate a next state
        agent.learn(state, action, reward, next_state)
        agent.decay_exploration()
        state = next_state

    print("Q-table after 1000 steps:")
    print(agent.q_table)
    print(f"Final exploration rate: {agent.exploration_rate}")
