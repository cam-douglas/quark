# brain_architecture/neural_core/motor_control/basal_ganglia/tactical_agent.py
"""
Purpose: Implements a Reinforcement Learning agent specialized for rapid,
         flexible adaptation to changing environments.
Inputs: State, Reward
Outputs: Action
Dependencies: numpy
"""

import numpy as np

class TacticalAgent:
    """
    A Q-learning agent optimized for tasks requiring rapid adaptation,
    exploration, and responsiveness to immediate feedback.
    """
    def __init__(self, num_states, num_actions):
        # Parameters are hard-coded for tactical thinking
        self.learning_rate = 0.2  # Higher learning rate for faster adaptation
        self.discount_factor = 0.8  # Lower discount factor focuses on more immediate rewards
        self.exploration_rate = 1.0
        self.exploration_decay_rate = 0.99 # Faster decay to quickly exploit new patterns
        self.min_exploration_rate = 0.1 # Higher minimum exploration to stay adaptable

        self.num_states = num_states
        self.num_actions = num_actions
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
    agent = TacticalAgent(num_states, num_actions)

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
