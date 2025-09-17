

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
# brain_modules/bas-al_ganglia/gating_system.py

"""
Purpose: Integrates the Basal Ganglia components into a cohesive gating system.
Inputs: State
Outputs: Action
Dependencies: rl_agent, dopamine_system
"""

try:
    from brain.architecture.neural_core.motor_control.basal_ganglia.rl_agent import QLearningAgent
except Exception:
    from .rl_agent import QLearningAgent  # type: ignore
from .dopamine_system import DopamineSystem
from .architecture import BasalGanglia
import numpy as np

class GatingSystem:
    """
    Integrates the RL agent and dopamine system to simulate the action
    selection (gating) process of the Basal Ganglia.
    """
    def __init__(self, num_states, num_actions):
        self.architecture = BasalGanglia()
        self.agent = QLearningAgent(num_states, num_actions)
        self.dopamine_system = DopamineSystem()

        # Connect the dopamine system to the agent's learning rate
        # This is a simplification to model dopamine's effect on learning
        self.modulate_learning_rate()

    def choose_action(self, state):
        """
        Selects an action using the RL agent.
        """
        return self.agent.choose_action(state)

    def learn_from_feedback(self, state, action, actual_reward, next_state):
        """
        The agent learns from the outcome of its action, with the learning
        process being modulated by the dopamine system.
        """
        # The agent's Q-table provides the predicted reward for the action taken
        predicted_reward = self.agent.q_table[state, action]

        # The dopamine system calculates the RPE and updates dopamine levels
        dopamine_level, rpe = self.dopamine_system.process_reward(predicted_reward, actual_reward)

        # The RPE is used as the learning signal for the agent
        # Note: This is a direct use of RPE. In a more complex model,
        # the dopamine level itself would modulate synaptic plasticity.
        self.agent.learn(state, action, rpe, next_state)

        # Modulate learning based on dopamine levels
        self.modulate_learning_rate(dopamine_level)

        # Decay exploration to shift from exploration to exploitation
        self.agent.decay_exploration()

    def modulate_learning_rate(self, dopamine_level=None):
        """
        Adjusts the agent's learning rate based on the current dopamine level.
        Higher dopamine (positive surprise) increases learning.
        Lower dopamine (negative surprise) can also increase learning to adapt faster.
        """
        if dopamine_level is None:
            dopamine_level = self.dopamine_system.baseline_dopamine

        # Example modulation: learning rate is proportional to the deviation from baseline
        # This is a simple heuristic.
        deviation = abs(dopamine_level - self.dopamine_system.baseline_dopamine)
        self.agent.learning_rate = 0.1 + deviation # Base learning rate + modulation
        self.agent.learning_rate = max(0.01, min(0.5, self.agent.learning_rate))


if __name__ == '__main__':
    # --- Simulation Setup ---
    num_states = 5
    num_actions = 2 # e.g., 'press lever' vs 'do nothing'
    gating_system = GatingSystem(num_states, num_actions)

    # --- A Simple Learning Scenario ---
    # Imagine a simple environment with 5 states.
    # In state 2, action 0 gives a high reward. In other states, rewards are random.

    print("--- Starting Simulation ---")
    state = 0
    for episode in range(2000):
        action = gating_system.choose_action(state)

        # Simulate environment's response
        if state == 2 and action == 0:
            actual_reward = 1.0 # Consistent high reward
        else:
            actual_reward = np.random.choice([0, 0.1, 0.2]) # Random, low reward

        next_state = (state + 1) % num_states # Move to the next state cyclically

        gating_system.learn_from_feedback(state, action, actual_reward, next_state)

        state = next_state

        if (episode + 1) % 200 == 0:
            print(f"\n--- Episode {episode + 1} ---")
            print("Q-table:")
            # Round for cleaner display
            print(np.round(gating_system.agent.q_table, 2))
            print(f"Current Exploration Rate: {gating_system.agent.exploration_rate:.2f}")

    print("\n--- Simulation Finished ---")
    print("Final Q-table:")
    print(np.round(gating_system.agent.q_table, 2))
    print("\nNote how the value for (State 2, Action 0) is significantly higher,")
    print("indicating the agent has learned to prefer this action in that state.")
