# brain_modules/basal_ganglia/actor_critic.py

"""
Purpose: Actor-Critic architecture for more biologically plausible reinforcement learning.
Inputs: State observations, rewards
Outputs: Actions, value estimates
Dependencies: numpy
"""

import numpy as np
from typing import List, Tuple, Optional
import random

class Actor:
    """Policy network (actor) that outputs action probabilities"""
    def __init__(self, num_states: int, num_actions: int, learning_rate: float = 0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Policy parameters (state-action preferences)
        self.policy_params = np.random.randn(num_states, num_actions) * 0.01
        
        # Eligibility traces for temporal credit assignment
        self.eligibility_traces = np.zeros((num_states, num_actions))
        self.trace_decay = 0.9
        
    def get_action_probabilities(self, state: int) -> np.ndarray:
        """Get action probabilities for current state using softmax"""
        logits = self.policy_params[state, :]
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities
    
    def choose_action(self, state: int, exploration_rate: float = 0.1) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < exploration_rate:
            return random.randint(0, self.num_actions - 1)
        
        probabilities = self.get_action_probabilities(state)
        return np.random.choice(self.num_actions, p=probabilities)
    
    def update_policy(self, state: int, action: int, advantage: float):
        """Update policy using policy gradient with advantage"""
        probabilities = self.get_action_probabilities(state)
        
        # Policy gradient update
        for a in range(self.num_actions):
            if a == action:
                # Increase probability of chosen action
                self.policy_params[state, a] += self.learning_rate * advantage * (1 - probabilities[a])
            else:
                # Decrease probability of other actions
                self.policy_params[state, a] -= self.learning_rate * advantage * probabilities[a]
        
        # Update eligibility traces
        self.eligibility_traces[state, action] += 1
    
    def decay_eligibility_traces(self):
        """Decay eligibility traces"""
        self.eligibility_traces *= self.trace_decay

class Critic:
    """Value network (critic) that estimates state values"""
    def __init__(self, num_states: int, learning_rate: float = 0.01):
        self.num_states = num_states
        self.learning_rate = learning_rate
        
        # Value function parameters
        self.value_params = np.random.randn(num_states) * 0.01
        
        # Eligibility traces
        self.eligibility_traces = np.zeros(num_states)
        self.trace_decay = 0.9
    
    def get_value(self, state: int) -> float:
        """Get value estimate for current state"""
        return self.value_params[state]
    
    def update_value(self, state: int, target_value: float):
        """Update value function using TD learning"""
        current_value = self.get_value(state)
        td_error = target_value - current_value
        
        # Update value parameters
        self.value_params[state] += self.learning_rate * td_error
        
        # Update eligibility traces
        self.eligibility_traces[state] += 1
    
    def decay_eligibility_traces(self):
        """Decay eligibility traces"""
        self.eligibility_traces *= self.trace_decay

class ActorCriticAgent:
    """Actor-Critic agent combining policy and value learning"""
    def __init__(self, num_states: int, num_actions: int, 
                 actor_lr: float = 0.01, critic_lr: float = 0.01,
                 discount_factor: float = 0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        
        self.actor = Actor(num_states, num_actions, actor_lr)
        self.critic = Critic(num_states, critic_lr)
        
        self.current_state = None
        self.current_action = None
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
    
    def choose_action(self, state: int) -> int:
        """Choose action using current policy"""
        self.current_state = state
        action = self.actor.choose_action(state, self.exploration_rate)
        self.current_action = action
        return action
    
    def learn(self, next_state: int, reward: float, done: bool = False):
        """Learn from experience using actor-critic update"""
        if self.current_state is None or self.current_action is None:
            return
        
        # Get current and next state values
        current_value = self.critic.get_value(self.current_state)
        next_value = 0.0 if done else self.critic.get_value(next_state)
        
        # Calculate TD target and error
        td_target = reward + self.discount_factor * next_value
        td_error = td_target - current_value
        
        # Update critic (value function)
        self.critic.update_value(self.current_state, td_target)
        
        # Update actor (policy) using advantage
        advantage = td_error  # Simple advantage estimate
        self.actor.update_policy(self.current_state, self.current_action, advantage)
        
        # Decay eligibility traces
        self.actor.decay_eligibility_traces()
        self.critic.decay_eligibility_traces()
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)
    
    def get_policy_entropy(self, state: int) -> float:
        """Calculate policy entropy for exploration monitoring"""
        probabilities = self.actor.get_action_probabilities(state)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        return entropy
    
    def get_state_value(self, state: int) -> float:
        """Get value estimate for a state"""
        return self.critic.get_value(state)
    
    def get_action_probabilities(self, state: int) -> np.ndarray:
        """Get action probabilities for a state"""
        return self.actor.get_action_probabilities(state)

class ExperienceReplay:
    """Experience replay buffer for more stable learning"""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add_experience(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool):
        """Add experience to replay buffer"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, batch_size)
    
    def get_size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)

class PrioritizedExperienceReplay(ExperienceReplay):
    """Prioritized experience replay with importance sampling"""
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0
    
    def add_experience(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool, td_error: float = None):
        """Add experience with priority based on TD error"""
        experience = (state, action, reward, next_state, done)
        
        # Set priority based on TD error or max priority
        priority = abs(td_error) if td_error is not None else self.max_priority
        priority = max(priority, 1e-6)  # Minimum priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities[len(self.buffer) - 1] = priority
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)
    
    def sample_batch(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) < batch_size:
            return self.buffer, np.ones(len(self.buffer)), np.arange(len(self.buffer))
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        
        batch = [self.buffer[i] for i in indices]
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

if __name__ == '__main__':
    # Test the actor-critic system
    print("Testing Actor-Critic Architecture")
    
    # Create agent
    agent = ActorCriticAgent(num_states=5, num_actions=3)
    
    # Test action selection
    state = 0
    action = agent.choose_action(state)
    print(f"State {state} -> Action {action}")
    
    # Test learning
    next_state = 1
    reward = 1.0
    agent.learn(next_state, reward, done=False)
    
    # Test policy entropy
    entropy = agent.get_policy_entropy(state)
    print(f"Policy entropy: {entropy:.3f}")
    
    # Test experience replay
    replay = PrioritizedExperienceReplay(capacity=100)
    replay.add_experience(0, 1, 1.0, 1, False, td_error=0.5)
    batch, weights, indices = replay.sample_batch(1)
    print(f"Sampled batch size: {len(batch)}")
    
    print("Actor-Critic test completed.")
