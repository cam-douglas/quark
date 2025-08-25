#!/usr/bin/env python3
"""
A simple Reinforcement Learning agent for Quark's embodied learning.
"""
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import StepLR

# Helper function to convert dict state to a tensor
def _state_to_tensor(state: dict) -> torch.Tensor:
    """Converts a state dictionary to a flattened PyTorch tensor."""
    state_vector = state['state_vector']
    return torch.FloatTensor(state_vector).unsqueeze(0)

class PolicyNetwork(nn.Module):
    """Actor Network: Decides which action to take."""
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std.exp()

class ValueNetwork(nn.Module):
    """Critic Network: Estimates the value of a given state."""
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.net(state)

class BalancingAgent:
    """
    An Actor-Critic agent that learns to balance the humanoid model.
    """
    def __init__(self, state_dim, action_dim, replay_buffer_size=10000, lr=3e-4, gamma=0.99):
        """
        Initializes the agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Add a learning rate scheduler
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=100, gamma=0.99)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=100, gamma=0.99)
        
        self.memory = deque(maxlen=replay_buffer_size)
        # For reward normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_buffer = deque(maxlen=1000)

    def get_action(self, state: dict) -> np.ndarray:
        """
        Given the current state, decide on an action.
        """
        state_tensor = _state_to_tensor(state)
        mean, std = self.actor(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.detach().numpy().flatten()

    def store_experience(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        self.reward_buffer.append(reward)

    def learn(self, batch_size=64):
        """
        Samples a batch of experiences from memory and updates the networks.
        """
        if len(self.memory) < batch_size:
            return # Not enough experiences to learn from yet

        # Update reward normalization stats
        if len(self.reward_buffer) > 1:
            self.reward_mean = np.mean(self.reward_buffer)
            self.reward_std = np.std(self.reward_buffer) + 1e-6 # Add epsilon to avoid division by zero

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Normalize rewards
        rewards = (np.array(rewards) - self.reward_mean) / self.reward_std

        # Convert to tensors
        states = torch.cat([_state_to_tensor(s) for s in states])
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.cat([_state_to_tensor(s) for s in next_states])
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # --- Update Critic ---
        state_values = self.critic(states)
        next_state_values = self.critic(next_states)
        # TD Target
        target_values = rewards + self.gamma * next_state_values * (1 - dones)
        
        critic_loss = nn.MSELoss()(state_values, target_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # --- Update Actor ---
        advantage = (target_values - state_values).detach()
        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions)
        
        actor_loss = -(log_probs * advantage).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Step the schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()
