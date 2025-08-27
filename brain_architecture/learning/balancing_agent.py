#!/usr/bin/env python3
"""
A simple Reinforcement Learning agent for Quark's embodied learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random

def _state_to_tensor(state: dict) -> torch.Tensor:
    """Converts a state dictionary to a flattened PyTorch tensor."""
    if isinstance(state, dict) and 'state_vector' in state:
        state_vector = state['state_vector']
    elif isinstance(state, np.ndarray):
        state_vector = state
    else:
        raise TypeError(f"Unsupported state type: {type(state)}")
    return torch.FloatTensor(state_vector).unsqueeze(0)

class PolicyNetwork(nn.Module):
    """Actor Network: Decides which action to take."""
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

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
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class BalancingAgent:
    """
    A PPO agent that learns to balance the humanoid model.
    """
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, gae_lambda=0.95, ppo_clip=0.2, ppo_epochs=20, batch_size=256):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        
        self.memory = []
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        
        # Initialize weights for stability
        self.actor.apply(self._init_weights)
        self.critic.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def get_action(self, state: dict) -> (np.ndarray, float):
        state_tensor = _state_to_tensor(state)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
        
        # Add exploration noise
        action += self.noise()
        return action.numpy().flatten(), log_prob.item()

    def store_experience(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def learn(self):
        if not self.memory:
            return

        # Convert memory to tensors
        states = torch.cat([_state_to_tensor(s['state_vector']) for s, a, r, ns, d, lp in self.memory])
        actions = torch.FloatTensor(np.array([a for s, a, r, ns, d, lp in self.memory]))
        rewards = torch.FloatTensor([r for s, a, r, ns, d, lp in self.memory]).unsqueeze(1)
        next_states = torch.cat([_state_to_tensor(ns['state_vector']) for s, a, r, ns, d, lp in self.memory])
        dones = torch.FloatTensor([d for s, a, r, ns, d, lp in self.memory]).unsqueeze(1)
        old_log_probs = torch.FloatTensor([lp for s, a, r, ns, d, lp in self.memory]).unsqueeze(1)

        # Calculate advantages using GAE
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            for t in reversed(range(len(rewards))):
                advantages[t] = deltas[t] + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
                last_advantage = advantages[t]
        
        # Normalize advantages only if there's more than one value to prevent division by zero
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = (advantages - advantages.mean()) # No std deviation for a single value

        # PPO Update
        for _ in range(self.ppo_epochs):
            for i in range(0, len(self.memory), self.batch_size):
                batch_indices = slice(i, i + self.batch_size)
                
                # Get policy and value for the current batch
                mean, std = self.actor(states[batch_indices])
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions[batch_indices]).sum(dim=1, keepdim=True)
                
                state_values = self.critic(states[batch_indices])
                
                # Ratio of new to old probabilities
                ratio = (new_log_probs - old_log_probs[batch_indices]).exp()
                
                # Clipped surrogate objective
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages[batch_indices]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(state_values, (advantages + values)[batch_indices].detach())
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        # Clear memory after learning
        self.memory = []
