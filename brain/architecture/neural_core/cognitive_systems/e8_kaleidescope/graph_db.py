"""Graph database and memory structures for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Optional dependencies
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class GraphDB:
    """A graph database wrapper around NetworkX for managing conceptual relationships."""
    def __init__(self):
        if nx is None:
            raise ImportError("networkx library is required for GraphDB.")
        self.graph = nx.Graph()

    def add_node(self, node_id: str, **attrs):
        """Adds a node to the graph with the given attributes."""
        self.graph.add_node(node_id, **attrs)

    def add_edge(self, source_id: str, target_id: str, **attrs):
        """Adds an edge between two nodes with the given attributes."""
        self.graph.add_edge(source_id, target_id, **attrs)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a node's data."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None

    def get_neighbors(self, node_id: str) -> List[str]:
        """Gets the neighbors of a node."""
        if self.graph.has_node(node_id):
            return list(self.graph.neighbors(node_id))
        return []

class ReplayBuffer:
    """Simple FIFO replay buffer for RL agents."""
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e5)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        """Add experience to buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch from buffer."""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            np.asarray(self.state[ind]),
            np.asarray(self.action[ind]),
            np.asarray(self.next_state[ind]),
            np.asarray(self.reward[ind]),
            np.asarray(self.done[ind]),
        )

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e5),
                 alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.frame = 1
        self.eps = 1e-6
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.priorities = np.zeros((self.max_size,), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        """Add experience with priority."""
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.priorities[self.ptr] = max_prio
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """Sample batch with importance sampling."""
        if self.size == self.max_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.size]

        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)

        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        self.frame += 1
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.done[indices],
            weights.reshape(-1, 1).astype(np.float32),
            indices
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.eps

if TORCH_AVAILABLE:
    class GaussianActor(nn.Module):
        """Gaussian policy actor for continuous control."""
        def __init__(self, state_dim: int, action_dim: int, max_action: float):
            super().__init__()
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.mu = nn.Linear(256, action_dim)
            self.log_std = nn.Linear(256, action_dim)
            self.max_action = float(max_action)

        def forward(self, state):
            h = torch.relu(self.l1(state))
            h = torch.relu(self.l2(h))
            mu = self.mu(h)
            log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
            return mu, log_std

        def sample(self, state):
            mu, log_std = self.forward(state)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.max_action
            log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            mu_action = torch.tanh(mu) * self.max_action
            return action, log_prob, mu_action

    class QCritic(nn.Module):
        """Q-function critic for SAC."""
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            self.l1 = nn.Linear(state_dim + action_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, 1)

        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            x = torch.relu(self.l1(x))
            x = torch.relu(self.l2(x))
            return self.l3(x)

else:
    # Placeholder classes when torch not available
    class GaussianActor:
        def __init__(self, *args, **kwargs):
            pass

    class QCritic:
        def __init__(self, *args, **kwargs):
            pass
