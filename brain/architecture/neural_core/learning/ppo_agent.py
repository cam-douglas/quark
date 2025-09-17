

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple, List


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RunningNorm:
    """Tracks running mean/variance to normalize observations and rewards."""
    def __init__(self, shape: int, epsilon: float = 1e-8):
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.eps = epsilon

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        new_mean = x.mean(axis=0)
        new_var = x.var(axis=0)
        if self.count == 0:
            self.mean = new_mean
            self.var = new_var
            self.count = n
            return
        total = self.count + n
        delta = new_mean - self.mean
        m_a = self.var * self.count
        m_b = new_var * n
        M2 = m_a + m_b + delta * delta * (self.count * n / total)
        self.mean = self.mean + delta * (n / total)
        self.var = M2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    logprob: float
    value: float
    reward: float
    done: bool


class PPOAgent:
    """Minimal PPO with GAE for discrete actions.

    Usage pattern in a step-based loop:
    - call select_action(obs) -> (action, logprob, value)
    - after obtaining next reward/done, call store_transition with the PREVIOUS triple
    - periodically call train_if_ready()
    """
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        rollout_len: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        lr: float = 3e-4,
        update_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.device = _get_device()
        self.model = ActorCritic(obs_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_len = rollout_len

        self.transitions: List[Transition] = []
        self.obs_norm = RunningNorm(obs_dim)
        self.reward_norm = RunningNorm(1)

        # cached previous step
        self.prev_obs = None
        self.prev_action = None
        self.prev_logprob = None
        self.prev_value = None

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.obs_norm.update(obs)
        obs_n = self.obs_norm.normalize(obs)
        obs_t = self._to_tensor(obs_n)
        logits, value = self.model(obs_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def store_transition(self, obs: np.ndarray, action: int, logprob: float, value: float, reward: float, done: bool):
        self.reward_norm.update(np.array([reward], dtype=np.float32))
        r = float(self.reward_norm.normalize(np.array([reward], dtype=np.float32))[0])
        self.transitions.append(Transition(obs=obs.copy(), action=action, logprob=logprob, value=value, reward=r, done=done))

    def _compute_gae(self, values: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            nextnonterminal = 1.0 - float(dones[t]) if t < len(dones) - 1 else 0.0
            nextvalue = values[t + 1] if t < len(values) - 1 else 0.0
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values[:len(advantages)]
        return advantages, returns

    def train_if_ready(self):
        if len(self.transitions) < self.rollout_len:
            return

        # Prepare tensors
        obs = np.stack([t.obs for t in self.transitions], axis=0)
        actions = np.array([t.action for t in self.transitions], dtype=np.int64)
        old_logprobs = np.array([t.logprob for t in self.transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        dones = np.array([t.done for t in self.transitions], dtype=np.bool_)

        with torch.no_grad():
            logits, values = self.model(self._to_tensor(obs))
            values = values.cpu().numpy().astype(np.float32)

        advantages, returns = self._compute_gae(values, rewards, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = self._to_tensor(obs)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_logprobs_t = torch.as_tensor(old_logprobs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        num_samples = obs.shape[0]
        idxs = np.arange(num_samples)

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = idxs[start:end]

                logits, value = self.model(obs_t[mb_idx])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logprob = dist.log_prob(actions_t[mb_idx])
                ratio = torch.exp(logprob - old_logprobs_t[mb_idx])

                surr1 = ratio * advantages_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(value, returns_t[mb_idx])
                entropy_loss = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self.transitions.clear()


