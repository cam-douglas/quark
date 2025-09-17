"""RL agents and learning algorithms for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_db import PrioritizedReplayBuffer, ReplayBuffer, GaussianActor, QCritic

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .graph_db import PrioritizedReplayBuffer, ReplayBuffer, GaussianActor, QCritic

    class SACMPOAgent:
        """SAC agent with MPO-style KL regularization."""
        def __init__(self, state_dim: int, action_dim: int, max_action: float,
                     console=None, tau: float = 0.005, use_per: bool = True, device=None):
            self.state_dim = int(state_dim)
            self.action_dim = int(action_dim)
            self.max_action = float(max_action)
            self.console = console
            self.tau = float(tau)
            self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

            # Actor networks
            self.actor = GaussianActor(state_dim, action_dim, max_action).to(self.device)
            self.actor_old = GaussianActor(state_dim, action_dim, max_action).to(self.device)
            self.actor_old.load_state_dict(self.actor.state_dict())

            # Critic networks
            self.critics = nn.ModuleList([QCritic(state_dim, action_dim).to(self.device) for _ in range(4)])
            self.critics_target = nn.ModuleList([QCritic(state_dim, action_dim).to(self.device) for _ in range(4)])
            for i in range(4):
                self.critics_target[i].load_state_dict(self.critics[i].state_dict())

            self.active_critics = 2

            # Optimizers
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic_opts = [torch.optim.Adam(self.critics[i].parameters(), lr=3e-4) for i in range(4)]
            self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

            self.alpha_min, self.alpha_max = 1e-4, 1.0
            self.replay = PrioritizedReplayBuffer(state_dim, action_dim, max_size=int(2e5)) if use_per else ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
            self._train_steps = 0
            self.batch_size = 256
            self.gamma = 0.99
            self.bh_pressure = 0.0
            self.kl_beta = 0.01

        def set_active_critics(self, n: int):
            """Set number of active critic networks."""
            self.active_critics = int(max(1, min(4, n)))

        @property
        def alpha(self):
            """Get current entropy regularization coefficient."""
            a = float(self.log_alpha.exp().item())
            return float(max(self.alpha_min, min(self.alpha_max, a)))

        def _target_entropy(self):
            """Calculate target entropy based on black hole pressure."""
            bh = float(max(0.0, min(1.5, self.bh_pressure)))
            base = -float(self.action_dim) * 0.60
            scale = 0.60 + 0.25 * bh
            return float(base * scale)

        def select_action(self, state, deterministic: bool = False):
            """Select action from current policy."""
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    mu, _ = self.actor.forward(s)
                    a = torch.tanh(mu) * self.max_action
                else:
                    a, _, _ = self.actor.sample(s)
            return a.squeeze(0).cpu().numpy().astype("float32")

        def store(self, state, action, next_state, reward, done):
            """Store experience in replay buffer."""
            self.replay.add(state, action, next_state, reward, done)

        def epistemic_std(self, state, action) -> float:
            """Calculate epistemic uncertainty using critic ensemble."""
            try:
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                a = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
                qs = []
                with torch.no_grad():
                    for i in range(self.active_critics):
                        qs.append(self.critics[i](s, a).cpu().item())
                if len(qs) <= 1:
                    return 0.0
                return float(np.std(np.array(qs)))
            except Exception:
                return 0.0

        def _soft_update(self, net, target):
            """Soft update target network."""
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        def update(self):
            """Update agent networks."""
            if self.replay.size < max(1024, self.batch_size):
                return

            use_per = isinstance(self.replay, PrioritizedReplayBuffer)
            batch = self.replay.sample(self.batch_size)

            if use_per:
                state_np, action_np, next_state_np, reward_np, done_np, weights_np, indices = batch
                weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device)
            else:
                state_np, action_np, next_state_np, reward_np, done_np = batch
                weights, indices = None, None

            state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
            done = torch.tensor(done_np, dtype=torch.float32, device=self.device)

            # Critic update
            with torch.no_grad():
                next_a, next_logp, _ = self.actor.sample(next_state)
                q_next = []
                for i in range(self.active_critics):
                    q_next.append(self.critics_target[i](next_state, next_a))
                q_next = torch.min(torch.stack(q_next, dim=0), dim=0).values
                target_v = q_next - self.log_alpha.exp() * next_logp
                target_q = reward + (1.0 - done) * self.gamma * target_v

            td_errors_for_buffer = []
            for i in range(self.active_critics):
                qi = self.critics[i](state, action)
                if i == 0 and use_per:  # Calculate TD errors once for buffer update
                    td_errors = torch.abs(qi - target_q).detach().cpu().numpy().flatten()
                    td_errors_for_buffer = td_errors

                if use_per and weights is not None:
                    li = (torch.nn.functional.mse_loss(qi, target_q, reduction='none') * weights).mean()
                else:
                    li = torch.nn.functional.mse_loss(qi, target_q)

                self.critic_opts[i].zero_grad()
                li.backward()
                self.critic_opts[i].step()

            if use_per and len(td_errors_for_buffer) > 0:
                self.replay.update_priorities(indices, td_errors_for_buffer)

            # Actor update
            a, logp, _ = self.actor.sample(state)
            q_pi = []
            for i in range(self.active_critics):
                q_pi.append(self.critics[i](state, a))
            q_pi = torch.min(torch.stack(q_pi, dim=0), dim=0).values

            # KL regularization (MPO-style)
            with torch.no_grad():
                mu_old, logstd_old = self.actor_old.forward(state)
            mu_new, logstd_new = self.actor.forward(state)
            kl = 0.5 * (
                (logstd_old.exp().pow(2) + (mu_old - mu_new).pow(2)) / (logstd_new.exp().pow(2) + 1e-8)
                + 2*(logstd_new - logstd_old) - 1.0
            ).sum(dim=1, keepdim=True).mean()

            actor_loss = (self.log_alpha.exp() * logp - q_pi).mean() + self.kl_beta * kl
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Alpha update
            target_ent = self._target_entropy()
            alpha_loss = -(self.log_alpha * (logp.detach() + target_ent)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # Soft updates
            for i in range(self.active_critics):
                self._soft_update(self.critics[i], self.critics_target[i])
            self._soft_update(self.actor, self.actor_old)

else:
    # Placeholder when torch not available
    class SACMPOAgent:
        def __init__(self, *args, **kwargs):
            pass

        def select_action(self, state, deterministic=False):
            return np.zeros(3, dtype=np.float32)

        def store(self, *args):
            pass

        def update(self):
            pass

        def epistemic_std(self, state, action):
            return 0.0
