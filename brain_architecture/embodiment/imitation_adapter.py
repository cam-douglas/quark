#!/usr/bin/env python3
"""
Imitation Policy Adapter

Loads a tiny supervised imitation policy (PyTorch) and produces 18-D joint
angle targets from a 24-D observation: [root_vel(3) + joint_vel(18) + pad(3)].
"""
from __future__ import annotations

from typing import Optional
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as _e:  # pragma: no cover
    torch = None
    nn = None


class TinyPolicy(nn.Module):
    def __init__(self, obs_dim: int = 24, act_dim: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class ImitationPolicyAdapter:
    def __init__(self, policy_path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch not available; cannot use imitation policy")
        self.device = torch.device("cpu")
        self.model = TinyPolicy().to(self.device)
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Imitation policy not found: {policy_path}")
        state = torch.load(policy_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        else:
            obs_t = torch.from_numpy(obs).float()
        with torch.no_grad():
            act = self.model(obs_t).cpu().numpy()
        return act[0] if obs.ndim == 1 else act


