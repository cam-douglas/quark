#!/usr/bin/env python3
"""Proto-Cortex Layer Sheet – Phase-1 Prototype

Simulates a simple 2-D layer of excitatory units with a homeostatic
plasticity rule (Turrigiano & Nelson 2004).  Each unit keeps an average
activity trace *m*; synaptic scaling factor *s* is adjusted so that *m*
approaches a target firing rate *m_target*.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HomeoParams:
    m_target: float = 0.1   # target mean activity
    eta: float = 0.01       # learning rate for scaling


class LayerSheet:
    def __init__(self, n: int = 100, params: HomeoParams | None = None, rng: np.random.Generator | None = None):
        self.n = n
        self.rng = rng or np.random.default_rng()
        self.params = params or HomeoParams()

        self.activity = self.rng.random(n) * 0.05  # low initial activity
        self.scaling = np.ones(n)

    def step(self, external_drive: np.ndarray | None = None) -> None:
        """One timestep update with optional external input."""
        if external_drive is None:
            external_drive = self.rng.random(self.n) * 0.1

        self.activity = np.tanh(self.scaling * (self.activity + external_drive))

        # homeostatic scaling
        delta = self.params.eta * (self.params.m_target - self.activity.mean())
        self.scaling += delta

    def mean_activity(self) -> float:  # noqa: D401
        return float(self.activity.mean())


__all__ = ["LayerSheet", "HomeoParams"]
