#!/usr/bin/env python3
"""
Hippocampal STDP Synapse ‑ Phase-1 Prototype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implements a pair-based spike-timing–dependent plasticity (STDP) rule as
first characterised in Bi & Poo 1998 (Nature).  The weight update Δw is
an exponential function of the pre- minus post-spike timing difference
Δt.

Δw = A_plus  * exp(-|Δt| / tau_plus)   if Δt > 0  (LTP)
Δw = -A_minus * exp(-|Δt| / tau_minus) if Δt < 0  (LTD)

This stub keeps only the latest spike times for pre- and post-neurons
and updates the synaptic weight whenever either neuron fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class STDPParams:
    A_plus: float = 0.01
    A_minus: float = 0.012
    tau_plus: float = 20.0   # ms
    tau_minus: float = 20.0  # ms
    w_min: float = 0.0
    w_max: float = 1.0


class STDPSynapse:
    """Minimal pair-based STDP synapse between two neurons."""

    def __init__(self, weight: float = 0.5, params: STDPParams | None = None):
        self.w: float = weight
        self.params: STDPParams = params or STDPParams()
        self._t_pre: Union[float, None] = None  # ms
        self._t_post: Union[float, None] = None

    # ------------------------------------------------------------------
    def pre_spike(self, t_ms: float) -> None:
        """Call when presynaptic neuron fires at time *t_ms* (ms)."""
        self._t_pre = t_ms
        if self._t_post is not None:
            self._update_weight(t_ms - self._t_post)

    def post_spike(self, t_ms: float) -> None:
        """Call when postsynaptic neuron fires at time *t_ms* (ms)."""
        self._t_post = t_ms
        if self._t_pre is not None:
            self._update_weight(self._t_pre - t_ms)

    # ------------------------------------------------------------------
    def _update_weight(self, delta_t_ms: float) -> None:
        p = self.params
        # Bi & Poo 1998: LTP when presynaptic fires *before* postsynaptic (Δt < 0)
        if delta_t_ms < 0:  # LTP
            dw = p.A_plus * (2.71828 ** (delta_t_ms / p.tau_plus))
        else:  # LTD
            dw = -p.A_minus * (2.71828 ** (-delta_t_ms / p.tau_minus))

        self.w = min(p.w_max, max(p.w_min, self.w + dw))

    # ------------------------------------------------------------------
    def weight(self) -> float:  # noqa: D401
        """Current synaptic weight."""
        return self.w


__all__ = ["STDPSynapse", "STDPParams"]
