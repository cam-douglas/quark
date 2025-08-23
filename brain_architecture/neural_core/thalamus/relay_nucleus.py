#!/usr/bin/env python3
"""
Thalamic Relay Nucleus ‑ Phase-1 Prototype
-----------------------------------------
Implements a minimal relay unit that forwards “sensory” packets to a
designated cortical target while optionally applying a simple gating
policy (on/off).  Based on principles discussed in Sherman & Guillery
(2006, *PNAS*), which emphasise single-relay nuclei routing as the core
function of first-order thalamus.

This module is intentionally lightweight; future work will incorporate
real synaptic dynamics and modulatory effects.

Inputs  : any Python object representing sensory data.
Outputs : the same object when gate is open, `None` when closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RelayStats:
    total_received: int = 0
    total_forwarded: int = 0
    gate_state: bool = True


class ThalamicRelay:
    """Minimal relay nucleus with binary gating."""

    def __init__(self, gate_open: bool = True):
        self._gate_open: bool = gate_open
        self._stats: RelayStats = RelayStats(gate_state=gate_open)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def receive(self, packet: Any) -> Optional[Any]:
        """Receive a *packet* and forward it if gate is open.

        Parameters
        ----------
        packet : Any
            Sensory or cortical message to relay.
        """
        self._stats.total_received += 1
        if self._gate_open:
            self._stats.total_forwarded += 1
            return packet
        return None

    def open_gate(self) -> None:  # noqa: D401 (imperative ok)
        """Open the relay gate."""
        self._gate_open = True
        self._stats.gate_state = True

    def close_gate(self) -> None:  # noqa: D401
        """Close the relay gate."""
        self._gate_open = False
        self._stats.gate_state = False

    # ------------------------------------------------------------------
    def stats(self) -> RelayStats:  # noqa: D401
        """Return cumulative statistics for debugging/metrics."""
        return self._stats


__all__ = ["ThalamicRelay", "RelayStats"]
