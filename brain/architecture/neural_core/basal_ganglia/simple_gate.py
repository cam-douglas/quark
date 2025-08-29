#!/usr/bin/env python3
"""Basal Ganglia Action Gate – Phase-1 Prototype

Implements a *focused selection* mechanism (Mink 1996) where multiple
candidate action channels compete.  The gate forwards only the channel
with the highest salience above a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class GateResult:
    selected_channel: Optional[str]
    salience_map: Dict[str, float]


class ActionGate:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def select(self, salience: Dict[str, float]) -> GateResult:
        """Return the channel with highest salience above threshold."""
        if not salience:
            return GateResult(None, salience)

        ch, val = max(salience.items(), key=lambda kv: kv[1])
        if val >= self.threshold:
            return GateResult(ch, salience)
        return GateResult(None, salience)


__all__ = ["ActionGate", "GateResult"]
