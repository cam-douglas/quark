"""Global Workspace (minimal runtime shim)

Purpose: Provide a lightweight global workspace interface for the brain
simulator. This shim supports status queries and optional broadcast logging
without introducing heavy dependencies. It can be replaced by a richer
implementation from the consciousness stack when available.

Inputs: None (constructed empty; modules may call `broadcast` to add items)
Outputs: `get_broadcast()` returns a summary dictionary suitable for status
         reporting.

Seed/Deps: None; deterministic and side-effect free other than in-memory log.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from __future__ import annotations

from typing import Any, Dict, List
import time


class GlobalWorkspace:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = max(1, int(capacity))
        self._contents: List[Dict[str, Any]] = []
        self.active = True

    def broadcast(self, content: Dict[str, Any]) -> None:
        """Add an item to the workspace with a timestamp."""
        if not isinstance(content, dict):
            return
        item = {"ts": time.time(), **content}
        self._contents.append(item)
        if len(self._contents) > self.capacity:
            self._contents.pop(0)

    def get_broadcast(self) -> Dict[str, Any]:
        """Return a compact summary of recent workspace activity."""
        recent = self._contents[-5:]
        return {
            "active": self.active,
            "capacity": self.capacity,
            "num_items": len(self._contents),
            "recent": recent,
        }

