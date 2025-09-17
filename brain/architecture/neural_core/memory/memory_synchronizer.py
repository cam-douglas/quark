"""MemorySynchronizer keeps EpisodicMemory and LongTermMemory in agreement.

Current strategy:
1. For every new episodic episode not yet mirrored, increment visit count in
   long-term memory using a hash of episode content as pseudo-state and action 0.
2. Maintains an internal set of forwarded episode ids for idempotence.
3. Designed to be lightweight; call `sync()` each simulation step.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from __future__ import annotations

from typing import Set, Dict, Any
import hashlib

from brain.architecture.neural_core.memory.episodic_store import EpisodicMemoryStore
from brain.architecture.neural_core.memory.longterm_store import LongTermMemoryStore


class MemorySynchronizer:
    """Synchronises episodic and long-term memory stores."""

    def __init__(self, episodic: EpisodicMemoryStore, long_term: LongTermMemoryStore):
        self.epi = episodic
        self.ltm = long_term
        self._forwarded: Set[str] = set()

    # ------------------------------------------------------------------
    def _episode_to_state_action(self, episode) -> tuple:
        """Very simple hash projection of episode content → (state_tuple, action)."""
        text = str(episode.content)
        # 64-bit hash to tuple of two 32-bit ints for example state representation
        h = hashlib.md5(text.encode()).hexdigest()[:16]
        hi = int(h, 16)
        state_tuple = (hi >> 32, hi & 0xFFFFFFFF)
        return state_tuple, 0  # default action 0 for now

    # ------------------------------------------------------------------
    def sync(self) -> Dict[str, Any]:
        """Forward unseen episodes to long-term counts and return summary."""
        new_cnt = 0
        for eid, episode in self.epi.mem.episodes.items():
            if eid in self._forwarded:
                continue
            state, action = self._episode_to_state_action(episode)
            self.ltm.ltm.record_experience(state, action)
            self._forwarded.add(eid)
            new_cnt += 1
        return {
            "forwarded": new_cnt,
            "total_forwarded": len(self._forwarded),
            "ltm_total": self.ltm.ltm.total_experiences,
        }
