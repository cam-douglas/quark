"""Adapter exposing LongTermMemory via the IMemoryStore protocol.

LongTermMemory stores state–action visit counts, which are different from
EpisodicMemory episodes. For synchronisation we approximate an “episode” as
(state_hash, action) tuple added via store_episode(). retrieve_episode returns
empty (not meaningful) but provided for API completeness.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from __future__ import annotations

from typing import Dict, Any

from brain.architecture.neural_core.learning.long_term_memory import LongTermMemory
from brain.architecture.neural_core.memory.imemory_store import IMemoryStore


class LongTermMemoryStore(IMemoryStore):  # type: ignore[misc]
    """Wrapper around LongTermMemory to satisfy IMemoryStore."""

    SCHEMA_VERSION = 1

    def __init__(self, ltm: LongTermMemory):
        self.ltm = ltm

    # Episodic concepts don’t map directly; we treat content as {"state": tuple, "action": int}
    def store_episode(self, content: Dict[str, Any], context: Dict[str, Any] | None = None):  # noqa: D401
        state = tuple(content.get("state", []))
        action = int(content.get("action", 0))
        self.ltm.record_experience(state, action)
        return f"ltm_{state}_{action}"

    def retrieve_episode(self, query: Dict[str, Any], max_results: int = 5):  # noqa: D401
        # Not meaningful; return empty list
        return []

    def save(self, path: str):
        """Persist long-term counts atomically with checksum & schema."""
        import json
        import gzip
        import time
        import zlib
        import os
        import tempfile
        import shutil

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": int(time.time()),
            "counts": {str(k): dict(v) for k, v in self.ltm.state_action_counts.items()},
            "total": self.ltm.total_experiences,
        }
        raw = json.dumps(payload).encode("utf-8")
        wrapper = {"crc32": zlib.crc32(raw) & 0xFFFFFFFF, "data": payload}

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(path) or ".")
        os.close(tmp_fd)
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(wrapper, f)
            shutil.move(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load(self, path: str):
        """Load snapshot; ignore if missing or checksum/schema mismatch."""
        import json
        import gzip
        import os
        import zlib
        import logging
        from collections import defaultdict

        if not os.path.exists(path):
            return

        with gzip.open(path, "rt", encoding="utf-8") as f:
            wrapper = json.load(f)

        raw = json.dumps(wrapper["data"]).encode("utf-8")
        if (zlib.crc32(raw) & 0xFFFFFFFF) != wrapper.get("crc32"):
            logging.warning("[LongTermMemoryStore] Checksum mismatch – snapshot ignored")
            return

        payload = wrapper["data"]
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            logging.warning("[LongTermMemoryStore] Unsupported schema version – snapshot ignored")
            return

        self.ltm.state_action_counts = defaultdict(lambda: defaultdict(int))
        for state_str, actions in payload.get("counts", {}).items():
            state = eval(state_str)
            for a_str, cnt in actions.items():
                self.ltm.state_action_counts[state][int(a_str)] = cnt
        self.ltm.total_experiences = int(payload.get("total", 0))

    def get_stats(self):
        return self.ltm.get_status()
