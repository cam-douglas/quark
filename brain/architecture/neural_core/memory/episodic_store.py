"""Adapter exposing EpisodicMemory via the IMemoryStore protocol."""
from __future__ import annotations

from typing import Dict, Any

from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory
from brain.architecture.neural_core.memory.imemory_store import IMemoryStore


class EpisodicMemoryStore(IMemoryStore):  # type: ignore[misc]
    """A thin shim that delegates to an EpisodicMemory instance.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

    SCHEMA_VERSION = 1

    def __init__(self, memory: EpisodicMemory):
        self.mem = memory

    # ------------------------------------------------------------------
    # IMemoryStore implementation
    # ------------------------------------------------------------------
    def store_episode(self, content: Dict[str, Any], context: Dict[str, Any]):
        return self.mem.store_episode(content=content, context=context)

    def retrieve_episode(self, query: Dict[str, Any], max_results: int = 5):
        return self.mem.retrieve_episode(query, max_results)

    def save(self, path: str):
        """Persist episodes to *path* atomically with checksum & schema."""
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
            "episodes": {eid: ep.__dict__ for eid, ep in self.mem.episodes.items()},
        }
        raw = json.dumps(payload).encode("utf-8")
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        wrapper = {"crc32": crc, "data": payload}

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
        """Load snapshot from *path*; silently returns if file missing."""
        import json
        import gzip
        import os
        import zlib
        import logging
        from brain.architecture.neural_core.hippocampus.episodic_memory import MemoryEpisode

        if not os.path.exists(path):
            return

        with gzip.open(path, "rt", encoding="utf-8") as f:
            wrapper = json.load(f)

        raw = json.dumps(wrapper["data"]).encode("utf-8")
        if (zlib.crc32(raw) & 0xFFFFFFFF) != wrapper.get("crc32"):
            logging.warning("[EpisodicMemoryStore] Checksum mismatch – snapshot ignored")
            return

        payload = wrapper["data"]
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            logging.warning("[EpisodicMemoryStore] Unsupported schema version – snapshot ignored")
            return

        for eid, fields in payload.get("episodes", {}).items():
            self.mem.episodes[eid] = MemoryEpisode(**fields)

    def get_stats(self):
        return self.mem.get_memory_stats()
