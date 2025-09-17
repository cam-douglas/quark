"""PersistenceManager saves/loads snapshots for multiple IMemoryStore instances."""
from __future__ import annotations

import os
import atexit
import logging
from typing import Dict
from contextlib import nullcontext

from brain.architecture.neural_core.memory.imemory_store import IMemoryStore

_STATE_DIR = os.path.join("state", "memory")

logger = logging.getLogger(__name__)


class MemoryPersistenceManager:
    def __init__(self, stores: Dict[str, IMemoryStore], auto_register_atexit: bool = True):
        self.stores = stores
        os.makedirs(_STATE_DIR, exist_ok=True)
        if auto_register_atexit:
            atexit.register(self.save_all)

    # ------------------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(_STATE_DIR, f"{name}.json.gz")

    def save_all(self):
        from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
        rm = ResourceManager.get_default()
        ctx = rm.request_resources("io") if rm else nullcontext()

        with ctx:
            for name, store in self.stores.items():
                try:
                    store.save(self._path(name))
                except Exception as e:
                    logger.warning(f"[PersistenceManager] Failed saving {name}: {e}")

    def load_all(self):
        from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
        rm = ResourceManager.get_default()
        ctx = rm.request_resources("io") if rm else nullcontext()

        with ctx:
            for name, store in self.stores.items():
                try:
                    store.load(self._path(name))
                except Exception as e:
                    logger.warning(f"[PersistenceManager] Failed loading {name}: {e}")
