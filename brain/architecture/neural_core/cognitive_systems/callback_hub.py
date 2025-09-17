"""Callback Hub – simple publish/subscribe utility for Quark.

Brain modules can register listeners to be notified of resource events
without tight coupling to ResourceManager.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Any
import threading

_EventListener = Callable[[str, Dict[str, Any]], None]

class CallbackHub:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._listeners: List[_EventListener] = []
        return cls._instance

    # ---------------------------------------------------------
    def register(self, listener: _EventListener) -> None:
        """Register a listener callable(event_name, data_dict)."""
        self._listeners.append(listener)

    def emit(self, event: str, **data):
        for fn in list(self._listeners):
            try:
                fn(event, data)
            except Exception as e:
                # listeners should handle their own errors; just continue
                import logging
                logging.getLogger("quark.callback").warning("Listener error: %s", e)

# Global singleton accessor
hub = CallbackHub()
