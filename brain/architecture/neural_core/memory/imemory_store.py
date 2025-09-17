"""Common memory-store interface for Quark.

This protocol allows different memory back-ends (episodic, long-term counts,
vector DB, etc.) to be accessed via a shared minimal API so orchestration code
(synchronisers, persistence layers) can operate generically.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from __future__ import annotations

from typing import Protocol, List, Dict, Any


class IMemoryStore(Protocol):
    """A minimal interface implemented by memory components that hold episodes or statistics."""

    # ------------------------------------------------------------------
    # Episode‐level operations (optional for some stores)
    # ------------------------------------------------------------------
    def store_episode(self, content: Dict[str, Any], context: Dict[str, Any]) -> str:  # noqa: D401,E501
        """Add a new episode and return its id."""

    def retrieve_episode(self, query: Dict[str, Any], max_results: int = 5) -> List[Any]:
        """Return episodes relevant to *query* ordered by relevance."""

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the store to *path* (implementation-specific format)."""

    def load(self, path: str) -> None:
        """Load the store contents from *path*."""

    # ------------------------------------------------------------------
    # Stats / metadata
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return lightweight diagnostic info (counts, utilisation, etc.)."""

    # ------------------------------------------------------------------
    # Schema / versioning helpers
    # ------------------------------------------------------------------
    SCHEMA_VERSION: str = "1.0"
    """Implementations can override to track snapshot schema version."""
