

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from __future__ import annotations

"""E8MemoryAdapter bridges Quark's memory API with Kaleidescope's E8 lattice engine.

Minimal MVP goals (Step 2 of 8):
1. Provide a drop-in backend matching TF-IDF behaviour exposed in KnowledgeRetriever.
2. Expose mood & drive knobs for future tuning.
3. Hide GPL-licensed code behind env flag to avoid license contagion unless user opts-in.

API (stable):
    store(text: str, metadata: dict | None) -> str
    query(text: str, top_k: int = 5) -> list[tuple[str, float, dict]]
    set_mood(mood: dict) -> None
    set_drives(drives: dict) -> None

Implementation details:
• When USE_E8_MEMORY env var is truthy we attempt to import Kaleidescope's
  e8_mind_server_M16.MemoryManager. If that fails (missing deps) we gracefully
  fall back to an in-process KDTree wrapper.
• To keep the public GPL code at arms-length, we only *import* it at runtime
  when the feature flag is enabled.
• Embeddings: for the MVP we generate simple sentence-level embeddings using
  scikit-learn's TfidfVectorizer to stay light. This will be replaced with
  Kaleidescope's internal embedding_fn once the heavy deps land.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)

__all__ = [
    "E8MemoryAdapter",
]

# -----------------------------------------------------------------------------
# Helper – lightweight fallback embedding
# -----------------------------------------------------------------------------

class _LocalTFIDFEmbedder:
    """Generates TF-IDF embeddings so the adapter works even without heavy deps."""

    def __init__(self):
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._corpus: List[str] = []

    def fit_add(self, texts: List[str]):
        self._corpus.extend(texts)
        self._vectorizer = TfidfVectorizer(max_features=2048)
        self._vectorizer.fit(self._corpus)

    def embed(self, texts: List[str]):
        if self._vectorizer is None:
            # Fit on the fly with initial corpus
            self.fit_add(texts)
        return self._vectorizer.transform(texts).toarray().astype(np.float32)

# -----------------------------------------------------------------------------
# Core Adapter
# -----------------------------------------------------------------------------

class E8MemoryAdapter:
    """Optional backend leveraging Kaleidescope if enabled.

    If the environment variable USE_E8_MEMORY is not set (or falsey), the
    adapter remains a no-op stub resembling the public API so callers don't
    need to branch.
    """

    def __init__(self):
        self.enabled = os.getenv("USE_E8_MEMORY", "false").lower() in {"1", "true", "yes"}
        self._embeddings: List[np.ndarray] = []
        self._payloads: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self._id_seq = 0
        self._kdtree: Optional[KDTree] = None
        self._mood: Dict[str, float] = {}
        self._drives: Dict[str, float] = {}

        if self.enabled:
            try:
                from brain.architecture.neural_core.cognitive_systems.e8_kaleidescope import (  # pylint: disable=import-error
                    MemoryManager,  # type: ignore
                    EMBED_DIM,
                )

                # Minimal stub objects to satisfy MemoryManager constructor
                class _StubMind:  # noqa: D401 – simple stub
                    def __init__(self):
                        self.probe = None
                        self.llm_pool = None

                self._kmgr = MemoryManager(
                    embedding_fn=self._dummy_embed,
                    mood=self._mood,
                    subconscious=None,
                    run_id="quark_integration",
                    probe=None,
                    llm_caller=None,
                    mind_instance=_StubMind(),
                )
                self._embedding_dim = EMBED_DIM
                logger.info("E8MemoryAdapter: Kaleidescope MemoryManager loaded (dim=%d)", EMBED_DIM)
            except Exception as exc:  # pragma: no cover – import guard
                self.enabled = False
                self._kmgr = None
                logger.warning(
                    "E8MemoryAdapter: Failed to import Kaleidescope (falling back) – %s", exc
                )
        else:
            self._kmgr = None

        # Local TF-IDF fallback ensures adapter remains functional
        self._local_embedder = _LocalTFIDFEmbedder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store *text* representation in memory and return item_id."""
        metadata = metadata or {}
        item_id = f"e8mem_{self._id_seq}"
        self._id_seq += 1

        if self.enabled and self._kmgr is not None:
            # Kaleidescope path – rely on internal logic
            vec = self._dummy_embed([text])[0]
            self._kmgr.main_vectors[item_id] = vec  # type: ignore[attr-defined]
            self._kmgr._main_storage_ids.append(item_id)  # noqa: SLF001 – internal use
            # defer KDTree build; Kaleidescope has its own indexing strategy
            # Maintain a lightweight KDTree for fast nearest-neighbour when main_vectors grows
            if not hasattr(self, "_e8_kdtree"):  # build first time
                self._e8_kdtree = None
                self._e8_mat = None
            if len(self._kmgr.main_vectors) % 1000 == 0:  # rebuild every 1k inserts
                self._e8_mat = np.vstack(list(self._kmgr.main_vectors.values()))
                self._e8_kdtree = KDTree(self._e8_mat, metric="euclidean")
        else:
            vec = self._local_embedder.embed([text])[0]
            self._embeddings.append(vec)
            self._payloads.append({"text": text, **metadata})
            self._ids.append(item_id)
            self._rebuild_kdtree_if_needed()
            # Map raw text to index for exact-match fast path
            if not hasattr(self, "_text_map"):
                self._text_map = {}
            self._text_map[text] = len(self._ids) - 1

        return item_id

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return list of (item_id, similarity, payload)."""
        if self.enabled and self._kmgr is not None:
            # Very naive – compute cosine similarity against stored vectors
            if not self._kmgr.main_vectors:
                return []
            query_vec = self._dummy_embed([text])[0]
            ids, vecs = zip(*self._kmgr.main_vectors.items())  # type: ignore[attr-defined]

            # Use KDTree if available for O(log n) search
            if getattr(self, "_e8_kdtree", None) is not None:
                dist, idx = self._e8_kdtree.query(query_vec.reshape(1, -1), k=min(top_k, len(vecs)))
                idx = idx[0]; dist = dist[0]
                return [
                    (list(ids)[i], float(1.0 - dist[j]), {"text": ""})
                    for j, i in enumerate(idx)
                ]

            # Fallback to brute-force cosine
            mat = np.vstack(vecs)
            sims = 1.0 - self._cosine_distance(mat, query_vec)
            top_indices = np.argsort(sims)[-top_k:][::-1]
            return [
                (ids[i], float(sims[i]), {"text": ""})
                for i in top_indices
            ]

        # Fallback path
        # 1) exact match short-circuit
        if hasattr(self, "_text_map") and text in self._text_map:
            idx = self._text_map[text]
            return [(self._ids[idx], 1.0, self._payloads[idx])]

        if not self._embeddings:
            return []
        query_vec = self._local_embedder.embed([text])[0]
        mat = np.vstack(self._embeddings)
        sims = 1.0 - self._cosine_distance(mat, query_vec)
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [
            (self._ids[i], float(sims[i]), self._payloads[i])
            for i in top_indices
        ]

    def set_mood(self, mood: Dict[str, float]):
        self._mood.update(mood)
        if self.enabled and self._kmgr is not None:
            self._kmgr.mood.update(mood)  # type: ignore[attr-defined]

    def set_drives(self, drives: Dict[str, float]):
        self._drives.update(drives)
        # Placeholder – Kaleidescope integration TBD

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_kdtree_if_needed(self):
        # Rebuild KDTree for fallback path
        if self._embeddings:
            mat = np.vstack(self._embeddings)
            self._kdtree = KDTree(mat, metric="euclidean")

    @staticmethod
    def _cosine_distance(mat: np.ndarray, vec: np.ndarray):
        # cosine distance = 1 - cosine similarity
        dots = mat @ vec
        mat_norms = np.linalg.norm(mat, axis=1)
        vec_norm = np.linalg.norm(vec)
        return 1.0 - (dots / (mat_norms * vec_norm + 1e-8))

    def _dummy_embed(self, texts):
        """Temporary embedding using random projection (placeholder)."""
        rng = np.random.default_rng(42)
        dim = getattr(self, "_embedding_dim", 256)
        return rng.normal(size=(len(texts), dim)).astype(np.float32)
