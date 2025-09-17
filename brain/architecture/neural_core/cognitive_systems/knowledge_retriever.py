"""KnowledgeRetriever converts natural-language questions into queries against EpisodicMemory.

This first pass keeps things lightweight by re-using the SimpleTFIDF vectorizer
for similarity and falling back to EpisodicMemory.retrieve_episode for final
scoring. It can be replaced later with dense embeddings without changing the
public API.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from brain._bootstrap import USE_E8_MEMORY

if USE_E8_MEMORY:
    try:
        from brain.architecture.neural_core.cognitive_systems.memory_providers.e8_adapter import (
            E8MemoryAdapter,
        )
    except Exception as _exc:  # pragma: no cover – optional dep
        logging.getLogger(__name__).warning(
            "KnowledgeRetriever: failed to import E8MemoryAdapter (%s); falling back to TF-IDF",
            _exc,
        )
        E8MemoryAdapter = None  # type: ignore

from brain.architecture.neural_core.cognitive_systems.tfidf_vectorizer import (
    SimpleTFIDF,
)

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Bridge between user questions and memory retrieval."""

    def __init__(self, episodic_memory: "EpisodicMemory"):
        self.memory = episodic_memory
        self.vectorizer = SimpleTFIDF()
        self._corpus: List[str] = []  # mirror of episode text used for TF-IDF
        self._episode_ids: List[str] = []
        self._fitted: bool = False

        # Optional E8 memory adapter
        self._e8_adapter = None
        if USE_E8_MEMORY and E8MemoryAdapter is not None:
            try:
                self._e8_adapter = E8MemoryAdapter()
                logger.info("KnowledgeRetriever: Using E8MemoryAdapter backend")
            except Exception as _exc:  # pragma: no cover
                logger.warning("E8MemoryAdapter init failed: %s", _exc)
                self._e8_adapter = None

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------
    def build_index(self):
        """Extract text from episodes and fit TF-IDF vectorizer."""
        self._corpus.clear()
        self._episode_ids.clear()
        for ep in self.memory.episodes.values():
            # naive: concat content dict into str
            text = str(ep.content)
            self._corpus.append(text)
            self._episode_ids.append(ep.episode_id)

            if self._e8_adapter is not None:
                self._e8_adapter.store(text, metadata={"episode_id": ep.episode_id})

        if not self._corpus:
            logger.info("KnowledgeRetriever: No episodes to index yet.")
            self._fitted = False
            return

        self.vectorizer.fit(self._corpus)
        self._fitted = True
        logger.info(
            "KnowledgeRetriever index built with %d episodes and %d terms",
            len(self._corpus),
            self.vectorizer.vocab_size,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 5):
        """Return episodes matching *question* ranked by TF-IDF cosine."""
        # E8 path first
        if self._e8_adapter is not None:
            results = self._e8_adapter.query(question, top_k=top_k)
            if results:
                return [
                    (self.memory.episodes[r[0]], r[1])
                    if r[0] in self.memory.episodes else (None, r[1])
                    for r in results
                ]

        if not self._fitted:
            self.build_index()
            if not self._fitted:
                return []

        ranked = self.vectorizer.most_similar(question, self._corpus, top_k=top_k)
        episode_scores: List[Tuple["MemoryEpisode", float]] = []
        for idx, score in ranked:
            ep_id = self._episode_ids[idx]
            ep = self.memory.episodes.get(ep_id)
            if ep:
                episode_scores.append((ep, score))

        # If TF-IDF fails (e.g., empty index), fall back to EpisodicMemory.retrieve_episode
        if not episode_scores:
            results = self.memory.retrieve_episode({"text": question}, max_results=top_k)
            return [(ep, 1.0) for ep in results]

        return episode_scores
