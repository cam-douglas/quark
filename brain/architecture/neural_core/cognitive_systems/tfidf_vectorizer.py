"""Simple TF-IDF vectorizer without external dependencies.

The implementation is intentionally minimalistic to avoid adding large
third-party libraries (e.g., scikit-learn) into the core brain package.
If you later decide to switch to a more sophisticated encoder, replace
this module while keeping the public interface intact.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Dict, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric word tokenization."""
    return _WORD_RE.findall(text.lower())


class SimpleTFIDF:
    """Very small TF-IDF implementation suitable for a few thousand docs."""

    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.vocab_size: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, corpus: List[str]) -> "SimpleTFIDF":
        """Compute IDF values from *corpus* (list of documents)."""
        doc_freq: Counter[str] = Counter()
        for doc in corpus:
            tokens = set(_tokenize(doc))
            doc_freq.update(tokens)

        num_docs = len(corpus)
        self.idf = {
            term: math.log((1 + num_docs) / (1 + df)) + 1.0 for term, df in doc_freq.items()
        }
        self.vocab_size = len(self.idf)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def transform(self, docs: List[str]) -> List[Dict[str, float]]:
        """Return sparse TF-IDF vectors as dict term→weight."""
        if not self._fitted:
            raise RuntimeError("SimpleTFIDF must be fitted before calling transform().")

        vectors: List[Dict[str, float]] = []
        for doc in docs:
            tf = Counter(_tokenize(doc))
            # term frequency normalised by document length
            denom = sum(tf.values()) or 1
            vec = {term: (freq / denom) * self.idf.get(term, 0.0) for term, freq in tf.items() if term in self.idf}
            vectors.append(vec)
        return vectors

    # Convenience helpers ------------------------------------------------
    def fit_transform(self, corpus: List[str]) -> List[Dict[str, float]]:
        self.fit(corpus)
        return self.transform(corpus)

    # Similarity ----------------------------------------------------------
    @staticmethod
    def _cosine(u: Dict[str, float], v: Dict[str, float]) -> float:
        if not u or not v:
            return 0.0
        # Intersection
        common_terms = set(u.keys()) & set(v.keys())
        num = sum(u[t] * v[t] for t in common_terms)
        denom_u = math.sqrt(sum(x * x for x in u.values()))
        denom_v = math.sqrt(sum(x * x for x in v.values()))
        if denom_u == 0 or denom_v == 0:
            return 0.0
        return num / (denom_u * denom_v)

    def most_similar(
        self,
        query: str,
        docs: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Return indices and scores of *docs* most similar to *query*."""
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before calling most_similar().")

        query_vec = self.transform([query])[0]
        doc_vecs = self.transform(docs)
        scores = [self._cosine(query_vec, dv) for dv in doc_vecs]
        # Return sorted indices & scores
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
