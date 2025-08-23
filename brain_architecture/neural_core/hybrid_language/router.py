#!/usr/bin/env python3
"""Hybrid SLM + LLM Sparse-Mixture Router ‑ Prototype

Inspired by Dettmers et al. 2023 (small-LM survey) and Kirkpatrick et al.
2022 (sparse mixture / continual-learning).  The router keeps a pool of
sub-models (strings in this stub) each with an associated *score* function
(here, a callable).  On a given prompt, the router evaluates all scores,
selects the *top-k* highest, and forwards the prompt only to those
models – enabling sparsity and modularity.

This initial prototype is framework-agnostic; downstream integration will
wrap actual model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any
import heapq


# ------------------------------------------------------------------
# Model registration
# ------------------------------------------------------------------


ScoreFn = Callable[[str], float]


@dataclass
class SubModel:
    name: str
    score_fn: ScoreFn  # returns relevance score for a prompt
    backend: str = "stub"  # e.g., "slm", "llm"


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------


class SparseMixtureRouter:
    """Selects top-k sub-models for each prompt, returns their outputs."""

    def __init__(self, sub_models: List[SubModel] | None = None, k: int = 2):
        self.models: List[SubModel] = sub_models or []
        self.k = k

    # --------------------------------------------------------------
    def register(self, model: SubModel) -> None:
        self.models.append(model)

    # --------------------------------------------------------------
    def route(self, prompt: str) -> Dict[str, Any]:
        """Return dict of model_name → mocked response for selected models."""
        if not self.models:
            raise ValueError("No sub-models registered")

        # compute scores and take top-k
        scored = [(m.score_fn(prompt), m) for m in self.models]
        top_k = heapq.nlargest(self.k, scored, key=lambda t: t[0])

        # Generate stubbed outputs
        out: Dict[str, Any] = {}
        for _score, model in top_k:
            out[model.name] = {
                "backend": model.backend,
                "response": f"<mock-reply from {model.name}>",
            }
        return out


__all__ = ["SparseMixtureRouter", "SubModel"]
