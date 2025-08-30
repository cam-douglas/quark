"""State-level facade for Quark's recommendation / roadmap engine.

This module wraps the original top-level ``quark_state_system`` package so that
external code can simply do::

    from state.quark_state_system import next_steps

and remain agnostic to the file-system re-organisation.
"""
from importlib import import_module
from types import ModuleType
from typing import Any, Callable
from .quark_recommendations import QuarkRecommendationsEngine

_orig: ModuleType | None = None
try:
    _orig = import_module("quark_state_system")
except ModuleNotFoundError:  # pragma: no cover – should not happen after move
    _orig = None  # type: ignore[assignment]

if _orig is not None and hasattr(_orig, "next_steps"):
    next_steps: Callable[..., Any] = getattr(_orig, "next_steps")  # type: ignore[assignment]
else:
    def next_steps(*args: Any, **kwargs: Any):  # type: ignore[return-type]
        raise RuntimeError(
            "quark_state_system.next_steps() is unavailable – original module not found."
        )

_ENGINE = QuarkRecommendationsEngine()


def ask_quark(query: str) -> str:
    """Unified entry-point: pass any natural-language request and get guidance."""
    return _ENGINE.provide_intelligent_guidance(query)

__all__ = ["next_steps"]
