"""State-level facade for Quark's recommendation / roadmap engine.

This module wraps the original top-level ``quark_state_system`` package so that
external code can simply do::

    from state.quark_state_system import next_steps

and remain agnostic to the file-system re-organisation.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
from importlib import import_module
from types import ModuleType
from typing import Any, Callable
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

# Lazy loading to avoid circular import
_ENGINE = None


def ask_quark(query: str) -> str:
    """Unified entry-point: pass any natural-language request and get guidance."""
    global _ENGINE
    if _ENGINE is None:
        from .quark_recommendations import QuarkRecommendationsEngine
        _ENGINE = QuarkRecommendationsEngine()
    return _ENGINE.provide_intelligent_guidance(query)

__all__ = ["next_steps"]