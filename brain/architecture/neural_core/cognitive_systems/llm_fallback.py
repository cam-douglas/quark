"""LLM fallback for answering questions when memory search yields nothing."""
from __future__ import annotations

from typing import Optional
import logging

try:
    from brain.architecture.neural_core.language.language_cortex import LanguageCortex
except ImportError as e:  # pragma: no cover
    # In stripped-down environments the heavy language cortex may not be present.
    LanguageCortex = None  # type: ignore
    _import_error = e
else:
    _import_error = None

logger = logging.getLogger(__name__)


_cortex: Optional[LanguageCortex] = None

def _get_cortex() -> Optional[LanguageCortex]:
    global _cortex
    if _cortex is None and LanguageCortex is not None:
        try:
            _cortex = LanguageCortex()
        except Exception as e:
            logger.error(f"Failed to init LanguageCortex: {e}")
            return None
    return _cortex


def answer_with_llm(question: str) -> Optional[str]:
    """Return an LLM-generated answer or None if unavailable.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
    if LanguageCortex is None:
        logger.warning(f"LanguageCortex import failed: {_import_error}")
        return None
    cortex = _get_cortex()
    if cortex is None:
        return None
    try:
        from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
        rm = ResourceManager.get_default()
        if rm:
            with rm.request_resources("nlp"):
                return cortex.process_input(question)
        return cortex.process_input(question)
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return None
