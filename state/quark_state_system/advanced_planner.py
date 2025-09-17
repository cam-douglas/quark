"""Advanced Planner – generates granular sub-tasks for a roadmap bullet.

Usage
-----
>>> from state.quark_state_system.advanced_planner import plan
>>> tasks = plan("Dopamine (RPE, motor/cognition)")

Returns a list of dicts: [{"title": str, "priority": "high|medium|low"}, ...]

The planner attempts to load a local Transformer-based LLM (Mistral-7B,
Llama-2, etc.) via HuggingFace 🤗. If no model is available, it falls back to a
very simple heuristic so that the pipeline never breaks.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""

import logging
from pathlib import Path
from typing import List, Dict

# Agile phase/step label helper
from state.quark_state_system.agile_utils import format_phase_step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Model bootstrap
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_NAME_CANDIDATES = [
    # Prefer already-downloaded local directory first
    "llama2_7b_chat_uncensored",
    "/Users/camdouglas/quark/data/models/llama2_7b_chat_uncensored",
    "local-mistral-7b",  # hypothetical local name
    "georgesung/llama2_7b_chat_uncensored",
]

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from brain.architecture.neural_core.cognitive_systems.local_llm_wrapper import LocalLLMWrapper
        # Search registry first
        from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
        rm = ResourceManager()
        for meta in rm.registry.values():
            if meta.get("type") == "model" and meta.get("name") in _MODEL_NAME_CANDIDATES:
                try:
                    _MODEL = LocalLLMWrapper(meta["integrated_path"]).generate
                    logger.info("Advanced planner loaded model via ResourceManager: %s", meta["name"])
                    return _MODEL
                except Exception as e:
                    logger.warning("Failed to init LocalLLMWrapper: %s", e)
        # Fallback: attempt direct path
        for name in _MODEL_NAME_CANDIDATES:
            mp = Path(name).expanduser()
            if mp.exists():
                try:
                    _MODEL = LocalLLMWrapper(mp).generate
                    logger.info("Advanced planner loaded model path: %s", mp)
                    return _MODEL
                except Exception as e:
                    logger.warning("LocalLLMWrapper failed for %s: %s", mp, e)
    except ImportError as e:
        logger.info("transformers/local wrapper not installed – falling back (%s)", e)
    _MODEL = None
    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan(text: str) -> List[Dict]:
    """Return granular work items derived from *text*.

    The strategy is:
    1. Try an LLM prompt that asks for 4-8 engineering subtasks.
    2. If the model isn’t available, fallback to heuristic (design/impl/test/doc).
    """
    model = _load_model()
    if model is not None:
        prompt = (
            "You are a senior AGI engineer. Break the following roadmap bullet into "
            "concrete engineering subtasks (4-10 items). Each item should start with a verb "
            "(Design, Implement, Create, Benchmark, Document, etc.) and be concise (~1 line).\n"
            f"BULLET: {text}\n"
            "Return as a numbered list."
        )
        try:
            gen = model(prompt, max_new_tokens=256, temperature=0.1)[0]["generated_text"]
            lines = [ln.strip(" –-*") for ln in gen.splitlines() if ln.strip()]
            tasks = []
            for ln in lines:
                if not ln[0].isdigit():
                    continue
                title = ln.lstrip("0123456789. ")
                tasks.append({"title": f"{format_phase_step(0,0,0,0)} — {text} – {title}", "priority": "medium"})
            if tasks:
                return tasks
        except Exception as e:
            logger.warning("LLM generation failed (%s). Falling back.", e)

    # Heuristic fallback
    return [
        {"title": f"{format_phase_step(0,0,0,0)} — {text} – Design", "priority": "medium"},
        {"title": f"{format_phase_step(0,0,0,0)} — {text} – Implementation", "priority": "medium"},
        {"title": f"{format_phase_step(0,0,0,0)} — {text} – Unit tests", "priority": "medium"},
        {"title": f"{format_phase_step(0,0,0,0)} — {text} – Documentation", "priority": "medium"},
    ]

__all__ = ["plan"]
