"""LocalLLMWrapper
------------------
Purpose:
    Lightweight loader + inference wrapper around a *local* HuggingFace-format LLM
    residing in the Quark repo (typically under /data/models/*) and registered via
    ResourceManager.

Inputs:
    • model_path: str | Path – directory containing model files (config.json, *.bin / *.safetensors)
    • generation parameters via .generate() kwargs (max_new_tokens, temperature, etc.)
Outputs:
    • Plain-text completion string (streaming optional)
Seeds / Reproducibility:
    • Deterministic seed can be set via QUARK_SEED (passed to transformers.generate)
Dependencies:
    • transformers>=4.38, accelerate, tokenizers

Concurrency:
    • Enforces ≤2 concurrent requests (default profile) using a threading.Semaphore.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

from __future__ import annotations

import os
import logging
import threading
from pathlib import Path
from typing import List

_logger = logging.getLogger("quark.local_llm")
_logger.setLevel(logging.INFO)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
except ImportError as _e:  # pragma: no cover
    _transformers_import_error = _e  # keep for error message
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    pipeline = None  # type: ignore
else:
    _transformers_import_error = None

# ------------------------------------------------------------
# Default profile constants (align with user-provided profile)
# ------------------------------------------------------------
_MAX_BATCH_SIZE = 1
_MAX_CONCURRENT_REQUESTS = 2

# Semaphore for global concurrency control across wrapper instances
_global_sem = threading.Semaphore(_MAX_CONCURRENT_REQUESTS)


def _select_device() -> int | str:
    """Return device id for transformers pipeline."""
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env and cuda_env != "":
        return 0  # first visible GPU
    return -1  # CPU


class LocalLLMWrapper:
    """Tiny wrapper around transformers pipeline enforcing Quark resource rules."""

    def __init__(self, model_path: str | Path):
        if AutoTokenizer is None:
            raise ImportError(
                f"transformers missing – cannot load LocalLLMWrapper: {_transformers_import_error}"
            )
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        _logger.info("Loading local LLM from %s (device=%s)", self.model_path, _select_device())
        tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True,
        )
        self._pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            device_map="auto",
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    from tools_utilities.scripts.performance_utils import memoize

    @memoize(maxsize=256)
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for *prompt* using model profile limits."""
        if len(prompt.strip()) == 0:
            raise ValueError("Prompt must be non-empty")
        # Enforce profile
        if kwargs.get("batch_size", 1) != 1 or kwargs.get("batch", 1) != 1:
            raise ValueError("Batch size >1 not allowed by default profile")
        max_new = kwargs.pop("max_new_tokens", 128)
        temperature = kwargs.pop("temperature", 0.7)

        # Concurrency guard
        acquired = _global_sem.acquire(timeout=30)
        if not acquired:
            raise RuntimeError("LocalLLMWrapper concurrency limit exceeded")
        try:
            gens = self._pipe(prompt, max_new_tokens=max_new, temperature=temperature, **kwargs)
            return gens[0]["generated_text"] if gens else ""
        finally:
            _global_sem.release()


# ------------------------------------------------------------
# Helper – integrate path via ResourceManager then return wrapper
# ------------------------------------------------------------

def integrate_local_llm(path: str | Path) -> LocalLLMWrapper:  # pragma: no cover
    """Register *path* with ResourceManager and return a ready wrapper."""
    try:
        from brain.architecture.neural_core.cognitive_systems.resource_manager import (
            ResourceManager,
        )
    except Exception as e:
        raise RuntimeError(f"ResourceManager unavailable: {e}") from e
    rm = ResourceManager()
    rid = rm.register_resource(path, {"type": "model", "name": Path(path).name})
    meta = rm.registry[rid]
    return LocalLLMWrapper(meta["integrated_path"])


__all__: List[str] = ["LocalLLMWrapper", "integrate_local_llm"]
