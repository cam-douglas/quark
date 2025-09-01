"""Tests for LocalLLMWrapper integration.

These tests will be skipped automatically if the heavy model files are not
available locally or if `transformers` is not installed.  They should therefore
run quickly on CI while still exercising the integration when resources are
present.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

MODEL_DIR = Path("/Users/camdouglas/quark/data/models/llama2_7b_chat_uncensored")

pytest.importorskip("transformers", reason="transformers not installed – skip LLM tests")

if not MODEL_DIR.exists():
    pytest.skip(
        "Local Llama2 model not present – skipping LLM integration tests", allow_module_level=True
    )

from brain.architecture.neural_core.cognitive_systems.local_llm_wrapper import LocalLLMWrapper
from state.quark_state_system import advanced_planner


def test_llm_smoke_hello():
    """Generate a short completion for a trivial prompt."""
    llm = LocalLLMWrapper(MODEL_DIR)
    out = llm.generate("Hello", max_new_tokens=1, temperature=0.0)
    assert isinstance(out, str) and len(out.strip()) > 0


def test_advanced_planner_tasks():
    """Ensure advanced_planner returns at least 4 tasks when LLM is available."""
    tasks = advanced_planner.plan("test bullet")
    assert len(tasks) >= 4
    assert all("title" in t for t in tasks)
