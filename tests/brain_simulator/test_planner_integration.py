"""Integration test: BrainSimulator ↔ Advanced Planner ↔ Global Workspace / MetaController.

Skipped if local Llama2 model or transformers missing. Marked heavy by default.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

MODEL_DIR = Path("/Users/camdouglas/quark/data/models/llama2_7b_chat_uncensored")
pytest.importorskip("transformers", reason="transformers not installed – skip heavy LLM tests")

if not MODEL_DIR.exists():
    pytest.skip(
        "Local model not present – planner integration test skipped", allow_module_level=True
    )

# Ensure planner picks up the registry entry
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager

rm = ResourceManager()
if not any(m.get("integrated_path") == str(MODEL_DIR) for m in rm.registry.values()):
    rm.register_resource(MODEL_DIR, {"type": "model", "name": "llama2_7b_chat_uncensored"})

from brain.architecture.brain_simulator import BrainSimulator


def test_planner_broadcast_and_meta_controller():
    bs = BrainSimulator()
    bullet = "Test proprioceptive feedback loop"
    bs.generate_subtasks(bullet)

    # Check global workspace broadcast
    recent = bs.global_workspace.get_broadcast()["recent"]
    assert any("tasks" in item and item.get("source") == "advanced_planner" for item in recent)

    # Check meta-controller queue populated
    pending = bs.meta_controller.get_pending_tasks()
    assert len(pending) >= 4
    assert all("title" in t for t in pending)
