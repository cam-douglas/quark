# ---- Test-only stubs to avoid heavy dependencies -----------------------
import sys
import types

import pytest

from brain.architecture.neural_core.cognitive_systems.knowledge_hub import (
    KnowledgeHub,
    KnowledgeType,
)
from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory

# Provide a lightweight stub for LanguageCortex so llm_fallback import is cheap
stub_mod = types.ModuleType("language_cortex_stub")


class _DummyLC:  # noqa: N801
    def process_input(self, prompt: str):
        return "stub-llm-answer"


stub_mod.LanguageCortex = _DummyLC
sys.modules["brain.architecture.neural_core.language.language_cortex"] = stub_mod

# Monkeypatch llm_fallback.answer_with_llm to avoid network
import brain.architecture.neural_core.cognitive_systems.llm_fallback as _lf  # noqa: E402

_lf.answer_with_llm = lambda q: "stub-llm-answer"


@pytest.fixture
def hub_and_memory():
    hub = KnowledgeHub()
    mem = EpisodicMemory(max_episodes=10)
    return hub, mem


def test_assimilate_text(hub_and_memory):
    hub, _ = hub_and_memory
    text = "Python is a popular programming language."
    objs = hub.assimilate(text, source="unit-test", citation="")
    assert len(objs) == 1
    obj = objs[0]
    assert obj.k_type in (KnowledgeType.DECLARATIVE, KnowledgeType.PROCEDURAL)
    assert obj.content["text"].startswith("Python is")


def test_retrieve_with_memory(hub_and_memory):
    hub, mem = hub_and_memory
    # Store a known fact
    objs = hub.assimilate("The capital of France is Paris.", source="wiki", citation="")
    for o in objs:
        mem.inject_knowledge(o)

    result = hub.retrieve("What is the capital of France?", episodic_memory=mem, top_k=3)
    assert result["episodes"], "Should retrieve at least one episode"
    assert "Paris" in str(result["episodes"][0].content)
