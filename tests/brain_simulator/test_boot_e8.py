import importlib
import os
import sys
import types

import pytest

# Mark this entire module as heavy so it's skipped in quick test runs
pytestmark = pytest.mark.heavy


@pytest.mark.parametrize("flag", ["false", "true"])
def test_brain_simulator_boot(monkeypatch, flag):
    os.environ["USE_E8_MEMORY"] = flag

    # Patch heavy submodules to lightweight stubs to keep boot fast
    monkeypatch.setitem(
        sys.modules,
        "brain.architecture.neural_core.cognitive_systems.local_llm_wrapper",
        types.ModuleType("stub"),
    )
    # stub ResourceManager class
    stub_rm = types.ModuleType("stub_rm")

    class _StubRM:  # noqa: D401 simple stub
        registry = {}

        def __init__(self):
            pass

    stub_rm.ResourceManager = _StubRM
    monkeypatch.setitem(
        sys.modules,
        "brain.architecture.neural_core.cognitive_systems.resource_manager",
        stub_rm,
    )

    try:
        importlib.reload(importlib.import_module("brain.architecture.brain_simulator"))
    except Exception as exc:
        pytest.fail(f"brain_simulator failed to import with USE_E8_MEMORY={flag}: {exc}")
