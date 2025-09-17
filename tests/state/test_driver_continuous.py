import sys
import types

import pytest

# Ensure placeholder for missing roadmap_controller to satisfy import chain
sys.modules.setdefault("quark_state_system.roadmap_controller", types.ModuleType("rc"))
if not hasattr(sys.modules["quark_state_system.roadmap_controller"], "RoadmapController"):

    class _StubRC:  # noqa: D401 simple stub
        def get_next_actionable_goal(self):
            return None

    sys.modules["quark_state_system.roadmap_controller"].RoadmapController = _StubRC

sys.modules.setdefault("brain_modules", types.ModuleType("bm"))
# Stub nested compliance engine path
pkg_ai = types.ModuleType("brain_modules.alphagenome_integration")
sys.modules.setdefault("brain_modules.alphagenome_integration", pkg_ai)
ce_mod = types.ModuleType("brain_modules.alphagenome_integration.compliance_engine")
if not hasattr(ce_mod, "ComplianceEngine"):

    class _ComplianceEngine:  # noqa: D401 simple stub
        def validate_action_legality(self, _):
            return True

    ce_mod.ComplianceEngine = _ComplianceEngine
sys.modules["brain_modules.alphagenome_integration.compliance_engine"] = ce_mod

from state.quark_state_system.agile_utils import parse_continuous
from state.quark_state_system.quark_driver import QuarkDriver


def test_parse_continuous_logic():
    assert parse_continuous("continuous + 3") == 3


@pytest.fixture()
def stub_driver(monkeypatch, tmp_path):
    # Patch AutonomousAgent to lightweight stub to avoid heavy deps
    class _StubAgent:
        def __init__(self, root):
            self.counter = 0
            self.roadmap = types.SimpleNamespace(get_next_actionable_goal=lambda: None)
            self.compliance = types.SimpleNamespace(validate_action_legality=lambda _: True)

        def execute_next_goal(self):
            self.counter += 1
            return True

    monkeypatch.setattr(
        "state.quark_state_system.quark_driver.AutonomousAgent", _StubAgent, raising=True
    )

    drv = QuarkDriver(str(tmp_path))

    # Fake a current goal so driver doesn't exit early
    drv.current_goal = {"task": "stub"}

    # Override run_phase_tasks to capture N
    calls = {}

    def _capture(max_tasks: int = 5):  # noqa: ANN001
        calls["n"] = max_tasks

    drv.run_phase_tasks = _capture  # type: ignore [method-assign]
    return drv, calls


def test_driver_continuous_executes_n(monkeypatch, stub_driver):
    drv, calls = stub_driver
    drv.process_prompt("Please run continuous + 2 tasks now")
    assert calls.get("n") == 2
