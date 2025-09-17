"""Tests for the new "continuous" trigger in ``QuarkDriver``.

The real ``AutonomousAgent`` performs heavyweight roadmap operations that are
not needed for verifying the driver loop.  We stub it with a lightweight fake
that simply records how many times ``execute_next_goal`` was called and can be
configured to raise exceptions to simulate run-time errors.
"""

import importlib
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyAgent:
    """A minimal stand-in for ``AutonomousAgent``.

    Parameters
    ----------
    fail_until : int, default 0
        How many initial calls to ``execute_next_goal`` should raise an
        exception to simulate errors.
    max_calls : int | None
        Optional hard cap – after this many successful calls, the agent will
        report that there are no more actionable goals by returning ``False``.
    """

    def __init__(self, *_, fail_until: int = 0, max_calls: int | None = None, **__):
        from types import SimpleNamespace

        self.calls = 0
        self.fail_until = fail_until
        self.max_calls = max_calls
        # provide stubbed attributes expected by QuarkDriver
        self.compliance = SimpleNamespace(validate_action_legality=lambda *_: True)
        self.roadmap = SimpleNamespace(
            get_next_actionable_goal=lambda: None, report_progress=lambda *_: None
        )

    # pylint: disable=unused-argument
    def execute_next_goal(self):  # noqa: D401 (simple verb ok)
        """Simulate goal execution with optional error injection."""
        if self.calls < self.fail_until:
            self.calls += 1
            raise RuntimeError("simulated failure")
        if self.max_calls is not None and self.calls >= self.max_calls:
            return False  # no more goals
        self.calls += 1
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_autonomous_agent(monkeypatch):
    """Replace the heavy AutonomousAgent with :class:`DummyAgent`."""
    # Ensure alias so internal imports resolve
    if "quark_state_system" not in sys.modules:
        alias_mod = types.ModuleType("quark_state_system")
        alias_mod.__path__ = []  # mark as package
        sys.modules["quark_state_system"] = alias_mod

    # Stub roadmap_controller to satisfy autonomous_agent import chain
    if "quark_state_system.roadmap_controller" not in sys.modules:
        rc_mod = types.ModuleType("quark_state_system.roadmap_controller")

        class _DummyRoadmap:
            def get_next_actionable_goal(self):
                return None

            def report_progress(self, *_):
                pass

        rc_mod.RoadmapController = _DummyRoadmap
        sys.modules["quark_state_system.roadmap_controller"] = rc_mod

    # Stub ComplianceEngine import chain
    comp_path = "brain_modules.alphagenome_integration.compliance_engine"
    if comp_path not in sys.modules:
        comp_mod = types.ModuleType(comp_path)

        class _DummyCompliance:
            def validate_action_legality(self, *_):
                return True

        comp_mod.ComplianceEngine = _DummyCompliance
        # Ensure parent packages exist
        pkg_parts = comp_path.split(".")
        for i in range(1, len(pkg_parts)):
            pkg_name = ".".join(pkg_parts[:i])
            if pkg_name not in sys.modules:
                pkg = types.ModuleType(pkg_name)
                pkg.__path__ = []
                sys.modules[pkg_name] = pkg
        sys.modules[comp_path] = comp_mod

    module_path = "state.quark_state_system.autonomous_agent"
    agent_mod = importlib.import_module(module_path)

    def _factory(*args, **kwargs):  # noqa: D401
        return DummyAgent(*args, **kwargs)

    monkeypatch.setattr(agent_mod, "AutonomousAgent", _factory)
    yield


@pytest.fixture()
def driver(monkeypatch):
    """Return a fresh ``QuarkDriver`` instance that uses the dummy agent."""
    module_path = "state.quark_state_system.quark_driver"
    drv_mod = importlib.import_module(module_path)

    # Patch the AutonomousAgent class *before* instantiating QuarkDriver
    monkeypatch.setattr(drv_mod, "AutonomousAgent", DummyAgent)

    driver_cls = drv_mod.QuarkDriver
    drv = driver_cls("/tmp")
    drv.current_goal = {"task": "dummy"}
    return drv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trigger_detection_runs_limit(driver):
    """Driver should run *limit* tasks when trigger has numeric argument."""
    driver.process_prompt("continuous 3")
    # Using DummyAgent we can directly access call count
    assert driver.agent.calls == 3


def test_error_resilience(driver):
    """Driver retries tasks up to 3 × and continues after resolution."""
    # Re-inject DummyAgent that fails twice then succeeds once, limit 1.
    driver.agent.fail_until = 2
    driver.agent.max_calls = 1  # stop after first successful goal

    driver.process_prompt("proceed continuous 1")
    # The agent should have been called 3 times (2 failures + 1 success)
    assert driver.agent.calls == 3
