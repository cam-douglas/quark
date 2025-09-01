#!/usr/bin/env python3
"""brain/brain_main.py
Unified entry-point that boots the Quark `BrainSimulator` and runs a simple
closed-loop simulation.  This replaces the old split between *embodied* and
*stand-alone* modes – everything now goes through this single file.

The script keeps the external API deliberately minimal so that other tooling
(e.g. CI, notebooks, or external environments) can launch the brain by simply
executing `python -m brain.brain_main`.

Key behaviours
==============
1. Tries to launch a MuJoCo viewer if available **and** the environment
   variable ``QUARK_USE_VIEWER`` is set to ``1``.  Otherwise it runs in
   head-less mode.
2. Automatically enables High-Level Road-map Module (HRM) unless the user sets
   the environment variable ``QUARK_DISABLE_HRM=1``.
3. Provides a tiny command-line interface so developers can adjust run time and
   simulation frequency without digging into the code.

NOTE
----
The file intentionally *avoids* importing the legacy `EmbodiedBrainSimulation`
wrapper so that we have a single, canonical control-loop that directly talks to
`BrainSimulator`.  All helper utilities that are safe to load lazily (e.g.
renderers, curriculum pipelines) are imported inside the main function to keep
startup latency low.
"""
from __future__ import annotations

import os
# ---------------------------------------------------------------------------
# Silence noisy native-library logs (Abseil RAW: Lock blocking … and TensorFlow)
# ---------------------------------------------------------------------------
# These have to be set *before* TensorFlow / Abseil-powered wheels are imported.
os.environ.setdefault("ABSL_LOG_FATAL_THRESHOLD", "4")  # mute RAW mutex contention logs
os.environ.setdefault("ABSL_DIAGNOSTICS", "0")          # disable extra diagnostics, if honoured
# Extra: silence TensorFlow / XLA spam if those libs are pulled in lazily later
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Local imports – kept here to avoid polluting module namespace when imported
# ---------------------------------------------------------------------------
try:
    from brain.core.brain_simulator_init import BrainSimulator
except ImportError as e:  # pragma: no cover – critical failure
    sys.stderr.write(f"[brain_main] 💥 Failed to import BrainSimulator: {e}\n")
    sys.exit(1)

# Optional MuJoCo integration – loaded only if the user asks for a viewer
_MUJOCO_AVAILABLE: bool
try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_AVAILABLE = True
except Exception:  # pragma: no cover – we treat *any* failure as absence
    _MUJOCO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _zero_sensory_stub() -> Dict[str, Any]:
    """Return an all-zero sensory input structure accepted by BrainSimulator."""
    return {
        "sensory_inputs": {"vision": None, "audio": None},
        "reward": 0.0,
        "qpos": np.zeros(23, dtype=np.float32),   # default humanoid DOF
        "qvel": np.zeros(23, dtype=np.float32),
        "is_fallen": False,
        "user_prompt": None,
    }


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Quark brain entry-point")
    parser.add_argument("--steps", type=int, default=10_000, help="Number of simulation steps to run")
    parser.add_argument("--hz", type=float, default=60.0, help="Simulation frequency (steps per second)")
    # Viewer ON by default. Use --no-viewer to disable.
    parser.add_argument(
        "--viewer",
        dest="viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch MuJoCo viewer (default: enabled). Pass --no-viewer to run head-less.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:  # pragma: no cover – CLI wrapper
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Initialise Brain – HRM enabled unless QUARK_DISABLE_HRM=1
    # ------------------------------------------------------------------
    use_hrm = os.getenv("QUARK_DISABLE_HRM", "0") != "1"
    brain = BrainSimulator(use_hrm=use_hrm)

    # ------------------------------------------------------------------
    # Start AutonomousAgent in a background thread to process roadmap goals
    # ------------------------------------------------------------------
    try:
        from state.quark_state_system.autonomous_agent import AutonomousAgent
        from state.quark_state_system.prompt_guardian import PromptGuardian
        from state.quark_state_system.quantum_decision_engine import QuantumDecisionEngine
        from state.quark_state_system.quantum_router import route_computation_intelligently  # noqa: F401 – keep import for side-effects
        # Optional extended utilities
        try:
            import state.quark_state_system.quark_quantum_integration  # noqa: F401
        except ImportError:
            pass

        _agent = AutonomousAgent(workspace_root=os.getcwd())  # project root
        _guardian = PromptGuardian(workspace_root=os.getcwd())
        _q_engine = QuantumDecisionEngine()
        print("🔐 PromptGuardian active; QuantumDecisionEngine initialised.")

        # Optional interactive driver (CLI/REPL) – does not spawn its own loop
        try:
            from state.quark_state_system.quark_driver import QuarkDriver

            _driver = QuarkDriver(workspace_root=os.getcwd())
            print("🚌 QuarkDriver instantiated (process_prompt API available).")
        except Exception as e:
            print(f"⚠️  QuarkDriver unavailable: {e}")

        import threading, time as _t

        def _agent_loop():  # noqa: D401 – simple loop
            """Continuously execute roadmap goals in the background."""
            while True:
                try:
                    # Fetch the current goal title for context
                    current_goal = _agent.roadmap.get_next_actionable_goal()
                    prompt_text = current_goal['task'] if current_goal else "autonomous-cycle"

                    # Validate plan via guardian before execution
                    proposed_action = {"action_type": "roadmap_step", "domain": "planning"}
                    if _guardian.validate_prompt(prompt_text, proposed_action):
                        did_work = _agent.execute_next_goal()
                    else:
                        print("[AutonomousAgent] Guardian rejected action – skipping goal.")
                        did_work = False
                except Exception as e:
                    print(f"[AutonomousAgent] error: {e}")
                    did_work = False
                # If no work performed, back-off for 2 min; else quick 5 s pause
                _t.sleep(5 if did_work else 120)

        threading.Thread(target=_agent_loop, daemon=True, name="AutonomousAgentLoop").start()
        print("🤖 AutonomousAgent background loop active.")
    except Exception as exc:
        print(f"⚠️  AutonomousAgent unavailable: {exc}")

    # ------------------------------------------------------------------
    # Optional Embodiment via MuJoCo – only if user requests and available
    # ------------------------------------------------------------------
    viewer: Optional[mujoco.viewer.Viewer] = None  # type: ignore[attr-defined]
    runner: Optional[SimpleNamespace] = None
    if args.viewer and _MUJOCO_AVAILABLE:
        try:
            from brain.architecture.embodiment.run_mujoco_simulation import MuJoCoRunner

            model_path = os.getenv("QUARK_MODEL_XML")  # path to MJCF model
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(
                    "Set environment variable QUARK_MODEL_XML to a valid MuJoCo XML model if using --viewer"
                )
            runner = MuJoCoRunner(model_path)
            viewer = mujoco.viewer.launch_passive(runner.model, runner.data)
            print("🖥️  MuJoCo viewer launched – running with embodiment.")
        except Exception as exc:
            print(f"⚠️  Failed to launch MuJoCo viewer ({exc}); falling back to head-less mode.")
            viewer = None
            runner = None

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------
    dt = 1.0 / max(args.hz, 1e-6)
    start_time = time.time()
    for step_idx in range(args.steps):
        loop_start = time.time()

        brain_inputs: Dict[str, Any]
        if runner is not None:
            # Embodied mode: read state from MuJoCo
            brain_inputs = {
                **_zero_sensory_stub(),
                "qpos": runner.data.qpos.copy(),
                "qvel": runner.data.qvel.copy(),
            }
        else:
            # Head-less mode
            brain_inputs = _zero_sensory_stub()

        brain_outputs = brain.step(brain_inputs)

        # Send low-level control to embodiment if available
        if runner is not None:
            ctrl = brain_outputs.get("action")
            if ctrl is not None:
                runner.step(ctrl)
            viewer.sync()  # type: ignore[union-attr]

        # Basic console diagnostics every 100 steps
        if step_idx % 100 == 0:
            action = brain_outputs.get("action")
            print(f"step={step_idx:05d} | action_shape={None if action is None else action.shape}")

        # Sleep to maintain target frequency
        elapsed = time.time() - loop_start
        sleep_dur = dt - elapsed
        if sleep_dur > 0:
            time.sleep(sleep_dur)

    total_time = time.time() - start_time
    print(f"✅ Simulation finished – {args.steps} steps in {total_time:.2f}s (avg {(total_time/args.steps):.4f}s/step)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover – entry-point
    main()
