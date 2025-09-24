#!/usr/bin/env python3
from __future__ import annotations

# Set environment variables to suppress mutex debugging BEFORE any imports
import os
# More aggressive mutex prevention
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MUJOCO_DISABLE_MUTEXES'] = '1'
os.environ['MUJOCO_DISABLE_TLS'] = '1'
os.environ['MUJOCO_MULTITHREAD'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

# Environment variables already set at top of file

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

from brain.architecture.learning.kpi_monitor import kpi_monitor
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager

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
    parser.add_argument("--steps", type=int, default=float('inf'), help="Number of simulation steps to run (default: INFINITE - runs until manually stopped)")
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
    # Initialize MuJoCo viewer FIRST if requested
    # ------------------------------------------------------------------
    viewer: Optional[mujoco.viewer.Viewer] = None  # type: ignore[attr-defined]
    runner: Optional[SimpleNamespace] = None

    if args.viewer and _MUJOCO_AVAILABLE:
        try:
            from brain.architecture.embodiment.run_mujoco_simulation import MuJoCoRunner

            # Set default model path if not provided
            model_path = os.getenv("QUARK_MODEL_XML")
            if not model_path:
                model_path = "/Users/camdouglas/quark/brain/architecture/embodiment/humanoid.xml"
                os.environ["QUARK_MODEL_XML"] = model_path
                print(f"🎯 Using default MuJoCo model: {model_path}")

            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

            print("🚀 Launching MuJoCo simulation...")
            # MuJoCoRunner expects config file path, not model path
            config_path = "brain/architecture/embodiment/config.ini"
            runner = MuJoCoRunner(config_path)

            # Check if we're running in background (no TTY) or foreground
            # Allow forcing interactive mode via environment variable
            force_interactive = os.getenv("QUARK_FORCE_INTERACTIVE", "0") == "1"
            is_background = not os.isatty(0) and not force_interactive  # Check if stdin is a TTY

            if is_background:
                print("🖥️  Background mode detected - using offscreen rendering")
                viewer = None  # Use offscreen rendering
                # Initialize offscreen renderer for background mode
                runner.renderer = mujoco.Renderer(runner.model, height=480, width=640)
            else:
                print("🖥️  Foreground mode - launching interactive viewer")
                viewer = mujoco.viewer.launch_passive(runner.model, runner.data)
                print("📌 The MuJoCo viewer window should now be visible on your desktop")
                # Allow OpenGL context to settle
                import time as _t
                _t.sleep(2.0)

        except Exception as exc:
            print(f"⚠️  Failed to launch MuJoCo viewer ({exc}); falling back to head-less mode.")
            viewer = None
            runner = None
    elif args.viewer and not _MUJOCO_AVAILABLE:
        print("⚠️  MuJoCo not available; falling back to head-less mode.")

    # ------------------------------------------------------------------
    # Enable AlphaGenome biological rules and simulations
    # ------------------------------------------------------------------
    os.environ['QUARK_DISABLE_ALPHAGENOME'] = '0'  # Enable AlphaGenome
    os.environ['QUARK_FORCE_ALPHAGENOME'] = '1'    # Force biological simulation
    print("🧬 AlphaGenome biological rules and agents ENABLED")

    # ------------------------------------------------------------------
    # Initialise Brain – HRM enabled unless QUARK_DISABLE_HRM=1
    # ------------------------------------------------------------------
    use_hrm = os.getenv("QUARK_DISABLE_HRM", "0") != "1"
    brain = BrainSimulator(use_hrm=use_hrm)

    # ------------------------------------------------------------------
    # Install Brainstem Segmentation Hook (Phase 4 Step 2.O2)
    # ------------------------------------------------------------------
    try:
        from brain.modules.brainstem_segmentation.segmentation_hook import install_segmentation_hook
        segmentation_hook = install_segmentation_hook(brain, auto_segment=True)
        print("🧠 Brainstem segmentation hook installed")

        # Run initial segmentation if hook is available
        if hasattr(segmentation_hook, 'on_brain_initialization'):
            seg_results = segmentation_hook.on_brain_initialization(brain)
            if seg_results.get('segmentation_status') == 'success':
                print("✅ Automatic brainstem segmentation completed on startup")
            elif seg_results.get('segmentation_status') == 'disabled':
                print("ℹ️  Brainstem segmentation disabled")
            else:
                print("⚠️  Brainstem segmentation setup (may complete during simulation)")

    except ImportError as e:
        print(f"⚠️  Brainstem segmentation hook not available: {e}")
    except Exception as e:
        print(f"⚠️  Failed to install brainstem segmentation hook: {e}")

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

        import threading
        import time as _t

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

    # MuJoCo viewer already initialized above

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------
    dt = 1.0 / max(args.hz, 1e-6)
    start_time = time.time()
    step_idx = 0

    def handle_user_interaction():
        """Handle user interaction when Ctrl+C is pressed."""
        print("\n" + "="*50)
        print("🗣️  USER PROMPT")
        print("="*50)
        print("💡 Press Ctrl+C anytime to pause and interact with Quark")
        print("💡 Type 'continue' or press Enter to resume simulation")
        print("💡 Type 'exit' or 'quit' to stop the simulation")
        print("-"*50)

        try:
            # Get user input
            user_input = input("Your message: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                return False  # Signal to exit
            elif user_input.lower() in ['continue', 'resume', '']:
                print("🔄 Resuming simulation...")
                return True  # Signal to continue

            # Process user input through language cortex
            try:
                # Access language cortex from brain
                language_cortex = None
                if hasattr(brain, 'language_cortex'):
                    language_cortex = brain.language_cortex
                elif hasattr(brain, 'modules') and 'language_cortex' in brain.modules:
                    language_cortex = brain.modules['language_cortex']

                if language_cortex and hasattr(language_cortex, 'process_input'):
                    print("🧠 Quark is thinking...")

                    # Temporarily suppress ALL verbose logging during user interaction
                    import logging
                    import sys
                    import warnings
                    from io import StringIO

                    # Suppress all logging, warnings, and stderr during interaction
                    original_level = logging.getLogger().level
                    original_stderr = sys.stderr
                    original_stdout = sys.stdout

                    # Set environment variable to suppress tokenizer warnings
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

                    try:
                        # Suppress only tokenizer warnings, allow model selection logs
                        logging.getLogger('transformers').setLevel(logging.CRITICAL)
                        logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
                        warnings.filterwarnings("ignore", category=UserWarning)

                        # Process input with enhanced model selection
                        print("🧠 Quark is analyzing your message...")
                        response = language_cortex.process_input(user_input)
                        print(f"🤖 Quark: {response}")

                    finally:
                        # Restore original settings
                        logging.getLogger().setLevel(original_level)
                        warnings.resetwarnings()
                else:
                    print("🤖 Quark: I hear you, but my language processing isn't fully connected yet.")
                    print(f"🤖 Quark: You said: '{user_input}' - I'll remember that.")
            except Exception as e:
                print(f"🤖 Quark: I'm having trouble processing that right now. ({e})")
                print(f"🤖 Quark: But I heard you say: '{user_input}'")

            # Ask if user wants to continue or provide more input
            while True:
                next_action = input("\nPress Enter to continue, type 'analytics' for model stats, or type another message: ").strip()
                if next_action == "":
                    print("🔄 Resuming simulation...")
                    break
                elif next_action.lower() == 'analytics':
                    # Show model selection analytics
                    if language_cortex and hasattr(language_cortex, 'print_selection_analytics'):
                        language_cortex.print_selection_analytics()
                    else:
                        print("📊 Analytics not available - language cortex not fully connected")
                elif next_action.lower() in ['exit', 'quit', 'stop']:
                    return False
                else:
                    # Process additional input
                    try:
                        if language_cortex and hasattr(language_cortex, 'process_input'):
                            # Suppress ALL verbose logging for clean interaction
                            original_level = logging.getLogger().level
                            original_stderr = sys.stderr
                            captured_stderr = StringIO()

                            try:
                                logging.getLogger().setLevel(logging.CRITICAL)
                                warnings.filterwarnings("ignore")
                                sys.stderr = captured_stderr

                                response = language_cortex.process_input(next_action)
                                print(f"🤖 Quark: {response}")
                            finally:
                                logging.getLogger().setLevel(original_level)
                                sys.stderr = original_stderr
                                warnings.resetwarnings()
                        else:
                            print(f"🤖 Quark: I acknowledge: '{next_action}'")
                    except Exception as e:
                        print(f"🤖 Quark: I heard you say: '{next_action}' (processing error: {e})")

            return True

        except (EOFError, KeyboardInterrupt):
            print("\n🔄 Resuming simulation...")
            return True

    def run_simulation_step():
        """Run a single simulation step."""
        nonlocal step_idx
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

        from brain.core.brain_simulator_adapter import step_with_metrics  # noqa: WPS433

        brain_outputs = step_with_metrics(brain, brain_inputs)

        # Send low-level control to embodiment if available
        if runner is not None:
            ctrl = brain_outputs.get("action")
            if ctrl is not None:
                runner.step(ctrl)
            # Sync viewer only if it exists (not in offscreen mode)
            if viewer is not None:
                viewer.sync()  # type: ignore[union-attr]

        # Basic console diagnostics every 100 steps
        if step_idx % 100 == 0:
            action = brain_outputs.get("action")
            print(f"step={step_idx:05d} | action_shape={None if action is None else action.shape}")

        # ---------------- KPI Monitoring & Auto-Learning ----------------
        # Expect brain_outputs to include optional "reward" and "loss" keys; if absent skip.
        metrics = {}
        if "reward" in brain_outputs:
            metrics["reward"] = brain_outputs["reward"]
        if "loss" in brain_outputs:
            metrics["loss"] = brain_outputs["loss"]
        if metrics:
            decision = kpi_monitor.update(metrics)
            if decision and step_idx % 500 == 0:  # debounce every 500 steps
                ResourceManager._DEFAULT.run_training_job(decision)

        # Sleep to maintain target frequency
        elapsed = time.time() - loop_start
        sleep_dur = dt - elapsed
        if sleep_dur > 0:
            time.sleep(sleep_dur)

        step_idx += 1

    # Print initial interaction instructions
    print("\n" + "="*60)
    print("🎮 INTERACTIVE BRAIN SIMULATION - INFINITE MODE")
    print("="*60)
    print("💡 Running INDEFINITELY until manually stopped")
    print("💡 Press Ctrl+C anytime to pause and chat with Quark!")
    print("💡 Quark will use its language cortex to respond to you")
    print("💡 Type 'exit', 'quit', or 'stop' to terminate simulation")
    print("="*60)

    # Main interactive simulation loop
    try:
        while step_idx < args.steps:
            run_simulation_step()

    except KeyboardInterrupt:
        # Handle Ctrl+C for user interaction
        should_continue = handle_user_interaction()

        # Continue simulation if user wants to
        while should_continue and step_idx < args.steps:
            try:
                while step_idx < args.steps:
                    run_simulation_step()
            except KeyboardInterrupt:
                # Allow multiple Ctrl+C interactions
                should_continue = handle_user_interaction()
                if not should_continue:
                    break

    total_time = time.time() - start_time
    if step_idx > 0:
        avg_time = total_time / step_idx
        print(f"✅ Simulation finished – {step_idx} steps in {total_time:.2f}s (avg {avg_time:.4f}s/step)")
    else:
        print("✅ Simulation ended")


# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover – entry-point
    main()
