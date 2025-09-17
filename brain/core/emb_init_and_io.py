from collections import deque
from typing import Dict, Any
import threading
import sys
import signal
import logging

from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
from brain.architecture.neural_core.cognitive_systems.callback_hub import hub

logger = logging.getLogger(__name__)


def setup_emb_init_and_io(self) -> None:
    # These will be initialized in the run method
    self.runner = None
    self.renderer = None
    self.brain = None
    self.self_learning_orchestrator = None
    self.safety_guardian = None
    self.robotics_controller = None
    self.developmental_curriculum = None
    self.training_pipeline = None
    self.actuator_ids = {}
    self.curriculum_stage = 0
    self.reward_buffer = deque(maxlen=200)
    self.stage_success_threshold = 0.8
    self.action_scale = 0.4
    self.last_brain_output: Dict[str, Any] = {}
    self._prev_x = None
    self._cumulative_distance = 0.0
    self._last_step_speed = 0.0
    # Interrupt handling
    self._interrupt_event = threading.Event()
    self.resource_manager = ResourceManager(auto_scan=False)

    def _on_resource(event, data):
        logger.info("[EmbodiedSim] Event %s: %s", event, data)

    hub.register(_on_resource)

    # ----------------------------------------------------------------------
    # signal-handler helper, preserved exactly
    # ----------------------------------------------------------------------
    def _install_signal_handlers():
        """Install SIGINT (Ctrl+C) handler to trigger interactive prompt."""
        def _on_sigint(signum, frame):
            # Set flag; main loop will handle prompt synchronously
            self._interrupt_event.set()

        signal.signal(signal.SIGINT, _on_sigint)
        # Avoid interrupting system calls like input()
        try:
            signal.siginterrupt(signal.SIGINT, False)
        except Exception:
            pass

    self._install_signal_handlers = _install_signal_handlers

    # ----------------------------------------------------------------------
    # safe-input helper, preserved exactly
    # ----------------------------------------------------------------------
    def _safe_input(prompt: str) -> str:
        """Prompt for input even if stdout/stdin are piped; uses /dev/tty when needed."""
        try:
            if not sys.stdin.isatty():
                try:
                    with open("/dev/tty", "r") as tty_in, open("/dev/tty", "w") as tty_out:
                        tty_out.write(prompt)
                        tty_out.flush()
                        line = tty_in.readline()
                        return "" if line is None else line.rstrip("\n")
                except Exception:
                    # Fallback to regular input
                    return input(prompt)
            return input(prompt)
        except (EOFError, BrokenPipeError):
            return ""

    self._safe_input = _safe_input
