

import logging
import sys

# --- Path Correction for Moved Directories ---
# Add the new 'brain' directory to the system path to ensure all sub-modules can be found.
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))



# from brain.architecture.neural_core.motor_control.robotics_motor_controller import RoboticsMotorController
from brain.tools.goal_poll import log_next_goal
log_next_goal("[Roadmap|Embodied]")
# === MOVE TO brain/core/01_header_imports_embodied.py (line 1-36) ===

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def user_input_thread(q):
    """A silent thread that waits for the user to press Enter and signals an interrupt."""
    while True:
        try:
            # This is a blocking call, but it's in a separate thread.
            input()
            q.put("INTERRUPT")
        except (KeyboardInterrupt, EOFError):
            q.put("exit")
            break
        # === MOVE TO brain/core/03_user_input_thread.py (line 41-50) ===

class EmbodiedBrainSimulation:
    # --- SIMPLIFIED: Single Flag Approach ---

    def __init__(self):
        """
        Initializes the EmbodiedBrainSimulation with a minimal set of parameters.
        Heavy components are now initialized in the `run` method to prevent startup freezes.
        """
        self.motor_primitives = [
            "crawl_contralateral_forward",
            "crawl_contralateral_backward",
            "crawl_homolateral_forward",
            "crawl_homolateral_backward",
            "push_off",
            "hold_steady"
        ]
        self.last_user_input = "artificial intelligence"
