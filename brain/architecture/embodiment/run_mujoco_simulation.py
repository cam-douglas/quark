#!/usr/bin/env python3
"""Direct MuJoCo Simulation Runner for Quark

This script provides a self-contained way to run and visualize Quark's
embodied agent in a MuJoCo simulation, bypassing the need for a separate
server/client architecture. It directly integrates the cognitive core with
the simulation loop and renders using the native MuJoCo viewer.

Integration: Not directly invoked by brain simulator; participates via imports or supporting workflows.
Rationale: Module is used by other components; no standalone simulator hook is required.
"""

import mujoco
import mujoco.viewer
import time
import logging
import os
import configparser

# Go up three levels from `embodiment` to the project root `quark`
# to ensure correct module resolution.
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# The EmbodiedCognitiveCore is no longer needed here, as the brain
# will be provided by the new top-level simulation script.
# from brain_architecture.embodiment.cognitive_core import EmbodiedCognitiveCore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuJoCoRunner:
    """
    Manages the MuJoCo simulation state (model and data).
    """
    def __init__(self, config_path='brain/architecture/embodiment/config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.model_path = self.config.get('Paths', 'ModelPath')

        self.model = None
        self.data = None

        try:
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found at: {self.model_path}")
                raise FileNotFoundError(f"MuJoCo model not found at {self.model_path}")

            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)

            logger.info(f"✅ MuJoCo model loaded successfully from {self.model_path}")
            logger.info(f"Model dimensions: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")

        except Exception as e:
            logger.error(f"❌ Failed to load MuJoCo model: {e}")
            # Re-raise the exception to halt execution if the model can't be loaded.
            raise e

    def step(self, ctrl):
        """Applies control and steps the simulation forward."""
        if self.model is None or self.data is None:
            logger.error("Cannot step: simulation is not initialized.")
            return

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def get_sensory_data(self) -> dict:
        """Constructs a sensory data dictionary from the simulation state."""
        # This method can be simplified or expanded as needed.
        return {
            "timestamp": time.time(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def _reset_simulation(self):
        """Reset the simulation state for a new episode."""
        if not self.model or not self.data:
            return

        try:
            # Reset MuJoCo state
            mujoco.mj_resetData(self.model, self.data)

            # Set initial pose for standing
            self.data.qpos[2] = 1.2  # Lower torso height for stability

            # Zero velocities and controls
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0

            logger.info("🔄 Simulation reset for new episode")

        except Exception as e:
            logger.error(f"Error resetting simulation: {e}")

# The `main` function at the end will be removed as this is now a library.
