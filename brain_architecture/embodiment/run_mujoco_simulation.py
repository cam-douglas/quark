#!/usr/bin/env python3
"""
Direct MuJoCo Simulation Runner for Quark

This script provides a self-contained way to run and visualize Quark's
embodied agent in a MuJoCo simulation, bypassing the need for a separate
server/client architecture. It directly integrates the cognitive core with
the simulation loop and renders using the native MuJoCo viewer.
"""

import mujoco
import mujoco.viewer
import time
import logging
import os
import numpy as np
import json
import configparser

# Go up three levels from `embodiment` to the project root `quark`
# to ensure correct module resolution.
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from brain_architecture.embodiment.cognitive_core import EmbodiedCognitiveCore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuJoCoRunner:
    """
    Manages the MuJoCo simulation, viewer, and cognitive core interaction.
    """
    def __init__(self, config_path='brain_architecture/embodiment/config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.model_path = self.config.get('Paths', 'ModelPath')

        self.model = None
        self.data = None
        self.viewer = None
        self.cognitive_core = None # Initialize as None
        self.body_id_left_foot = None
        self.body_id_right_foot = None
        self.original_jnt_range = None

    def setup(self):
        """Loads the model and sets up the simulation and viewer."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found at: {self.model_path}")
                return False

            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # Cache foot body IDs for contact/height sensing
            try:
                self.body_id_left_foot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
                self.body_id_right_foot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
            except Exception:
                self.body_id_left_foot = None
                self.body_id_right_foot = None
            
            # Cache original joint ranges to restore later
            import numpy as _np
            self.original_jnt_range = _np.copy(self.model.jnt_range)
            
            # Now that the model is loaded, initialize the cognitive core with correct dimensions
            self.cognitive_core = EmbodiedCognitiveCore(model_path=self.model_path)
            
            logger.info(f"✅ MuJoCo model loaded successfully from {self.model_path}")
            logger.info(f"Model dimensions: nq={self.model.nq}, nv={self.model.nv}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load MuJoCo model: {e}")
            return False

    def _set_curriculum_stage(self, stage: int):
        """Sets the difficulty of the simulation based on the curriculum stage."""
        if self.original_jnt_range is None:
            logging.warning("Original joint ranges not cached; skipping curriculum range changes.")
            return
        
        if stage == 0:
            # Stage 0: Lock all joints except the root freejoint (index 0)
            for i in range(1, len(self.model.jnt_type)):
                self.model.jnt_range[i] = [0, 0]
        elif stage == 1:
            # Restore all, then mildly restrict to focus on hips/knees
            self.model.jnt_range[:] = self.original_jnt_range
        elif stage == 2:
            # Restore all for arms as well
            self.model.jnt_range[:] = self.original_jnt_range
        elif stage == 3:
            # Stage 3: Restore full original ranges for all joints
            self.model.jnt_range[:] = self.original_jnt_range
        else:
            pass
        logging.info(f"📚 Curriculum stage set to {stage}. Joint ranges adjusted.")

    def get_sensory_data(self) -> dict:
        """Constructs a sensory data dictionary from the simulation state."""
        feet = {}
        try:
            if self.body_id_left_foot is not None:
                feet["left_z"] = float(self.data.xpos[self.body_id_left_foot][2])
                feet["left_fz"] = float(self.data.cfrc_ext[self.body_id_left_foot][2])
            if self.body_id_right_foot is not None:
                feet["right_z"] = float(self.data.xpos[self.body_id_right_foot][2])
                feet["right_fz"] = float(self.data.cfrc_ext[self.body_id_right_foot][2])
        except Exception:
            feet = {}
        return {
            "timestamp": time.time(),
            "state_vector": np.concatenate([self.data.qpos, self.data.qvel]).tolist(),
            "model_info": {
                "num_joints": self.model.njnt,
                "num_actuators": self.model.nu,
                "num_bodies": self.model.nbody
            },
            "feet": feet
        }

    def run_simulation(self, duration_sec=300, max_episodes=50):
        """Runs the simulation loop with automatic episode restarts until Quark succeeds."""
        if not self.model or not self.data:
            logger.error("Simulation not set up. Call setup() first.")
            return

        logger.info(f"🚀 Launching MuJoCo viewer and starting continuous learning...")
        logger.info(f"🎯 Objective: Quark learns to walk")
        logger.info(f"⏱️  Max duration: {duration_sec} seconds, Max episodes: {max_episodes}")

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                self.cognitive_core.curriculum_stage = 3
                self._set_curriculum_stage(3) # Full body control
                start_time = time.time()
                episode = 0
                step = 0

                while (self.viewer.is_running() and
                       time.time() - start_time < duration_sec and
                       episode < max_episodes):

                    step_start = time.time()

                    # Check for episode end condition from cognitive core
                    if self.cognitive_core.step_count == 0 and step > 0:
                        logger.info(f"📉 Episode {episode} ended. Resetting simulation...")
                        self._reset_simulation()
                        self._set_curriculum_stage(3)
                        episode = self.cognitive_core.episode_count
                    
                    step += 1

                    # 1. Get sensory data from simulation
                    sensory_data_dict = self.get_sensory_data()
                    sensory_data_str = json.dumps(sensory_data_dict)

                    # 2. Process data and generate motor command from the brain
                    motor_command_str = self.cognitive_core.generate_motor_command_from_sensory_data(sensory_data_str)

                    try:
                        motor_command = json.loads(motor_command_str)
                        if "actuators" in motor_command and "controls" in motor_command["actuators"]:
                            controls = motor_command["actuators"]["controls"]
                            logger.info(f"CONTROLS_SHAPE: {len(controls)}")
                            self.data.ctrl[:self.model.nu] = controls
                            
                            # Log learning progress every 50 steps
                            if self.cognitive_core.step_count % 50 == 0:
                                logger.info(f"Step {self.cognitive_core.step_count}, Episode {self.cognitive_core.episode_count}, Reward: {self.cognitive_core.episode_reward:.2f}")

                    except Exception as e:
                        logger.warning(f"Could not parse or apply motor command: {e}")
                        self.data.ctrl[:] = 0

                    # 3. Step the simulation
                    mujoco.mj_step(self.model, self.data)

                    # 4. Sync the viewer to render the new state
                    self.viewer.sync()

                    # 5. Maintain simulation speed
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                logger.info("⏰ Time limit reached. Quark needs more training time.")
                logger.info(f"📊 Final Learning Stats: {self.cognitive_core.get_learning_stats()}")

        except Exception as e:
            logger.error(f"❌ An error occurred during simulation: {e}")
            logger.error("Please ensure you are running this script using 'mjpython' on macOS.")
        finally:
            if self.viewer:
                self.viewer.close()

    def _reset_simulation(self):
        """Reset the simulation state for a new episode."""
        try:
            # Reset MuJoCo state
            mujoco.mj_resetData(self.model, self.data)

            # Set initial pose for standing
            self.data.qpos[2] = 1.2  # Lower torso height for stability
            self.data.qpos[3:7] = [1, 0, 0, 0]  # Torso orientation (upright)
            # Slight forward pitch to bias forward
            self.data.qpos[4] = 0.05  # quaternion x small angle

            # Neutral legs: small knee flexion
            # Find the indices for the leg joints by name
            r_hip_x_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_x")
            r_hip_y_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_y")
            r_knee_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
            l_hip_x_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_x")
            l_hip_y_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_y")
            l_knee_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
            
            self.data.qpos[r_hip_x_idx] = 0.0
            self.data.qpos[r_hip_y_idx] = 0.0
            self.data.qpos[r_knee_idx] = 0.1
            self.data.qpos[l_hip_x_idx] = 0.0
            self.data.qpos[l_hip_y_idx] = 0.0
            self.data.qpos[l_knee_idx] = 0.1


            # Zero velocities (no initial shove)
            self.data.qvel[:] = 0.0

            # Reset controls
            self.data.ctrl[:] = 0.0

            logger.info("🔄 Simulation reset for new episode")

        except Exception as e:
            logger.error(f"Error resetting simulation: {e}")

def main():
    """Main function to run the simulation."""
    try:
        runner = MuJoCoRunner()
        if runner.setup():
            runner.run_simulation()
    except Exception as e:
        logger.error(f"❌ An error occurred in the main execution block: {e}", exc_info=True)

if __name__ == "__main__":
    main()
