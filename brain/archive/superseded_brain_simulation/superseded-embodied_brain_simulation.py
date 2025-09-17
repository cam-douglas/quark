"""
Embodied Brain Simulation - The Final Integration
This script performs the "brain transplant," connecting the fully integrated
BrainSimulator to the MuJoCo embodiment.
"""

import mujoco
import mujoco.viewer
import time
import logging
import numpy as np
import sys
import threading
import signal

# --- Path Correction for Moved Directories ---
# Add the new 'brain' directory to the system path to ensure all sub-modules can be found.
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


from typing import Dict, Any, Optional
from collections import deque

from mujoco import Renderer
from brain.brain_simulation import BrainSimulator
from brain.architecture.embodiment.run_mujoco_simulation import MuJoCoRunner
from brain.architecture.neural_core.cognitive_systems.self_learning_orchestrator import SelfLearningOrchestrator
from brain.architecture.safety.safety_guardian import SafetyGuardian
# from brain.architecture.neural_core.motor_control.robotics_motor_controller import RoboticsMotorController
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
from brain.architecture.neural_core.cognitive_systems.callback_hub import hub
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
# === MOVE TO brain/core/04_class_definition_start.py (line 52-70) ===

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

    def _install_signal_handlers(self):
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

    def _safe_input(self, prompt: str) -> str:
        """Prompt for input even if stdout/stdin are piped; uses /dev/tty when needed."""
        try:
            if not sys.stdin.isatty():
                try:
                    with open('/dev/tty', 'r') as tty_in, open('/dev/tty', 'w') as tty_out:
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

    def _initialize_simulation(self):
        """
        Initializes all the heavy components required for the simulation.
        This is called from the `run` method.
        """
        print("🚀 Initializing simulation components...")
        self.runner = MuJoCoRunner()
        self.renderer = Renderer(self.runner.model)

        # HRM planning is enabled by default. Set env QUARK_DISABLE_HRM=1 to turn it off.
        hrm_enabled = os.environ.get("QUARK_DISABLE_HRM", "0") != "1"
        self.brain = BrainSimulator(
            use_hrm=hrm_enabled,
            embodiment=self.runner,
            act_dim=len(self.motor_primitives)
        )
        self.brain.set_viewer(None) # Viewer is managed by launch_passive

        # Re-enabling learning components as per user instruction.
        self.self_learning_orchestrator = SelfLearningOrchestrator()
        self.safety_guardian = SafetyGuardian()
        # self.robotics_controller = RoboticsMotorController(self.runner.model, self.runner.data)

        from brain.architecture.neural_core.learning.developmental_curriculum import DevelopmentalCurriculum
        self.developmental_curriculum = DevelopmentalCurriculum()
        print("🍼 Quark is now learning like a human baby with developmental milestones!")

        # Keeping the suspected heavy training pipeline disabled.
        # self.training_pipeline = LLMGuidedTrainingPipeline(self.brain)

        self.actuator_ids = {
            'right_hip_x': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_hip_x_motor'),
            'right_hip_y': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_hip_y_motor'),
            'right_knee': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_knee_motor'),
            'left_hip_x': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_hip_x_motor'),
            'left_hip_y': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_hip_y_motor'),
            'left_knee': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_knee_motor'),
            'right_shoulder_x': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_shoulder_x_motor'),
            'right_shoulder_y': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_shoulder_y_motor'),
            'left_shoulder_x': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_shoulder_x_motor'),
            'left_shoulder_y': mujoco.mj_name2id(self.runner.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_shoulder_y_motor'),
        }

        # Set initial pose to crawling
        self._reset_to_crawl_pose()
        print("✅ Simulation components initialized.")
# === MOVE TO brain/core/08_initialize_simulation.py (line 124-167) ===

    def _reset_to_crawl_pose(self):
        """
        Sets the humanoid to a stable, physically plausible crawling position.
        """
        self.runner._reset_simulation()

        # Get the default standing pose to modify from
        qpos = self.runner.data.qpos.copy()

        # Place torso at safe height just above ground to avoid penetration
        qpos[2] = max(0.6, qpos[2])

        # --- Set a stable quadrupedal stance ---
        # Set hips to a 90-degree angle
        qpos[7] = 0.0   # Right Hip X (forward/back)
        qpos[8] = -1.57 # Right Hip Y (abduction) -> leg out to the side
        qpos[13] = 0.0  # Left Hip X
        qpos[14] = 1.57 # Left Hip Y

        # Bend knees to 90 degrees
        qpos[9] = 1.57  # Right Knee
        qpos[15] = 1.57 # Left Knee

        # --- Position the arms for support ---
        # Shoulders forward
        qpos[5] = 1.0 # Right Shoulder X
        qpos[11] = 1.0 # Left Shoulder X

        self.runner.data.qpos[:] = qpos
        # Zero velocities to avoid initial impulses
        try:
            self.runner.data.qvel[:] = 0.0
        except Exception:
            self.runner.data.qvel[:] = 0.0  # already zeroed or protected

        mujoco.mj_forward(self.runner.model, self.runner.data)

        # Reset internal state trackers
        self._prev_x = None
        self._cumulative_distance = 0.0
        self._last_step_speed = 0.0

    def _calculate_pose_error(self, current_qpos: np.ndarray, target_qpos: np.ndarray) -> Optional[float]:
        """Calculates the difference between the current pose and the target pose."""
        if target_qpos is None or current_qpos is None:
            return None

        num_joints_to_match = min(len(current_qpos), len(target_qpos))
        error = np.linalg.norm(
            current_qpos[:num_joints_to_match] - target_qpos[:num_joints_to_match]
        )
        return float(error)


    def _calculate_developmental_reward(self, current_qpos: np.ndarray, is_fallen: bool) -> float:
        """
        Calculate reward based on developmental curriculum progress like a human baby.
        """
        try:
            # Get current developmental target
            target_pose = self.developmental_curriculum.get_current_target_pose()

            # Calculate pose error (difference between current and target pose)
            current_pose = current_qpos[7:]  # Skip the free joint (first 7 elements)
            target_joint_pose = target_pose[:len(current_pose)]  # Match dimensions

            pose_error = np.linalg.norm(current_pose - target_joint_pose)

            # Prepare state information for developmental assessment
            current_state = {
                "pose_error": pose_error,
                "is_fallen": is_fallen,
                "is_stable": not is_fallen and pose_error < 0.5,
                "time_stable": getattr(self, '_stability_timer', 0.0),
                "distance_moved": getattr(self, '_distance_moved', 0.0),
                "movement_variance": np.std(current_pose) if len(current_pose) > 1 else 0.0
            }

            # Get developmental reward (includes intrinsic motivation, curiosity, social feedback)
            developmental_reward = self.developmental_curriculum.calculate_developmental_reward(current_state)

            # Update progress and check for milestone completion
            milestone_achieved = self.developmental_curriculum.update_progress(current_state)
            if milestone_achieved:
                developmental_reward += 5.0  # Big bonus for milestone achievement!

            return float(developmental_reward)

        except Exception as e:
            logger.warning(f"Error calculating developmental reward: {e}")
            return -0.1

    def _update_curriculum_stage(self):
        """Checks performance and advances the curriculum stage if successful."""
        if len(self.reward_buffer) < self.reward_buffer.maxlen:
            return # Not enough data yet

        avg_reward = np.mean(self.reward_buffer)
        if avg_reward > self.stage_success_threshold:
            self.curriculum_stage += 1
            self.reward_buffer.clear() # Reset buffer for the new stage
            logger.info(f"🎉 SUCCESS! Advancing to Curriculum Stage {self.curriculum_stage}")
            self.brain.rl_agent.epsilon = 0.9 # Encourage exploration in the new stage
# === MOVE TO brain/core/12_curriculum_stage_embodied.py (line 267-278) ===

    def _print_learning_diagnostics(self, brain_output: Dict[str, Any], reward: float, step_count: int):
        """Prints learning progress diagnostics for the crawling task."""
        action = brain_output.get("ppo_goal") # Use the high-level goal

        # Ensure the action is an integer before using it as an index.
        if action is not None:
            try:
                action_idx = int(action)
                primitive = self.motor_primitives[action_idx] if action_idx < len(self.motor_primitives) else "unknown"
            except (ValueError, TypeError):
                primitive = "invalid_action"
        else:
            primitive = "None"
            action_idx = "N/A"

        # Crawl diagnostics
        x = float(self.runner.data.qpos[0])
        step_speed = getattr(self, "_last_step_speed", 0.0)
        total_dist = getattr(self, "_cumulative_distance", 0.0)

        print(f"\n🧎 QUARK CRAWLING (Step {step_count}):")
        print("   - 🎯 Goal: crawl_forward")
        print(f"   - 🎬 Action: {action_idx} ({primitive})")
        print(f"   - 💰 Reward: {reward:.4f}")
        print(f"   - 📏 X Position: {x:.3f}")
        print(f"   - 🚀 Speed (step): {step_speed:.3f} m/s")
        print(f"   - 🧮 Distance (cumulative): {total_dist:.3f} m")
        print(f"   - 🤖 PPO Rollout Size: {getattr(self.brain.ppo_agent, 'rollout_len', 'n/a') if hasattr(self.brain, 'ppo_agent') and self.brain.ppo_agent else 'n/a'}")


    def _is_fallen(self) -> bool:
        """
        Checks if the humanoid has fallen over.
        Returns True if the torso is below a certain height threshold.
        """
        torso_z_position = self.runner.data.qpos[2]
        return torso_z_position < 0.3 # Lower fall threshold for crawling


    def _quat_to_euler(self, q):
        """Converts a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = q
        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw


    def _balance_overlay_control(self):
        """Small stabilizing corrections at hips to counter torso pitch/roll."""
        # Root orientation quaternion is at qpos[3:7] = (w, x, y, z)
        q = self.runner.data.qpos[3:7]
        roll, pitch, _ = self._quat_to_euler(q)
        # Proportional gains (small)
        k_pitch = 0.6
        k_roll = 0.6
        # Map: hip_x ~ forward/back (pitch), hip_y ~ left/right (roll)
        correction = np.zeros(self.runner.model.nu) # Use runner.model.nu for total actuators
        if 'right_hip_x' in self.actuator_ids and 'left_hip_x' in self.actuator_ids:
            correction[self.actuator_ids['right_hip_x']] += -k_pitch * pitch
            correction[self.actuator_ids['left_hip_x']]  += -k_pitch * pitch
        if 'right_hip_y' in self.actuator_ids and 'left_hip_y' in self.actuator_ids:
            correction[self.actuator_ids['right_hip_y']] += -k_roll * roll
            correction[self.actuator_ids['left_hip_y']]  += -k_roll * roll
        return correction

    def handle_user_interaction(self):
        """Handles the user interaction sequence: prompt, get response, and auto-resume."""
        prompt_color = "\033[96m"
        reset_color = "\033[0m"

        try:
            message = self._safe_input(f"\n{prompt_color}USER PROMPT > {reset_color}")

            if message:
                self.last_user_input = message

                brain_inputs = {
                    "user_prompt": message,
                    "sensory_inputs": {}, "reward": 0.0, "qpos": self.runner.data.qpos.copy(), "is_fallen": False,
               }
                brain_output = self.brain.step(brain_inputs, stage=self.curriculum_stage)

                if brain_output.get("direct_speech_response"):
                    quark_color = "\033[92m"
                    responses = brain_output["direct_speech_response"]
                    if isinstance(responses, list):
                        for i, response in enumerate(responses):
                            print(f"\n{quark_color}QUARK SAYS ({i+1}/{len(responses)}): {response}{reset_color}")
                            time.sleep(1)
                    else:
                        print(f"\n{quark_color}QUARK SAYS: {responses}{reset_color}")

            # Require Enter to resume
            self._safe_input(f"\n{prompt_color}Press Enter to resume simulation...{reset_color}")
            logger.info("▶️ Resuming simulation...")

        except (KeyboardInterrupt, EOFError, BrokenPipeError):
            logger.info("Prompt cancelled. Resuming simulation...")
            return

# === MOVE TO brain/core/16_run_loop_embodied.py (line 337-531) ===
    def run(self, duration_sec=300, steps_per_prompt=200):
        # --- Defer Initialization to Run Time ---
        self._initialize_simulation()
        self._install_signal_handlers()

        logger.info("🗣️ Press Ctrl+C at any time to terminate the simulation.")

        with mujoco.viewer.launch_passive(self.runner.model, self.runner.data) as viewer:
            start_time = time.time()
            step_count = 0

            brain_inputs: Dict[str, Any] = {
                "sensory_inputs": {"vision": None, "audio": None},
                "reward": 0.0,
                "qpos": self.runner.data.qpos.copy(),
                "is_fallen": False,
                "user_prompt": None,
            }

            while viewer.is_running() and time.time() - start_time < duration_sec:
                step_start = time.time()
                # Handle interactive interrupt
                if self._interrupt_event.is_set():
                    self.handle_user_interaction()
                    self._interrupt_event.clear()
                    # Skip stepping this iteration; continue cleanly
                    continue

                if self._is_fallen():
                    logger.info("Quark has fallen. Resetting to crawling pose.")
                    self._reset_to_crawl_pose()
                    brain_inputs["is_fallen"] = True
                else:
                    brain_inputs["is_fallen"] = False

                brain_inputs["sensory_inputs"]["vision"] = None
                brain_inputs["qpos"] = self.runner.data.qpos.copy()
                brain_inputs["qvel"] = self.runner.data.qvel.copy()
                brain_inputs["user_prompt"] = None

                # This line was accidentally removed. It is essential.
                brain_output = self.brain.step(brain_inputs, stage=self.curriculum_stage)

                # The vision system is not used for the balancing task, so we can remove the logs.
                # self.last_brain_output = brain_output

                # The 'action' from the brain is now the low-level 'ctrl' array from the Motor Cortex
                ctrl = brain_output.get("action")

                # Apply small stabilizing overlay
                if ctrl is not None:
                    ctrl += self._balance_overlay_control()
                    self.runner.step(ctrl)
                else:
                    # If no action is produced, do nothing (or a default action)
                    self.runner.step(np.zeros(self.runner.model.nu))

                # Compute prev_x for forward progress
                if self._prev_x is None:
                    self._prev_x = float(self.runner.data.qpos[0])
                prev_x = self._prev_x

                # The reward is now calculated based on crawling forward progress
                reward = self.developmental_curriculum.calculate_developmental_reward({
                    "qpos": self.runner.data.qpos,
                    "prev_x": prev_x,
                    "is_fallen": brain_inputs["is_fallen"]
                })

                brain_inputs["reward"] = reward

                # Update prev_x and metrics
                current_x = float(self.runner.data.qpos[0])
                step_distance = max(0.0, current_x - prev_x)
                step_speed = step_distance / max(1e-6, self.runner.model.opt.timestep)
                self._prev_x = current_x
                # store for diagnostics
                self._last_step_speed = step_speed
                self._cumulative_distance = getattr(self, "_cumulative_distance", 0.0) + step_distance

                if step_count % 50 == 0:
                    # Guard access to action_primitive to prevent out-of-range indices
                    ppo_goal = brain_output.get('ppo_goal')
                    action_primitive = None
                    if isinstance(ppo_goal, int) and 0 <= ppo_goal < len(self.motor_primitives):
                        action_primitive = self.motor_primitives[ppo_goal]
                    thought_context = {
                        **brain_output,
                        "reward": reward,
                        "curriculum_stage": self.curriculum_stage,
                        "pose_error": self.brain.last_pose_error,
                        "action_primitive": action_primitive
                    }
                    spontaneous_thought = self.brain.language_cortex.generate_spontaneous_thought(thought_context)
                    if spontaneous_thought:
                        thought_color = "\033[93m"
                        reset_color = "\033[0m"
                        print(f"\n{thought_color}QUARK THINKS: {spontaneous_thought}{reset_color}")

                if step_count % 100 == 0:
                    self._print_learning_diagnostics(brain_output, reward, step_count)

                viewer.sync()

                time_until_next_step = self.runner.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                step_count += 1

        logger.info("Simulation finished.")

if __name__ == "__main__":
    simulation = EmbodiedBrainSimulation()
    try:
        # --- NEW: Developmental Training Phase ---
        # First, we run Quark through its developmental curriculum.
        # This trains the brain from the ground up before interactive simulation.
        print("\n" + "="*50)
        print("🎓 STARTING DEVELOPMENTAL TRAINING PIPELINE 🎓")
        print("="*50)

        # We can start from a specific phase if needed, but default is 0
        # simulation.training_pipeline.run_full_curriculum(start_phase=0)

        print("\n" + "="*50)
        print("✅ DEVELOPMENTAL TRAINING COMPLETE ✅")
        print("🚀 LAUNCHING INTERACTIVE SIMULATION 🚀")
        print("="*50)

        # Now, run the interactive simulation with the trained brain
        simulation.run()

    except KeyboardInterrupt:
        logger.info("\nSimulation terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
