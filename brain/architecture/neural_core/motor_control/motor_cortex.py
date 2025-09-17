

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
import logging
import os
import random
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

class MotorCortex:
    """
    Loads and provides expert motion data from a local AMASS dataset,
    now with support for a curriculum.
    """
    def __init__(self, amass_data_path: str):
        self.data_path = amass_data_path
        self.motion_files = self._find_motion_files()
        self._last_qpos: Optional[np.ndarray] = None
        self._policy: Optional[nn.Module] = None
        self._load_policy_if_available()

        # --- NEW: Pre-sort motions for curriculum learning ---
        self.standing_motions = [f for f in self.motion_files if "stand" in f.lower() or "balance" in f.lower()]
        self.simple_motions = [f for f in self.motion_files if any(kw in f.lower() for kw in ["walk", "jog", "step"])]

        self.current_motion_data = None
        self.current_frame = 0

        if not self.motion_files:
            logger.warning(f"No .npz motion files found in {self.data_path}. Falling back to learned policy and primitives.")
        else:
            print(f"🧠 Motor Cortex initialized. Found {len(self.motion_files)} motion files.")
            print(f"   -> Found {len(self.standing_motions)} standing motions for curriculum Stage 0.")
            self._load_next_motion(stage=0) # Start with a balancing task

    def _load_policy_if_available(self):
        """Attempt to load a supervised imitation policy if present."""
        if not _TORCH_OK:
            return
        policy_path = os.environ.get(
            "QUARK_MOTOR_POLICY",
            "/Users/camdouglas/quark/data/datasets/synth_crawl/policy_supervised.pt"
        )
        if not os.path.isfile(policy_path):
            return
        class TinyPolicy(nn.Module):
            def __init__(self, obs_dim: int = 24, act_dim: int = 18):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 128), nn.ReLU(),
                    nn.Linear(128, 128), nn.ReLU(),
                    nn.Linear(128, act_dim)
                )
            def forward(self, x):
                return self.net(x)
        try:
            policy = TinyPolicy(obs_dim=24, act_dim=18)
            state = torch.load(policy_path, map_location="cpu")
            policy.load_state_dict(state)
            policy.eval()
            self._policy = policy
            logger.info(f"Loaded Motor Cortex policy from {policy_path}")
        except Exception as e:
            logger.warning(f"Failed to load Motor Cortex policy: {e}")

    def _find_motion_files(self) -> List[str]:
        """Recursively finds all .npz files in the data path."""
        npz_files = []
        if not os.path.isdir(self.data_path):
            logger.error(f"AMASS data path does not exist: {self.data_path}")
            return npz_files

        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".npz"):
                    npz_files.append(os.path.join(root, file))
        return npz_files

    def _load_next_motion(self, stage: int = 0):
        """Loads a new motion file appropriate for the curriculum stage."""
        if not self.motion_files:
            return

        motion_file = None
        if stage == 0 and self.standing_motions:
            # Stage 0: Focus on standing/balancing
            motion_file = random.choice(self.standing_motions)
        elif stage == 1 and self.simple_motions:
            # Stage 1: Simple walking/stepping motions
            motion_file = random.choice(self.simple_motions)
        else:
            # Stage 2 or fallback: any motion
            motion_file = random.choice(self.motion_files)

        try:
            data = np.load(motion_file)
            # The 'poses' key contains the body pose parameters.
            # The shape is (num_frames, 156), representing joint rotations.
            self.current_motion_data = data['poses']
            self.current_frame = 0
            logger.info(f"Loaded new expert motion: {os.path.basename(motion_file)} ({len(self.current_motion_data)} frames)")
        except Exception as e:
            logger.error(f"Error loading motion file {motion_file}: {e}")
            self.current_motion_data = None

    def get_target_pose(self, stage: int = 0) -> Optional[np.ndarray]:
        """Returns the target pose, aware of the curriculum stage."""
        if self.current_motion_data is None:
            return None

        # For early stages, we can hold a single pose for longer to make it easier
        if stage == 0: # Stage 0: Baby-like learning - just basic stability
            # Return a very simple neutral standing pose (not complex AMASS data)
            neutral_pose = np.zeros(156)  # All joints neutral
            return neutral_pose

        if self.current_frame >= len(self.current_motion_data):
            self._load_next_motion(stage)
            if self.current_motion_data is None:
                return None

        target_pose = self.current_motion_data[self.current_frame]
        self.current_frame += 1
        return target_pose

    def step(self, ppo_goal: int, hrm_subgoal: Optional[Dict] = None, current_qpos: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        The main step function, now generating low-level motor commands based on high-level goals.

        Args:
            ppo_goal (int): The high-level goal from the PPO agent (e.g., 0 for 'crawl_forward').
            hrm_subgoal (Optional[Dict]): The long-term subgoal from the HRM planner.
            current_qpos (Optional[np.ndarray]): The current joint positions of the embodiment.

        Returns:
            A dictionary containing the low-level control signals ('ctrl') and the target pose.
        """
        # For now, we'll use a simple mapping from PPO goal to a motor primitive concept
        # This will be replaced with more sophisticated goal translation logic.

        # This is a placeholder for the logic that would translate the goal
        # into a sequence of motor commands. For now, we'll use a simplified version
        # of the logic that was in `_execute_motor_primitive`.

        num_actuators = 16 # discover from model if available
        ctrl = np.zeros(num_actuators, dtype=np.float32)

        # If a learned policy is available, use it to generate a proposal
        if self._policy is not None and _TORCH_OK and current_qpos is not None:
            try:
                # Build 24-D observation: root vel (3) + joint vels (~act_dim) + pad to 24
                # Approximate root vel from base link translation if available; fall back to zeros
                root_vel = np.zeros(3, dtype=np.float32)
                # Determine available joint count from qpos length
                total_qpos = current_qpos.shape[0]
                start_idx = 7
                avail = max(0, total_qpos - start_idx)
                use_n = min(18, avail)
                if self._last_qpos is None or self._last_qpos.shape[0] != total_qpos:
                    joint_vels = np.zeros(use_n, dtype=np.float32)
                else:
                    joint_vels = (current_qpos[start_idx:start_idx+use_n] - self._last_qpos[start_idx:start_idx+use_n]).astype(np.float32)
                obs_core = np.concatenate([root_vel, joint_vels], axis=0)
                if obs_core.shape[0] < 24:
                    obs = np.pad(obs_core, (0, 24 - obs_core.shape[0]))
                else:
                    obs = obs_core[:24]
                with torch.no_grad():
                    act = self._policy(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
                # Map 18-dim action to 16 actuators (truncate extras)
                ctrl = act[:num_actuators].astype(np.float32)
                # Scale down for safety
                ctrl = np.clip(ctrl * 0.5, -1.0, 1.0)
            except Exception as e:
                logger.warning(f"Policy step failed, fallback to primitives: {e}")

        # Fallback primitive if no policy output
        if not np.any(ctrl):
            s = 0.5
            if ppo_goal == 0: # crawl_contralateral_forward
                ctrl[8] = s   # right_shoulder_y
                ctrl[4] = -s  # left_hip_x
            elif ppo_goal == 1: # crawl_contralateral_backward
                ctrl[8] = -s
                ctrl[4] = s
            elif ppo_goal == 4: # push_off
                ctrl[3] = -s  # right_knee
                ctrl[6] = -s  # left_knee
                ctrl[8] = -s  # right_shoulder_y
                ctrl[11] = -s # left_shoulder_y

        # In the future, the hrm_subgoal and AMASS data would be used here to
        # generate much more complex and nuanced movements via IK or trajectory optimization.

        target_pose = self.get_target_pose(stage=0) # Use a simple stage for now
        # Cache current qpos for velocity calc next step
        if current_qpos is not None:
            self._last_qpos = current_qpos.copy()

        return {"target_pose": target_pose, "ctrl": ctrl}
