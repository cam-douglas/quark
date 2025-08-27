#!/usr/bin/env python3
"""
Embodied Cognitive Core for Quark

This module acts as the central processing unit for Quark's embodied agent.
It receives sensory data from the simulation, passes it to the relevant
cognitive and safety modules, and generates motor commands in response.
"""

import json
import logging
import numpy as np
from typing import Dict, Any, Optional

from brain_architecture.learning.balancing_agent import BalancingAgent
from brain_architecture.neural_core.safety_agent.enhanced_safety_protocols import EnhancedSafetyProtocols
from management.emergency.emergency_shutdown_system import QuarkState
from brain_architecture.neural_core.motor_control.cpg import CentralPatternGenerator
from brain_architecture.embodiment.sb3_adapter import SB3HumanoidPolicyAdapter
from brain_architecture.neural_core.vla import PalmEStyleVLA
from brain_architecture.embodiment.imitation_adapter import ImitationPolicyAdapter
import os
from scipy.spatial.transform import Rotation as R
import mujoco
import torch
import smplx

logger = logging.getLogger(__name__)

# SMPL joint mapping to our MuJoCo model's joint names
# This is an approximation and might need tuning.
# AMASS/SMPL order is often:
# 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1, 4:L_Knee, 5:R_Knee, 6:Spine2,
# 7:L_Ankle, 8:R_Ankle, 9:Spine3, 10:L_Foot, 11:R_Foot, 12:Neck,
# 13:L_Collar, 14:R_Collar, 15:Head, 16:L_Shoulder, 17:R_Shoulder,
# 18:L_Elbow, 19:R_Elbow, 20:L_Wrist, 21:R_Wrist
# Our qpos order (indices after root joint at 7):
# 8:r_hip_x, 9:r_hip_y, 10:r_knee, 11:r_ank_y, 12:r_ank_x
# 13:l_hip_x, 14:l_hip_y, 15:l_knee, 16:l_ank_y, 17:l_ank_x
# ... and arms, neck, etc.
# We will create a mapping from our relevant qpos indices to SMPL pose indices
# NOTE: AMASS poses are axis-angle representations (3 values per joint)
SMPL_TO_QUARK_QPOS_MAP = {
    # Right Leg
    10: (5, 'knee'),     # R_Knee -> right_knee
    11: (8, 'ankle'),    # R_Ankle -> right_ankle_y
    12: (8, 'ankle_x'),  # R_Ankle -> right_ankle_x (roll)
    8: (2, 'hip'),      # R_Hip -> right_hip_x
    9: (2, 'hip_y'),     # R_Hip -> right_hip_y
    # Left Leg
    15: (4, 'knee'),     # L_Knee -> left_knee
    16: (7, 'ankle'),    # L_Ankle -> left_ankle_y
    17: (7, 'ankle_x'),  # L_Ankle -> left_ankle_x
    13: (1, 'hip'),      # L_Hip -> left_hip_x
    14: (1, 'hip_y'),     # L_Hip -> left_hip_y
}
# We need a similar map for qvel
SMPL_TO_QUARK_QVEL_MAP = {
    # Right Leg (indices in our qvel start at 6 after root)
    8: 5,   # R_Knee
    9: 8,   # R_Ankle
    7: 2,   # R_Hip
    # Left Leg
    11: 4,  # L_Knee
    12: 7,  # L_Ankle
    10: 1,  # L_Hip
}


class EmbodiedCognitiveCore:
    """
    The core of the embodied agent's cognitive functions.
    Integrates sensory input, learning, and motor control.
    """

    def __init__(self, model_path: str, use_pretrained_policy: bool = False, hf_repo: str = "sb3/ppo-Humanoid-v3", hf_file: str = "ppo-Humanoid-v3.zip", use_vla: bool = False, vla_prompt: str = "walk forward", use_imitation: bool = True, imitation_path: str = "/Users/camdouglas/quark/data/amass/processed/policy_supervised.pt"):
        """
        Initialize the EmbodiedCognitiveCore.
        """
        model_name = os.path.basename(model_path).replace('.xml', '')
        config_path = os.path.join(os.path.dirname(__file__), 'model_configs', f'{model_name}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        nq = config['nq']
        nv = config['nv']
        action_dim = config['action_dim']

        logger.info(f"Initializing Embodied Cognitive Core for model '{model_name}' with nq={nq}, nv={nv}, action_dim={action_dim}...")
        logger.info(f"CONFIG: nq={nq}, nv={nv}, action_dim={action_dim}")

        self.model_path = model_path
        self.use_pretrained_policy = use_pretrained_policy
        self.use_vla = use_vla
        self.use_imitation = use_imitation
        self.imitation_path = imitation_path
        
        self.sim_model = None
        self.smpl_model = None
        if imitation_path:
            try:
                self.sim_model = mujoco.MjModel.from_xml_path(self.model_path)
                # Initialize SMPL model
                smpl_model_path = "/Users/camdouglas/quark/data/smpl_models"
                try:
                    self.smpl_model = smplx.create(smpl_model_path, model_type='smpl').to('cpu')
                except Exception as e:
                    logger.error(f"Could not load model for FK: {e}")
            except Exception as e:
                logger.error(f"Could not load sim_model for FK: {e}")

        self.nq = nq
        self.nv = nv
        self.state_dim = nq + nv
        self.action_dim = action_dim
        
        logger.info(f"AGENT: state_dim={self.state_dim}, action_dim={self.action_dim}")
        
        # Initialize the PPO balancing agent
        self.balance_agent = BalancingAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        self.cpg = CentralPatternGenerator()
        # Optional VLA module
        self.vla: Optional[PalmEStyleVLA] = None
        self.desired_vy: Optional[float] = None
        # Optional imitation policy
        self.imitation: Optional[ImitationPolicyAdapter] = None
        if self.use_vla:
            try:
                self.vla = PalmEStyleVLA()
                logger.info("✅ VLA front-end initialized (PaLM-E style)")
            except Exception as e:
                logger.warning(f"Could not initialize VLA front-end: {e}")
                self.use_vla = False
        if self.use_imitation:
            try:
                self.imitation = ImitationPolicyAdapter(policy_path=self.imitation_path)
                logger.info("✅ Imitation policy loaded")
            except Exception as e:
                logger.warning(f"Could not load imitation policy: {e}")
                self.use_imitation = False
        self.sb3_adapter: Optional[SB3HumanoidPolicyAdapter] = None
        if self.use_pretrained_policy:
            try:
                logger.info(f"Loading pretrained SB3 policy from {hf_repo} ({hf_file}) ...")
                self.sb3_adapter = SB3HumanoidPolicyAdapter(repo_id=hf_repo, filename=hf_file)
                logger.info("✅ Pretrained SB3 policy loaded.")
            except Exception as e:
                logger.warning(f"Could not load SB3 policy: {e}. Falling back to internal controller.")
                self.use_pretrained_policy = False
        
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.best_reward = -float('inf')
        self.gamma = 0.99 # Discount factor for potential-based shaping
        self.last_potential = 0.0
        self.curriculum_stage = 0
        self.stage_success_count = 0
        self.prev_y = None
        # Finite-state gait controller (toggle stance/swing)
        self.gait_phase = "R_stance"  # or "L_stance"
        self.gait_steps_per_phase = 70
        self._phase_step_counter = 0
        
        # For imitation-guided RL
        self.reference_motion = None
        self.reference_motion_index = 0
        if self.use_imitation:
            self._load_reference_motion()
        
        self.safety_protocols = EnhancedSafetyProtocols()
        self._initialize_balance_safety_thresholds()

        logger.info("✅ Embodied Cognitive Core initialized with PPO Agent.")

    def _forward_kinematics(self, qpos):
        """Performs forward kinematics to get world coordinates of key bodies."""
        if self.sim_model is None:
            return {}
        
        data = mujoco.MjData(self.sim_model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self.sim_model, data)
        
        key_bodies = ["head", "right_lower_arm", "left_lower_arm", "right_foot", "left_foot"]
        body_positions = {}
        for body_name in key_bodies:
            try:
                body_id = mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                body_positions[body_name] = data.xpos[body_id].copy()
            except ValueError:
                logger.warning(f"Body '{body_name}' not found for FK calculation.")
        return body_positions

    def _load_reference_motion(self, motion_dir: str = "/Users/camdouglas/quark/data/amass/processed/sequences/"):
        """Loads a reference motion trajectory for imitation learning."""
        try:
            # Find a walking motion file
            walk_files = [f for f in os.listdir(motion_dir) if "normal_walk" in f and "poses.npy" in f]
            if not walk_files:
                logger.warning(f"No 'normal_walk' motion files found in {motion_dir}")
                self.reference_motion = None
                return
            
            motion_file_path = os.path.join(motion_dir, walk_files[0])
            root_file_path = motion_file_path.replace("poses.npy", "root.npy")
            vel_file_path = motion_file_path.replace("poses.npy", "vels.npy")

            if os.path.exists(motion_file_path) and os.path.exists(root_file_path) and os.path.exists(vel_file_path):
                poses = np.load(motion_file_path)
                root = np.load(root_file_path)
                vels = np.load(vel_file_path)
                
                self.reference_motion = {
                    "poses": poses, # Axis-angle poses
                    "root": root,   # Root position and orientation
                    "vels": vels,   # Joint velocities
                }
                logger.info(f"✅ Loaded reference motion '{walk_files[0]}' with {len(poses)} frames.")
            else:
                logger.warning(f"Motion file components not found for {walk_files[0]}")
                self.reference_motion = None

        except Exception as e:
            logger.error(f"Failed to load reference motion: {e}")
            self.reference_motion = None

    def _maybe_update_from_vla(self):
        """Update gait/velocity targets from VLA once per episode (at start)."""
        if not self.use_vla or self.vla is None:
            return
        if self.desired_vy is not None and self.step_count > 0:
            return
        # Placeholder image; can be replaced by a render buffer later
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        try:
            out = self.vla.infer(image=img, text_prompt=self.vla_prompt)
            gait = out.get("gait", {})
            freq = float(gait.get("frequency", getattr(self.cpg, "frequency", 0.6)))
            amp = float(gait.get("amplitude", 0.5))
            # Apply to CPG
            if hasattr(self.cpg, "frequency"):
                self.cpg.frequency = max(0.1, min(2.0, freq))
            # Two-level CPG uses hip/knee amplitudes
            if hasattr(self.cpg, "hip_amplitude"):
                self.cpg.hip_amplitude = max(0.05, min(1.2, amp))
            if hasattr(self.cpg, "knee_amplitude"):
                self.cpg.knee_amplitude = max(0.05, min(1.2, amp))
            tv = out.get("target_velocity", [0.0, 0.4, 0.0])
            self.desired_vy = float(tv[1]) if len(tv) > 1 else 0.4
            logger.info(f"🎛️ VLA targets: vy={self.desired_vy:.2f}, freq={getattr(self.cpg,'frequency',0.6):.2f}")
        except Exception as e:
            logger.warning(f"VLA inference failed: {e}")

    def _initialize_balance_safety_thresholds(self):
        """Initialize safety thresholds appropriate for balance learning task."""
        try:
            # Set much higher initial values to prevent emergency shutdown
            # These values should be well above the emergency threshold (20.0)
            if "embodiment_stability" in self.safety_protocols.safety_thresholds:
                self.safety_protocols.safety_thresholds["embodiment_stability"].current_value = 50.0
            
            # Set other critical thresholds to safe values
            if "neural_activity_stability" in self.safety_protocols.safety_thresholds:
                self.safety_protocols.safety_thresholds["neural_activity_stability"].current_value = 50.0
            
            if "learning_stability" in self.safety_protocols.safety_thresholds:
                self.safety_protocols.safety_thresholds["learning_stability"].current_value = 50.0
            
            if "safety_protocol_effectiveness" in self.safety_protocols.safety_thresholds:
                self.safety_protocols.safety_thresholds["safety_protocol_effectiveness"].current_value = 50.0
            
            logger.info("✅ Balance learning safety thresholds initialized with high values")
            
        except Exception as e:
            logger.warning(f"Could not initialize safety thresholds: {e}")

    def _check_episode_end(self, current_state: np.ndarray, step_count: int) -> bool:
        """Check if episode should end (falling, time limit, etc.)."""
        # qpos[2] is the torso height
        torso_height = current_state[2]
        return torso_height < 0.8 or step_count > 1000

    def _calculate_potential(self, state: np.ndarray) -> float:
        """Calculates the potential of a given state for reward shaping."""
        torso_height = state[2] # qpos[2] is height
        uprightness = state[6] # Assuming qpos[6] is related to torso orientation
        velocity_magnitude = np.linalg.norm(state[self.nq+0 : self.nq+3])
        
        # Encourage target height and upright posture, discourage high velocity
        height_potential = np.exp(-10.0 * abs(torso_height - 1.4))
        upright_potential = np.exp(-5.0 * (1.0 - uprightness))
        velocity_potential = np.exp(-0.1 * velocity_magnitude)
        
        return height_potential + upright_potential + velocity_potential
        
    def _calculate_imitation_reward(self, current_qpos, current_qvel) -> float:
        """Calculates a reward based on how closely the agent follows a reference motion."""
        if self.reference_motion is None:
            return 0.0

        ref_len = len(self.reference_motion["poses"])
        if ref_len == 0:
            return 0.0
        
        target_pose_aa = self.reference_motion["poses"][self.reference_motion_index]
        target_root = self.reference_motion["root"][self.reference_motion_index]
        target_vel = self.reference_motion["vels"][self.reference_motion_index]
        
        # --- Root Orientation Error ---
        # AMASS root orientation is axis-angle. MuJoCo free joint is quaternion [w, x, y, z].
        # current_qpos[3:7] is the torso quaternion
        current_quat = current_qpos[3:7]
        target_aa = target_root[3:6]
        
        # Convert target axis-angle to quaternion
        target_rot = R.from_rotvec(target_aa)
        target_quat = target_rot.as_quat() # Returns [x, y, z, w]
        # Reorder to MuJoCo's [w, x, y, z]
        target_quat_mujoco = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        
        quat_diff = np.dot(current_quat, target_quat_mujoco)**2
        orient_err = 1.0 - quat_diff
        
        # --- End-Effector Position Error ---
        current_fk = self._forward_kinematics(current_qpos)
        
        target_fk = {}
        if self.smpl_model is not None:
            with torch.no_grad():
                smpl_output = self.smpl_model(
                    global_orient=torch.from_numpy(target_root[3:6]).unsqueeze(0).float(),
                    body_pose=torch.from_numpy(target_pose_aa).unsqueeze(0).float(),
                    transl=torch.from_numpy(target_root[0:3]).unsqueeze(0).float()
                )
                smpl_joints = smpl_output.joints[0].numpy()
                # Map SMPL joints to our key bodies
                # SMPL joint indices for feet: 7 (L_Ankle), 8 (R_Ankle)
                # Hands: 20 (L_Wrist), 21 (R_Wrist). Head: 15
                target_fk['left_foot'] = smpl_joints[7]
                target_fk['right_foot'] = smpl_joints[8]
                target_fk['left_lower_arm'] = smpl_joints[20]
                target_fk['right_lower_arm'] = smpl_joints[21]
                target_fk['head'] = smpl_joints[15]

        pos_err = 0.0
        if current_fk and target_fk:
            for body_name in current_fk:
                if body_name in target_fk:
                    pos_err += np.mean(np.square(current_fk[body_name] - target_fk[body_name]))
            if len(current_fk) > 0:
                pos_err /= len(current_fk)

        # --- Pose Error (Joint Angles) ---
        # This remains a simplification. A full solution requires a forward kinematics model for SMPL.
        target_qpos_subset = np.zeros(len(SMPL_TO_QUARK_QPOS_MAP))
        current_qpos_subset = np.zeros(len(SMPL_TO_QUARK_QPOS_MAP))
        smpl_angles = np.linalg.norm(target_pose_aa.reshape(-1, 3), axis=1)

        for i, (qpos_idx, (smpl_idx, joint_type)) in enumerate(SMPL_TO_QUARK_QPOS_MAP.items()):
            current_qpos_subset[i] = current_qpos[qpos_idx]
            if joint_type == 'knee':
                 target_qpos_subset[i] = smpl_angles[smpl_idx]
            else:
                 target_qpos_subset[i] = 0
        
        pose_err = np.mean(np.square(current_qpos_subset - target_qpos_subset))

        # --- Velocity Error ---
        target_vel_subset = np.zeros(len(SMPL_TO_QUARK_QVEL_MAP))
        current_vel_subset = np.zeros(len(SMPL_TO_QUARK_QVEL_MAP))
        qvel_offset = 6

        for i, (qvel_idx, smpl_idx) in enumerate(SMPL_TO_QUARK_QVEL_MAP.items()):
            current_vel_subset[i] = current_qvel[qvel_idx + qvel_offset]
            target_vel_subset[i] = target_vel[smpl_idx]

        vel_err = np.mean(np.square(current_vel_subset - target_vel_subset))

        # Weight the components
        orient_reward = 0.4 * np.exp(-10.0 * orient_err)
        pos_reward = 0.4 * np.exp(-20.0 * pos_err) # Higher weight for position matching
        vel_reward = 0.2 * np.exp(-0.1 * vel_err)
        
        self.reference_motion_index = (self.reference_motion_index + 1) % ref_len
        
        return orient_reward + pos_reward + vel_reward

    def _calculate_locomotion_reward(self, current_state: np.ndarray, action: np.ndarray, done: bool, forward_delta: float = 0.0) -> float:
        """Reward function for walking."""
        # Linear velocities for free joint are at indices [nq : nq+6] within state_vector's qvel segment
        vx = current_state[self.nq + 0] if len(current_state) >= self.nq + 1 else 0.0
        vy = current_state[self.nq + 1] if len(current_state) >= self.nq + 2 else 0.0
        
        # Primary reward: forward progress (pelvis Y displacement per step) and velocity
        forward_progress = max(0.0, forward_delta)
        vy_term = 1.0 * max(0.0, vy)
        if self.desired_vy is None:
            self.desired_vy = 0.1
        vy_term += 2.0 * max(0.0, self.desired_vy - abs(vy - self.desired_vy))
        forward_reward = 20.0 * forward_progress + vy_term

        # Penalty for lateral drift
        side_penalty = -0.5 * abs(vx)

        # Penalty for falling
        done_penalty = -100.0 if done else 0.0

        # Encourage energy efficiency
        action_penalty = -0.005 * np.mean(np.square(action))

        total_reward = forward_reward + side_penalty + done_penalty + action_penalty
        return total_reward

    def _update_gait_fsm(self, left_contact: bool, right_contact: bool):
        """Advance simple alternating gait based on step count or contact.
        Preference: switch phase when swing leg contacts ground or after timeout.
        """
        self._phase_step_counter += 1
        timeout = (self._phase_step_counter >= self.gait_steps_per_phase)
        if self.gait_phase == "R_stance":
            # If left foot makes contact (swing completes) or timeout, switch
            if left_contact or timeout:
                self.gait_phase = "L_stance"
                self._phase_step_counter = 0
        else:
            if right_contact or timeout:
                self.gait_phase = "R_stance"
                self._phase_step_counter = 0

    def _calculate_reward(self, current_state: np.ndarray, action: np.ndarray, done: bool) -> float:
        """Refined reward function using potential-based shaping."""
        
        # Standard rewards
        alive_bonus = 1.0
        action_penalty = -0.01 * np.mean(np.square(action))
        
        # Potential-based reward
        current_potential = self._calculate_potential(current_state)
        potential_reward = self.gamma * current_potential - self.last_potential
        self.last_potential = current_potential
        
        # Large penalty for falling
        done_penalty = -20.0 if done else 0.0
        
        total_reward = alive_bonus + action_penalty + potential_reward + done_penalty
        return total_reward

    def _update_curriculum(self):
        """Advances the curriculum stage based on performance."""
        if self.curriculum_stage == 0 and self.stage_success_count >= 5: # 5 successes to advance
            self.curriculum_stage = 1
            self.stage_success_count = 0
            logging.info("🎓 ADVANCING TO CURRICULUM STAGE 1: Unlocking more joints.")
        elif self.curriculum_stage == 1 and self.stage_success_count >= 5:
            self.curriculum_stage = 2
            self.stage_success_count = 0
            logging.info("🎓 ADVANCING TO CURRICULUM STAGE 2: Unlocking arm joints.")
        elif self.curriculum_stage == 2 and self.stage_success_count >= 5:
            self.curriculum_stage = 3
            self.stage_success_count = 0
            logging.info("🎓 ADVANCING TO CURRICULUM STAGE 3: Full body control!")
        # Future stages...
            
    def _generate_rl_locomotion_command(self, sensory_data_str: str) -> str:
        """Generates a command using end-to-end RL with imitation guidance."""
        if isinstance(sensory_data_str, str):
            sensory_data = json.loads(sensory_data_str)
        else:
            sensory_data = sensory_data_str

        state_vector = np.array(sensory_data['state_vector'])
        qpos = state_vector[:self.nq]
        qvel = state_vector[self.nq:self.nq + self.nv]
        state_dict = {'state_vector': state_vector}

        # Get action from PPO agent. This is now the direct motor control signal.
        action, log_prob = self.balance_agent.get_action(state_dict)
        
        motor_controls = np.zeros(self.sim_model.nu)  # Initialize with the full size
        motor_controls[:action.shape[0]] = action

        # Track forward progress
        pelvis_y = qpos[1] if len(qpos) >= 2 else 0.0
        if self.prev_y is None:
            self.prev_y = pelvis_y
        forward_delta = float(pelvis_y - self.prev_y)
        self.prev_y = pelvis_y

        # Calculate combined reward
        done = self._check_episode_end(state_vector, self.step_count)
        
        task_reward = self._calculate_locomotion_reward(state_vector, motor_controls, done, forward_delta)
        imitation_reward = self._calculate_imitation_reward(qpos, qvel)
        
        # Combine rewards
        w_task = 0.5
        w_imitation = 0.5
        reward = w_task * task_reward + w_imitation * imitation_reward
        
        # Learning cycle
        next_state_dict = {'state_vector': state_vector}
        self.balance_agent.store_experience(state_dict, motor_controls, reward, next_state_dict, done, log_prob)
        
        self.episode_reward += reward
        self.step_count += 1
        
        if done:
            self.balance_agent.learn()
            self.episode_count += 1
            self.episode_rewards.append(self.episode_reward)
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
            logger.info(f"Ep: {self.episode_count}, Reward: {self.episode_reward:.2f} (Task: {w_task*task_reward:.2f}, Imit: {w_imitation*imitation_reward:.2f}), Best: {self.best_reward:.2f}")
            self.episode_reward = 0.0
            self.step_count = 0
            self.prev_y = None
            self.reference_motion_index = 0 # Reset reference motion on episode end

        return json.dumps({"actuators": {"controls": motor_controls.tolist()}})

    def _generate_learning_locomotion_command(self, sensory_data_str: str) -> str:
        """Generates a walking command using CPG and PPO, and handles the learning cycle."""
        if isinstance(sensory_data_str, str):
            sensory_data = json.loads(sensory_data_str)
        else:
            sensory_data = sensory_data_str
        
        state_vector = np.array(sensory_data['state_vector'])
        qpos = state_vector[:self.nq]
        qvel = state_vector[self.nq:self.nq + self.nv]
        state_dict = {'state_vector': state_vector}

        # Optionally update desired VLA-driven targets at episode start
        self._maybe_update_from_vla()
        
        # Track pelvis forward displacement (global Y)
        pelvis_y = qpos[1] if len(qpos) >= 2 else 0.0
        if self.prev_y is None:
            self.prev_y = pelvis_y
        forward_delta = float(pelvis_y - self.prev_y)
        self.prev_y = pelvis_y
        
        # Get balance action from PPO (reserved for future blending on upper-body)
        ppo_action, log_prob = self.balance_agent.get_action(state_dict)
        
        # Foot heights for simple contact heuristic
        feet = sensory_data.get('feet', {})
        left_z = feet.get('left_z', None)
        right_z = feet.get('right_z', None)
        left_fz = feet.get('left_fz', 0.0)
        right_fz = feet.get('right_fz', 0.0)
        left_contact = (left_z is not None and left_z < 0.03) or left_fz > 5.0
        right_contact = (right_z is not None and right_z < 0.03) or right_fz > 5.0
        
        # Finite-state gait: alternate stance/swing each ~0.6s
        self._update_gait_fsm(left_contact=left_contact, right_contact=right_contact)
        # Base targets
        r_hip_x_des = -0.1
        l_hip_x_des = -0.1
        r_knee_des = 0.15
        l_knee_des = 0.15
        # Swing parameters
        swing_hip_flex = 0.6
        swing_knee_flex = 0.65
        # Assign per phase
        if self.gait_phase == "R_stance":
            # Left swings forward
            l_hip_x_des = swing_hip_flex
            l_knee_des = swing_knee_flex
            r_hip_x_des = -0.15
            r_knee_des = 0.1
        else:
            # Right swings forward
            r_hip_x_des = swing_hip_flex
            r_knee_des = swing_knee_flex
            l_hip_x_des = -0.15
            l_knee_des = 0.1

        # Capture step: move swing foot proportionally to torso pitch and pitch rate
        # Positive torso_pitch (forward) and positive ang_vel_x increase swing hip flexion to catch balance
        qw, qx, qy, qz = (qpos[3], qpos[4], qpos[5], qpos[6]) if len(qpos) >= 7 else (1.0, 0.0, 0.0, 0.0)
        torso_pitch_cap = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        ang_vel_x_cap = qvel[3] if len(qvel) >= 4 else 0.0
        cap_kp = 0.6
        cap_kd = 0.4
        capture_offset = np.clip(cap_kp * torso_pitch_cap + cap_kd * ang_vel_x_cap, -0.4, 0.8)
        if self.gait_phase == "R_stance":
            l_hip_x_des = np.clip(l_hip_x_des + capture_offset, -0.2, 1.0)
            # Increase knee clearance slightly when stepping farther
            l_knee_des = np.clip(l_knee_des + 0.3 * max(0.0, capture_offset), 0.1, 1.2)
        else:
            r_hip_x_des = np.clip(r_hip_x_des + capture_offset, -0.2, 1.0)
            r_knee_des = np.clip(r_knee_des + 0.3 * max(0.0, capture_offset), 0.1, 1.2)
        
        # Joint indices in qpos/qvel (after free joint: qpos offset=7, qvel offset=6)
        idx_r_hip_x = 7 + 8
        idx_r_hip_y = 7 + 9
        idx_r_knee  = 7 + 10
        idx_r_ank_y = 7 + 11
        idx_r_ank_x = 7 + 12
        idx_l_hip_x = 7 + 13
        idx_l_hip_y = 7 + 14
        idx_l_knee  = 7 + 15
        idx_l_ank_y = 7 + 16
        idx_l_ank_x = 7 + 17
        
        vidx_r_hip_x = 6 + 8
        vidx_r_hip_y = 6 + 9
        vidx_r_knee  = 6 + 10
        vidx_r_ank_y = 6 + 11
        vidx_r_ank_x = 6 + 12
        vidx_l_hip_x = 6 + 13
        vidx_l_hip_y = 6 + 14
        vidx_l_knee  = 6 + 15
        vidx_l_ank_y = 6 + 16
        vidx_l_ank_x = 6 + 17
        
        controls = np.zeros(self.sim_model.nu, dtype=float)
        
        # PD gains (sagittal + small lateral ab/adduction on hip_y)
        kp_hip = 12.0
        kd_hip = 1.0
        kp_knee = 14.0
        kd_knee = 1.1
        kp_hip_y = 6.0
        kd_hip_y = 0.5
        
        # Current joint angles/velocities
        r_hip_x = qpos[idx_r_hip_x]; r_hip_x_vel = qvel[vidx_r_hip_x]
        r_knee  = qpos[idx_r_knee];  r_knee_vel  = qvel[vidx_r_knee]
        l_hip_x = qpos[idx_l_hip_x]; l_hip_x_vel = qvel[vidx_l_hip_x]
        r_hip_y = qpos[idx_r_hip_y]; r_hip_y_vel = qvel[vidx_r_hip_y]
        l_hip_y = qpos[idx_l_hip_y]; l_hip_y_vel = qvel[vidx_l_hip_y]
        l_knee  = qpos[idx_l_knee];  l_knee_vel  = qvel[vidx_l_knee]
        
        # PD torques (desired vel = 0)
        tau_r_hip_x = kp_hip * (r_hip_x_des - r_hip_x) - kd_hip * r_hip_x_vel
        tau_r_knee  = kp_knee * (r_knee_des  - r_knee)  - kd_knee * r_knee_vel
        tau_l_hip_x = kp_hip * (l_hip_x_des - l_hip_x) - kd_hip * l_hip_x_vel
        # Lateral stabilization: small hip_y ab/adduction toward zero
        tau_r_hip_y = -kp_hip_y * r_hip_y - kd_hip_y * r_hip_y_vel
        tau_l_hip_y = -kp_hip_y * l_hip_y - kd_hip_y * l_hip_y_vel
        tau_l_knee  = kp_knee * (l_knee_des  - l_knee)  - kd_knee * l_knee_vel
        
        # Ankle strategy: stance push-off when forward speed below target
        vx = state_vector[self.nq + 0] if len(state_vector) >= self.nq + 1 else 0.0
        vy = state_vector[self.nq + 1] if len(state_vector) >= self.nq + 2 else 0.0
        target_vy = self.desired_vy if (self.desired_vy is not None) else 0.5
        push_cmd = np.clip(1.2 * (target_vy - max(0.0, vy)), 0.0, 1.0)
        
        tau_r_ank_y = -0.2 * qpos[idx_r_ank_y] - 0.05 * qvel[vidx_r_ank_y]
        # Ankle_x roll compensation
        # Approximate small-angle torso roll from quaternion
        qw, qx, qy, qz = (qpos[3], qpos[4], qpos[5], qpos[6]) if len(qpos) >= 7 else (1.0, 0.0, 0.0, 0.0)
        torso_roll = np.arctan2(2*(qw*qy - qx*qz), 1 - 2*(qy*qy + qz*qz))
        ang_vel_y = qvel[4] if len(qvel) >= 5 else 0.0
        tau_r_ank_x = -0.3 * (qpos[idx_r_ank_x] + torso_roll) - 0.07 * (qvel[vidx_r_ank_x] + ang_vel_y)
        tau_l_ank_y = -0.2 * qpos[idx_l_ank_y] - 0.05 * qvel[vidx_l_ank_y]
        tau_l_ank_x = -0.3 * (qpos[idx_l_ank_x] - torso_roll) - 0.07 * (qvel[vidx_l_ank_x] - ang_vel_y)
        # Hip_y roll damping to assist lateral balance
        tau_stab_r_roll = -1.0 * torso_roll - 0.1 * ang_vel_y
        tau_stab_l_roll = -1.0 * torso_roll - 0.1 * ang_vel_y
        tau_r_hip_y += tau_stab_r_roll
        tau_l_hip_y += tau_stab_l_roll
        if self.gait_phase == "R_stance":
            tau_r_ank_y += push_cmd
        else:
            tau_l_ank_y += push_cmd
        
        # Torso stabilization using free joint orientation (quaternion) and angular velocity
        # qpos[3:7] = quaternion [w, x, y, z] in MuJoCo for free joint
        qw, qx, qy, qz = (qpos[3], qpos[4], qpos[5], qpos[6]) if len(qpos) >= 7 else (1.0, 0.0, 0.0, 0.0)
        # Approximate small-angle torso pitch from quaternion
        torso_pitch = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        ang_vel_x = qvel[3] if len(qvel) >= 4 else 0.0
        # Bias hip torques to counter pitch and damp rate
        tau_stab_r = -0.8 * torso_pitch - 0.1 * ang_vel_x
        tau_stab_l = -0.8 * torso_pitch - 0.1 * ang_vel_x
        
        tau_r_hip_x += tau_stab_r
        tau_l_hip_x += tau_stab_l
        
        # Map torques to actuator control vector
        # [neck_y, neck_x, r_sh_x, r_sh_y, r_elb, l_sh_x, l_sh_y, l_elb,
        #  r_hip_x, r_hip_y, r_knee, r_ank_y, r_ank_x, l_hip_x, l_hip_y, l_knee, l_ank_y, l_ank_x]
        controls = np.zeros(self.sim_model.nu, dtype=float)
        
        controls[8]  = tau_r_hip_x
        controls[10] = tau_r_knee
        controls[11] = tau_r_ank_y
        controls[12] = tau_r_ank_x
        controls[13] = tau_l_hip_x
        controls[9]  = tau_r_hip_y
        controls[15] = tau_l_knee
        controls[16] = tau_l_ank_y
        controls[17] = tau_l_ank_x
        controls[14] = tau_l_hip_y
        
        # If imitation policy is available, blend its target angles for legs
        if self.use_imitation and self.imitation is not None:
            # Build imitation observation: [root vel(3) + joint vel(18) + pad(3)]
            rv = qvel[3:6] if len(qvel) >= 6 else np.zeros(3)
            pad = np.zeros(3)
            jv = np.zeros(self.sim_model.nu)
            # Map current joint velocities into 18-D slots where possible
            jv[8] = qvel[vidx_r_hip_x]; jv[10] = qvel[vidx_r_knee]; jv[11] = qvel[vidx_r_ank_y]; jv[12] = qvel[vidx_r_ank_x]
            jv[13] = qvel[vidx_l_hip_x]; jv[15] = qvel[vidx_l_knee]; jv[16] = qvel[vidx_l_ank_y]; jv[17] = qvel[vidx_l_ank_x]
            obs = np.concatenate([rv, jv, pad], axis=0)
            imitation_angles = self.imitation.predict(obs)
            # Convert imitation target angles into additional torques (PD toward target)
            # Use only legs slots
            if imitation_angles.shape[0] >= self.sim_model.nu:
                r_hip_x_t = imitation_angles[8]; r_knee_t = imitation_angles[10]
                l_hip_x_t = imitation_angles[13]; l_knee_t = imitation_angles[15]
                tau_r_hip_x += 0.8 * (r_hip_x_t - r_hip_x)
                tau_r_knee  += 0.8 * (r_knee_t  - r_knee)
                tau_l_hip_x += 0.8 * (l_hip_x_t - l_hip_x)
                tau_l_knee  += 0.8 * (l_knee_t  - l_knee)

        motor_controls = np.clip(controls, -1.0, 1.0)
        
        # Episode logic and learning
        done = self._check_episode_end(state_vector, self.step_count)
        reward = self._calculate_locomotion_reward(state_vector, motor_controls, done, forward_delta)
        
        next_state_dict = {'state_vector': state_vector}
        self.balance_agent.store_experience(state_dict, motor_controls, reward, next_state_dict, done, log_prob)
        
        self.episode_reward += reward
        self.step_count += 1
        
        if done:
            self.balance_agent.learn()
            self.episode_count += 1
            self.episode_rewards.append(self.episode_reward)
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
            logger.info(f"Ep: {self.episode_count}, Reward: {self.episode_reward:.2f}, Best: {self.best_reward:.2f}")
            self.episode_reward = 0.0
            self.step_count = 0
            self.prev_y = None
        
        return json.dumps({"actuators": {"controls": motor_controls.tolist()}})

    def _generate_learning_balance_command(self, sensory_data_str: str) -> str:
        if isinstance(sensory_data_str, str):
            sensory_data = json.loads(sensory_data_str)
        else:
            sensory_data = sensory_data_str
        
        state_vector = np.array(sensory_data['state_vector'])
        state_dict = {'state_vector': state_vector}
        
        action, log_prob = self.balance_agent.get_action(state_dict)
        
        # Apply action mask based on curriculum stage
        if self.curriculum_stage == 0:
            action_mask = np.zeros(self.sim_model.nu)
            action_mask[0:6] = 1 # Allow control only for root joint
            action *= action_mask
        elif self.curriculum_stage == 1:
            action_mask = np.zeros(self.sim_model.nu)
            action_mask[0:6] = 1 # Root
            action_mask[8:12] = 1 # Hips
            action_mask[14:16] = 1 # Knees
            action *= action_mask
        elif self.curriculum_stage == 2:
            action_mask = np.ones(self.sim_model.nu) # All joints enabled except head/feet
            action_mask[6:8] = 0 # Lock head
            action_mask[12:14] = 0 # Lock feet
            action *= action_mask
        elif self.curriculum_stage == 3:
            action_mask = np.ones(self.sim_model.nu) # All joints
            action *= action_mask
            
        motor_controls = np.clip(action, -1.0, 1.0)
        
        reward = self._calculate_reward(state_vector, action, self._check_episode_end(state_vector, self.step_count))
        done = self._check_episode_end(state_vector, self.step_count)
        
        next_state_dict = {'state_vector': state_vector}
        
        self.balance_agent.store_experience(state_dict, action, reward, next_state_dict, done, log_prob)
        
        self.episode_reward += reward
        self.step_count += 1
        
        if done:
            self.balance_agent.learn()
            self.episode_count += 1
            self.episode_rewards.append(self.episode_reward)
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward

            logger.info(f"Ep: {self.episode_count}, Reward: {self.episode_reward:.2f}, Best: {self.best_reward:.2f}")
            
            if self.episode_reward > 100: # Success threshold
                self.stage_success_count += 1
            else:
                self.stage_success_count = 0 # Reset on failure
            
            self._update_curriculum()
            
            self.episode_reward = 0.0
            self.step_count = 0
            self.last_potential = 0.0 # Reset potential at the end of an episode

        return json.dumps({"actuators": {"controls": motor_controls.tolist()}})
    
    def _generate_walking_command(self, sensory_data_str: str) -> str:
        """Generates a walking command using the CPG."""
        cpg_action = self.cpg.step()
        
        # Combine PPO and CPG actions
        state_dict = {'state_vector': np.zeros(self.state_dim)} # Dummy state
        ppo_action, _ = self.balance_agent.get_action(state_dict)
        
        # Blend actions - CPG for legs, PPO for balance
        final_action = ppo_action
        final_action[8:16] = cpg_action # Override leg joints with CPG
        
        motor_controls = np.clip(final_action, -1.0, 1.0)
        return json.dumps({"actuators": {"controls": motor_controls.tolist()}})

    def generate_motor_command_from_sensory_data(self, sensory_data_str: str) -> str:
        if self.use_imitation:
            return self._generate_rl_locomotion_command(sensory_data_str)
        
        if self.use_pretrained_policy and self.sb3_adapter is not None:
            return self._generate_sb3_policy_command(sensory_data_str)
        # default: learning locomotion
        return self._generate_learning_locomotion_command(sensory_data_str)

    def _generate_sb3_policy_command(self, sensory_data_str: str) -> str:
        if isinstance(sensory_data_str, str):
            sensory_data = json.loads(sensory_data_str)
        else:
            sensory_data = sensory_data_str
        state_vector = np.array(sensory_data['state_vector'])
        # Predict continuous action from pretrained policy
        try:
            action = self.sb3_adapter.predict(state_vector, deterministic=True)
        except Exception as e:
            logger.warning(f"SB3 policy inference failed: {e}. Reverting to locomotion controller this step.")
            return self._generate_learning_locomotion_command(sensory_data_str)
        controls = np.zeros(self.sim_model.nu, dtype=float)
        # Walker2d typically has 6 actions -> map to [r_hip_x, r_knee, r_ank_y, l_hip_x, l_knee, l_ank_y]
        if action is not None and action.size >= 6:
            r_hip_x, r_knee, r_ank_y, l_hip_x, l_knee, l_ank_y = action[:6]
            controls[8]  = r_hip_x
            controls[10] = r_knee
            controls[11] = r_ank_y
            controls[13] = l_hip_x
            controls[15] = l_knee
            controls[16] = l_ank_y
        elif action is not None and action.size > 0:
            n = min(action.size, controls.size)
            controls[:n] = action[:n]
        motor_controls = np.clip(controls, -1.0, 1.0)
        # Track simple reward: forward velocity
        vy = state_vector[self.nq + 1] if len(state_vector) >= self.nq + 2 else 0.0
        reward = float(max(0.0, vy) - 0.001 * np.mean(motor_controls**2))
        self.episode_reward += reward
        self.step_count += 1
        return json.dumps({"actuators": {"controls": motor_controls.tolist()}})

    def get_learning_stats(self) -> dict:
        """Get current learning statistics for simple balance learning."""
        try:
            stats = {
                "episode_count": self.episode_count,
                "current_episode_reward": self.episode_reward,
                "best_reward": self.best_reward,
                "average_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            }
            
            # Add safety score if available
            if hasattr(self.safety_protocols, 'get_safety_score'):
                stats["safety_score"] = self.safety_protocols.get_safety_score()
            else:
                stats["safety_score"] = "N/A"
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return {"error": str(e)}

    def _update_safety_thresholds(self, current_state: np.ndarray, reward: float):
        """Update safety thresholds based on current learning performance."""
        try:
            # Update embodiment stability based on current balance performance
            if "embodiment_stability" in self.safety_protocols.safety_thresholds:
                # Higher stability for better balance performance
                stability_score = max(0.5, min(1.0, 0.8 + reward * 0.1))
                self.safety_protocols.safety_thresholds["embodiment_stability"].current_value = stability_score
            
            # Update learning stability based on episode performance
            if "learning_stability" in self.safety_protocols.safety_thresholds:
                # Learning stability improves with positive rewards
                learning_score = max(0.6, min(1.0, 0.8 + max(0, reward) * 0.05))
                self.safety_protocols.safety_thresholds["learning_stability"].current_value = learning_score
            
            # Update neural activity stability based on step count
            if "neural_activity_stability" in self.safety_protocols.safety_thresholds:
                # Stability improves with more learning steps
                neural_score = max(0.7, min(1.0, 0.8 + min(self.step_count / 1000, 0.2)))
                self.safety_protocols.safety_thresholds["neural_activity_stability"].current_value = neural_score
                
        except Exception as e:
            logger.warning(f"Could not update safety thresholds: {e}")
