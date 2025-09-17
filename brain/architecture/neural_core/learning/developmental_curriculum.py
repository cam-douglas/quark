"""Developmental Motor Curriculum System
Trains Quark like a human baby with progressive motor milestones

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import numpy as np
import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class DevelopmentalStage(Enum):
    """Simplified stages for crawling task."""
    CRAWLING = "crawling"

class MotorMilestone:
    """A simplified structure for a motor skill."""
    def __init__(self, name: str, stage: DevelopmentalStage, success_criteria: Dict[str, float]):
        self.name = name
        self.stage = stage
        self.success_criteria = success_criteria
        self.achieved = False

class DevelopmentalCurriculum:
    """
    A curriculum focused on crawling by rewarding forward progress.
    """
    def __init__(self):
        self.milestones = self._create_developmental_milestones()
        self.current_milestone = self.milestones["crawl_forward"]
        print("🧎 Quark's curriculum is now focused on crawling forward.")

    def _create_developmental_milestones(self) -> Dict[str, MotorMilestone]:
        milestones = {}
        milestones["crawl_forward"] = MotorMilestone(
            name="crawl_forward",
            stage=DevelopmentalStage.CRAWLING,
            success_criteria={"distance": 2.0}  # target to crawl 2 meters total
        )
        return milestones

    def calculate_developmental_reward(self, current_state: Dict[str, Any]) -> float:
        """
        Reward = forward progress + coordination bonus - posture penalty
        """
        qpos = current_state.get("qpos")
        qvel = current_state.get("qvel")
        prev_x = float(current_state.get("prev_x", qpos[0]))
        is_fallen = bool(current_state.get("is_fallen", False))
        if qpos is None or qvel is None:
            return 0.0

        if is_fallen:
            return -5.0

        x = float(qpos[0])
        z = float(qpos[2])
        forward_delta = max(0.0, x - prev_x)

        # --- Coordination Bonus ---
        # Get velocities for right arm (shoulder_y) and left leg (hip_x)
        right_shoulder_vel = qvel[6] # Heuristic index, check your model
        left_hip_vel = qvel[9]       # Heuristic index, check your model

        # Reward if they move in the same direction (e.g., both forward)
        coordination_bonus = 0.0
        if np.sign(right_shoulder_vel) == np.sign(left_hip_vel):
            coordination_bonus = 0.5 * min(abs(right_shoulder_vel), abs(left_hip_vel))

        # Penalize being too high (trying to stand)
        height_penalty = max(0, (z - 0.8)) * 0.5

        reward = (10.0 * forward_delta) + coordination_bonus - height_penalty
        return float(reward)

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_milestone": self.current_milestone.name,
            "current_stage": self.current_milestone.stage.value,
        }
