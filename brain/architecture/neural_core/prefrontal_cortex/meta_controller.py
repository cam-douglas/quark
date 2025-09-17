"""Meta-Controller (PFC Analogue) - Phase 3 Prototype
Balances extrinsic and intrinsic rewards to guide goal-directed behavior.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

from typing import Dict

class MetaController:
    """
    A simplified model of a meta-controller (analogous to PFC) that balances
    extrinsic and intrinsic rewards to guide the agent's learning.
    """
    def __init__(self, extrinsic_weight=0.7, intrinsic_weight=0.3):
        """
        Initializes the MetaController.
        Args:
            extrinsic_weight (float): The weight to give to the task-based reward.
            intrinsic_weight (float): The weight to give to the novelty-based reward.
        """
        self.extrinsic_weight = extrinsic_weight
        self.intrinsic_weight = intrinsic_weight
        self._pending_tasks = []  # NEW: tasks from advanced planner
        print(f"🌟 Meta-Controller initialized with weights (Extrinsic: {extrinsic_weight}, Intrinsic: {intrinsic_weight})")

    # ------------------------------------------------------------------
    # Task ingestion (from Advanced Planner)
    # ------------------------------------------------------------------
    def ingest_tasks(self, tasks):
        """Receive list of task dicts from Advanced Planner."""
        if isinstance(tasks, list):
            self._pending_tasks.extend(tasks)

    def get_pending_tasks(self):
        return list(self._pending_tasks)

    def pop_next_task(self):
        if self._pending_tasks:
            return self._pending_tasks.pop(0)
        return None

    # ------------------------------------------------------------------
    # Reward blending
    # ------------------------------------------------------------------
    def blend_rewards(self, extrinsic_reward: float, intrinsic_reward: float) -> float:
        """
        Combines the extrinsic and intrinsic rewards into a single value.
        """
        blended_reward = (self.extrinsic_weight * extrinsic_reward) + \
                         (self.intrinsic_weight * intrinsic_reward)
        return blended_reward

    def step(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        The main step function for the module.
        """
        extrinsic_reward = inputs.get("extrinsic_reward", 0.0)
        intrinsic_reward = inputs.get("intrinsic_reward", 0.0)

        final_reward = self.blend_rewards(extrinsic_reward, intrinsic_reward)

        return {"blended_reward": final_reward}
