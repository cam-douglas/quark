import os
import numpy as np
from typing import Dict, Any, Optional


class HRMPlanner:
    """
    Thin adapter around sapientinc/HRM to provide high-level planning priors.

    This adapter is intentionally defensive: if the external repo or its heavy
    dependencies are not available, it gracefully degrades to a no-op prior.
    """

    def __init__(self, num_actions: int):
        self.num_actions = int(num_actions)
        self.available = False

        # Detect external repo presence
        self.repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../external/hrm"))
        if os.path.isdir(self.repo_path):
            # We keep integration minimal; skip heavy imports by default
            self.available = True
            print("✅ HRM Planner adapter initialized. Model is available (but not yet loaded).")
        else:
            print("⚠️ HRM Planner adapter: external/hrm repository not found. Planner will be disabled.")


    def plan(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Returns a dict with optional keys:
          - subgoal: textual description (optional)
          - policy_prior: np.ndarray shape (num_actions,) with action preferences
          - confidence: float in [0,1]
        """
        if not self.available or obs is None:
            return {
                "subgoal": None,
                "policy_prior": np.zeros(self.num_actions, dtype=np.float32),
                "confidence": 0.0,
            }

        # Placeholder prior: encourage small forward-leaning primitives if present.
        # This is a stand-in for a real model call.
        prior = np.zeros(self.num_actions, dtype=np.float32)
        
        return {
            "subgoal": "crawl_forward", # Placeholder subgoal
            "policy_prior": prior,
            "confidence": 0.2, # Placeholder confidence
        }


