#!/usr/bin/env python3
"""
SB3 Policy Adapter for Quark

Purpose: Load a pretrained Stable-Baselines3 policy from Hugging Face and
produce continuous actions for Quark's actuators given MuJoCo state.

Inputs:
- repo_id: Hugging Face repository id (e.g., 'sb3/ppo-Humanoid-v3')
- filename: model file in the repo (e.g., 'ppo-Humanoid-v3.zip')
- optional vecnormalize file (if present): 'vecnormalize.pkl' or 'statistics.pkl'

Outputs:
- numpy array action in [-1, 1]

Seeds/Deps:
- Python 3.11, stable-baselines3, torch, huggingface_hub, numpy
"""

from typing import Optional
import os
import numpy as np

from huggingface_hub import hf_hub_download

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
except Exception:  # pragma: no cover
    PPO = None
    VecNormalize = None


class SB3HumanoidPolicyAdapter:
    def __init__(
        self,
        repo_id: str = "sb3/ppo-Humanoid-v3",
        filename: str = "ppo-Humanoid-v3.zip",
        vecnormalize_filename_candidates: tuple[str, ...] = ("vecnormalize.pkl", "statistics.pkl"),
    ) -> None:
        if PPO is None:
            raise RuntimeError("stable-baselines3 is not installed. Please install it to use SB3HumanoidPolicyAdapter.")

        # Download model
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Try to download vecnormalize stats if exist
        self.vecnormalize_path: Optional[str] = None
        for cand in vecnormalize_filename_candidates:
            try:
                self.vecnormalize_path = hf_hub_download(repo_id=repo_id, filename=cand)
                break
            except Exception:
                continue

        # Load model (policy only; no actual env needed for inference)
        self.model = PPO.load(self.model_path, device="cpu")

        # Load VecNormalize parameters if available to normalize observations
        self.obs_rms_mean: Optional[np.ndarray] = None
        self.obs_rms_var: Optional[np.ndarray] = None
        if self.vecnormalize_path and VecNormalize is not None:
            try:
                # VecNormalize.load requires an env; we instead load dict manually
                import pickle
                with open(self.vecnormalize_path, "rb") as f:
                    stats = pickle.load(f)
                # RL Zoo stores as dict with keys 'obs_rms', etc.
                obs_rms = stats.get("obs_rms") if isinstance(stats, dict) else None
                if obs_rms is not None:
                    self.obs_rms_mean = getattr(obs_rms, "mean", None)
                    self.obs_rms_var = getattr(obs_rms, "var", None)
            except Exception:
                self.obs_rms_mean = None
                self.obs_rms_var = None

        # derive dims from spaces if present
        self.policy_obs_dim: Optional[int] = None
        self.policy_action_dim: Optional[int] = None
        try:
            if hasattr(self.model, "observation_space") and hasattr(self.model.observation_space, "shape"):
                shp = self.model.observation_space.shape
                if shp is not None and len(shp) > 0:
                    self.policy_obs_dim = int(np.prod(shp))
        except Exception:
            pass
        try:
            if hasattr(self.model, "action_space") and hasattr(self.model.action_space, "shape"):
                ashp = self.model.action_space.shape
                if ashp is not None and len(ashp) > 0:
                    self.policy_action_dim = int(np.prod(ashp))
        except Exception:
            pass

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms_mean is None or self.obs_rms_var is None:
            return obs
        # Standard VecNormalize transform
        eps = 1e-8
        return (obs - self.obs_rms_mean) / np.sqrt(self.obs_rms_var + eps)

    def _project_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.policy_obs_dim is None:
            return obs
        if obs.shape[0] == self.policy_obs_dim:
            return obs
        if obs.shape[0] > self.policy_obs_dim:
            return obs[: self.policy_obs_dim]
        # pad with zeros
        padded = np.zeros(self.policy_obs_dim, dtype=obs.dtype)
        padded[: obs.shape[0]] = obs
        return padded

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return continuous action in [-1, 1].

        observation: 1D numpy array (concatenated qpos+qvel is fine)
        """
        if observation.ndim != 1:
            observation = observation.flatten()
        obs = self._project_obs(observation)
        obs = self._normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        if isinstance(action, np.ndarray):
            return action
        return np.asarray(action)


