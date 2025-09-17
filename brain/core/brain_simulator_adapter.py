"""brain_simulator_adapter.py – Ensures BrainSimulator.step returns reward & loss.

This small adapter keeps the main simulation loop agnostic to whether the
underlying BrainSimulator implementation already includes KPI keys.
"""
from __future__ import annotations
from typing import Any, Dict


def step_with_metrics(simulator: "BrainSimulator", inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
    """Call simulator.step and guarantee reward & loss keys in output."""
    out = simulator.step(inputs)

    # Ensure dict return type
    if not isinstance(out, dict):
        out = {"action": out}

    # Extract reward/loss if provided by the model
    reward = out.get("reward")
    loss = out.get("loss")

    # Sync back into simulator attributes or provide fallback
    if reward is not None:
        setattr(simulator, "current_reward", float(reward))
    else:
        reward = getattr(simulator, "current_reward", 0.0)
        out["reward"] = reward

    if loss is not None:
        setattr(simulator, "last_policy_loss", float(loss))
    else:
        loss = getattr(simulator, "last_policy_loss", 0.0)
        out["loss"] = loss

    return out
