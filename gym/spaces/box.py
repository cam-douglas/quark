"""
Compatibility shim for 'gym.spaces.box.Box' using Gymnasium's Box.
"""
from gymnasium.spaces import Box  # noqa: F401

__all__ = ["Box"]


