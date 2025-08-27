"""
Compatibility shim for 'gym.spaces' that re-exports Gymnasium spaces.
This allows loading legacy SB3 models expecting 'gym.spaces'.
"""
from gymnasium.spaces import *  # noqa: F401,F403


