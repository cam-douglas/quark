"""
Compatibility shim: provide a minimal 'gym' namespace backed by gymnasium.

This allows loading legacy Stable-Baselines3 models that import 'gym'.
"""
from gymnasium import *  # noqa: F401,F403
from gymnasium import make, register, spaces, wrappers  # noqa: F401
from gymnasium.core import Env  # noqa: F401

__all__ = []
try:
    # Populate __all__ for basic compatibility
    from gymnasium import __all__ as _ga
    __all__ = list(_ga)
except Exception:
    pass

__version__ = "0.0.0-shim"


