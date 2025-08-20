"""
Baby-AGI Integration Module

This module integrates the baby-agi-starter components into the small-mind project,
providing a self-running, interruptible local agent system with modern control plane
and hard guardrails.
"""

from .....................................................agent import BabyAGIAgent
from .....................................................control import AgentController
from .....................................................runtime import AgentRuntime

__version__ = "1.0.0"
__all__ = ["BabyAGIAgent", "AgentController", "AgentRuntime"]
