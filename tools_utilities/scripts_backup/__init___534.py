"""
Exploration Module - Active Knowledge Discovery
==============================================

The exploration module implements active learning strategies to
discover new knowledge through strategic interaction with the environment.

Features:
- Multi-strategy exploration (random, guided, curiosity-driven)
- Safe exploration with uncertainty bounds
- Exploration planning and optimization
- Knowledge discovery tracking
- Environment interaction protocols
"""

from ................................................module import ExplorationModule
from ................................................strategies import ExplorationStrategy, RandomStrategy, GuidedStrategy
from ................................................planner import ExplorationPlanner
from ................................................environment import EnvironmentInterface

__all__ = [
    "ExplorationModule",
    "ExplorationStrategy",
    "RandomStrategy", 
    "GuidedStrategy",
    "ExplorationPlanner",
    "EnvironmentInterface"
]
