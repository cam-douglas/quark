"""
Exponential Learning System for Small-Mind
==========================================

A hierarchical learning architecture that implements exponential knowledge growth
through curiosity-driven exploration and intelligent synthesis.

Components:
- Curiosity: Interest-driven learning and question generation
- Exploration: Active environment interaction and knowledge discovery
- Synthesis: Pattern recognition and knowledge integration

This system is designed to scale learning efficiency exponentially through
recursive knowledge building and meta-learning capabilities.
"""

__version__ = "0.1.0"

from .....................................................curiosity import CuriosityEngine
from .....................................................exploration import ExplorationModule
from .....................................................synthesis import SynthesisEngine

__all__ = [
    "CuriosityEngine",
    "ExplorationModule", 
    "SynthesisEngine"
]
