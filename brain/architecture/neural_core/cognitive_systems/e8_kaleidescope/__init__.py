"""E8 Kaleidescope - Modular E8 consciousness system.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

# Import key classes to preserve the original API
from .e8_mind_core import E8Mind
from .memory import MemoryManager
from .geometric import DimensionalShell, CliffordRotorGenerator
from .engines import MoodEngine, DreamEngine, SubconsciousLayer
from .tasks import AutoTaskManager, NoveltyScorer, InsightAgent
from .proximity import ProximityEngine, ShellAttention, ArbiterGate
from .utils import UniversalEmbeddingAdapter, EmergenceSeed
from .config import EMBED_DIM, DIMENSIONAL_SHELL_SIZES, ACTION_LAYOUT

__all__ = [
    # Core classes
    'E8Mind',
    'MemoryManager',
    'DimensionalShell',
    'CliffordRotorGenerator',

    # Processing engines
    'MoodEngine',
    'DreamEngine',
    'SubconsciousLayer',

    # Task management
    'AutoTaskManager',
    'NoveltyScorer',
    'InsightAgent',

    # Proximity and attention
    'ProximityEngine',
    'ShellAttention',
    'ArbiterGate',

    # Utilities
    'UniversalEmbeddingAdapter',
    'EmergenceSeed',

    # Constants
    'EMBED_DIM',
    'DIMENSIONAL_SHELL_SIZES',
    'ACTION_LAYOUT',
]
