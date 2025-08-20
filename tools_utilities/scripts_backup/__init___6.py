"""
Quark Brain Simulation Framework

A newborn cognitive brain simulation framework with biologically-grounded neural architectures.
"""

__version__ = "0.1.0"
__author__ = "Quark Brain Team"
__email__ = "team@quarkbrain.ai"

# Import core modules for easy access
from .....................................................core.brain_launcher_v3 import Brain, main as run_brain_simulation

__all__ = [
    "Brain",
    "run_brain_simulation",
    "__version__",
    "__author__",
    "__email__",
]
