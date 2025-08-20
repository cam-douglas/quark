"""
Core brain simulation modules for Quark.

This package contains the main brain simulation components:
- brain_launcher_v3: Main brain simulation engine
- rules_loader: Rule validation and enforcement
"""

from ................................................brain_launcher_v3 import Brain, main as run_brain_simulation
from ................................................rules_loader import load_rules, validate_connectome, instrument_agent

__all__ = [
    "Brain",
    "run_brain_simulation", 
    "load_rules",
    "validate_connectome",
    "instrument_agent",
]
