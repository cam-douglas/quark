"""
Core brain simulation modules for Quark.

This package contains the main brain simulation components:
- rules_loader: Rule validation and enforcement
"""

from ................................................rules_loader import load_rules, validate_connectome, instrument_agent

__all__ = [
    "load_rules",
    "validate_connectome",
    "instrument_agent",
]
