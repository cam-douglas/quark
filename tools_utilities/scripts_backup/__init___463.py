"""
Configuration package for Quark Brain Simulation

Contains connectome configurations and other brain simulation settings.
"""

from pathlib import Path

# Get the config directory path
CONFIG_DIR = Path(__file__).parent

# Default connectome configuration path
DEFAULT_CONNECTOME_PATH = CONFIG_DIR / "connectome.yaml"

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_CONNECTOME_PATH",
]
