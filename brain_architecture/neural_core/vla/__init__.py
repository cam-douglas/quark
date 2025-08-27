"""
VLA (Vision-Language-Action) minimal scaffold for Quark.

Purpose: Provide a PaLM-E–style interface that fuses vision and text to
produce high-level control intents (e.g., target velocity, gait params).

Inputs: image (H,W,3) numpy array or torch tensor; text prompt (str)
Outputs: dict with fields like {"target_velocity": [vx, vy, vz], "gait": [...]}.

Seeds/Deps: torch, torchvision (optional). No external network calls.
"""

from .vla_interface import VLAInterface  # noqa: F401
from .palme_style_model import PalmEStyleVLA  # noqa: F401


