

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
try:
    from brain.architecture.neural_core.cognitive_systems.plugins import register_plugin, ResourcePlugin
except Exception:
    from . import register_plugin, ResourcePlugin  # type: ignore
from typing import Dict, Any

@register_plugin
class NoOpPlugin(ResourcePlugin):
    """Sample plugin that logs but never handles resources."""
    def can_handle(self, meta: Dict[str, Any]) -> bool:
        return False

    def integrate(self, meta: Dict[str, Any]) -> bool:
        return False
