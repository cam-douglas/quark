from . import register_plugin, ResourcePlugin
from typing import Dict, Any

@register_plugin
class NoOpPlugin(ResourcePlugin):
    """Sample plugin that logs but never handles resources."""
    def can_handle(self, meta: Dict[str, Any]) -> bool:
        return False

    def integrate(self, meta: Dict[str, Any]) -> bool:
        return False
