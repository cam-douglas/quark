"""Plugin framework for ResourceManager."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type

class ResourcePlugin(ABC):
    """Base class for resource integration plugins.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
    @abstractmethod
    def can_handle(self, meta: Dict[str, Any]) -> bool: ...

    @abstractmethod
    def integrate(self, meta: Dict[str, Any]) -> bool:
        """Return True if plugin handled the resource and default flow should stop."""
        ...

_plugins: List[ResourcePlugin] = []

def register_plugin(cls: Type[ResourcePlugin]):
    _plugins.append(cls())
    return cls

def get_plugins() -> List[ResourcePlugin]:
    return _plugins
