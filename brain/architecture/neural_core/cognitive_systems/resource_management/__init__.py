#!/usr/bin/env python3
"""Resource Management Module - Interface for cognitive resource management.

Integration: Main resource management interface for cognitive systems.
Rationale: Modular resource management with preserved functionality.
"""

# Import from the core manager
from .manager_core import ResourceManager, compute_sha256

# Export for backward compatibility
__all__ = ['ResourceManager', 'compute_sha256']
