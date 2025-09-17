#!/usr/bin/env python3
"""S3 Integration Module - Main interface for S3 and cloud storage integration.

Provides unified interface to S3 infrastructure and model management.

Integration: Main S3 interface for QuarkDriver and AutonomousAgent.
Rationale: Clean API abstraction over S3 integration modules.
"""

from .s3_infrastructure import S3Infrastructure

# Main integration class that replaces the original QuarkS3StateIntegration
class QuarkS3StateIntegration:
    """Unified S3 integration interface."""

    def __init__(self):
        self.infrastructure = S3Infrastructure()
        self.quark_root = self.infrastructure.quark_root
        self.state_dir = self.quark_root / "state" / "quark_state_system"
        self.instance_specs = self.infrastructure.instance_specs

    def get_s3_status(self):
        """Get S3 status."""
        return self.infrastructure.get_s3_status()

    def validate_s3_integration(self):
        """Validate S3 integration."""
        return self.infrastructure.validate_s3_integration()

    def get_storage_optimization_config(self):
        """Get storage optimization configuration."""
        return self.infrastructure.get_storage_optimization_config()

    def update_state_with_s3_config(self):
        """Update state with S3 configuration (compatibility method)."""
        # This would update state files with S3 configuration
        # For now, return the current S3 status
        return self.get_s3_status()

# Export main interface
__all__ = [
    'QuarkS3StateIntegration',
    'S3Infrastructure'
]
