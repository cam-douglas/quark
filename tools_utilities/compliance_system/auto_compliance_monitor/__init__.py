"""
Auto Compliance Monitor Package

Real-time compliance monitoring system that automatically
runs compliance checks after any file operation.

Refactored from 305-line file to maintain <300 line limit.

Author: Quark AI
Date: 2025-01-27
"""

from .core_monitor import AutoComplianceMonitor

__all__ = ['AutoComplianceMonitor']
