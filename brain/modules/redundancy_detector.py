#!/usr/bin/env python3
"""Redundancy Detector - Interface to modular redundancy analysis.

Integration: Analysis tool for brain optimization and cleanup workflows.
Rationale: Streamlined interface to modular redundancy detection components.
"""

# Import from modular analysis tools
from .analysis_tools.redundancy_core import *

# Re-export for backward compatibility
__all__ = ['RedundancyDetector']
