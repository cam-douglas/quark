#!/usr/bin/env python3
"""Quantum Computing Module - Main interface for quantum integration.

Provides unified interface to quantum brain functions, infrastructure, and decision routing.

Integration: Main quantum interface for QuarkDriver and AutonomousAgent.
Rationale: Clean API abstraction over quantum computing modules.
"""

from .quantum_brain_functions import (
    create_quantum_enhanced_brain_functions,
    get_quantum_brain_recommendations,
    validate_quantum_brain_integration
)
from .quantum_infrastructure import QuantumInfrastructure, get_usage_report
from .quantum_decision_routing import QuantumDecisionRouter, ComputationType

# Main integration class that replaces the original QuarkQuantumIntegration
class QuarkQuantumIntegration:
    """Unified quantum integration interface."""

    def __init__(self):
        self.infrastructure = QuantumInfrastructure()
        self.router = QuantumDecisionRouter()

    def create_quantum_enhanced_brain_functions(self):
        """Create quantum-enhanced brain functions."""
        return create_quantum_enhanced_brain_functions()

    def setup_braket_integration(self):
        """Set up Braket integration."""
        return self.infrastructure.setup_braket_integration()

    def route_computation_intelligently(self, task_type: str, problem_size: int, max_cost: float = 1.0):
        """Route computation intelligently."""
        return self.router.route_computation_intelligently(task_type, problem_size, max_cost)

    def get_quantum_resource_status(self):
        """Get quantum resource status."""
        return self.infrastructure.get_quantum_resource_status()

    def validate_quantum_integration(self):
        """Validate quantum integration."""
        return self.infrastructure.validate_quantum_infrastructure()

    def get_usage_report(self):
        """Get usage report."""
        return get_usage_report()

# Export main interface
__all__ = [
    'QuarkQuantumIntegration',
    'QuantumInfrastructure',
    'QuantumDecisionRouter',
    'ComputationType',
    'create_quantum_enhanced_brain_functions',
    'get_quantum_brain_recommendations',
    'validate_quantum_brain_integration',
    'get_usage_report'
]
