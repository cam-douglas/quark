"""
Multi-Agent Orchestrator Package

Modular system for coordinating specialized engineering agents based on the
engineering roles defined in .quark/rules/engineering-roles.mdc.

Author: Quark AI
Date: 2025-01-27
"""

from .core_orchestrator import MultiAgentOrchestrator, SpecializedTask, AgentRole, TaskStatus
from .role_configs import initialize_role_configs
from .specialist_workers import (
    OrchestratorWorker, NeurobiologistWorker, GeneticistWorker,
    ComputationalBiologistWorker, SafetyOfficerWorker, QAEngineerWorker
)

__all__ = [
    'MultiAgentOrchestrator',
    'SpecializedTask',
    'AgentRole', 
    'TaskStatus',
    'initialize_role_configs',
    'OrchestratorWorker',
    'NeurobiologistWorker',
    'GeneticistWorker',
    'ComputationalBiologistWorker',
    'SafetyOfficerWorker',
    'QAEngineerWorker'
]
