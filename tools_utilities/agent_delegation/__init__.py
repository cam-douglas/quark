"""
Agent Delegation System Package

Modular system for delegating tasks to background agents with reporting back
to the leading agent when tasks are completed.

Author: Quark AI
Date: 2025-01-27
"""

from .core_system import AgentDelegationSystem, DelegatedTask, TaskStatus
from .agent_workers import (
    ClineAgentWorker, ComplianceAgentWorker, 
    TestingAgentWorker, DocumentationAgentWorker
)

__all__ = [
    'AgentDelegationSystem',
    'DelegatedTask', 
    'TaskStatus',
    'ClineAgentWorker',
    'ComplianceAgentWorker',
    'TestingAgentWorker', 
    'DocumentationAgentWorker'
]
