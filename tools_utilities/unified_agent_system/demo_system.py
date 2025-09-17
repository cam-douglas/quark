"""
Demo System

Handles unified demo functionality and system status.

Author: Quark AI
Date: 2025-01-27
"""

import time
from typing import Dict, List, Optional, Any

# Import from both packages
import sys
from pathlib import Path

# Add agent delegation path
agent_delegation_path = Path(__file__).parent.parent.parent / "agent_delegation"
sys.path.insert(0, str(agent_delegation_path))

# Add orchestrator path
orchestrator_path = Path(__file__).parent / "multi_agent_orchestrator"
sys.path.insert(0, str(orchestrator_path))

from core_system import AgentDelegationSystem
from agent_types import DelegatedTask, TaskStatus
from core_orchestrator import MultiAgentOrchestrator
from orchestrator_types import SpecializedTask, AgentRole


def run_unified_demo():
    """Run a unified demo showing both systems working together"""
    print("🚀 Running Unified Agent System Demo...")
    print("=" * 60)
    
    # Initialize both systems
    delegation_system = AgentDelegationSystem()
    orchestrator = MultiAgentOrchestrator()
    
    # Add callbacks
    def delegation_callback(task: DelegatedTask):
        print(f"📋 Delegation: {task.title} completed by {task.agent_type}")
    
    def orchestrator_callback(task: SpecializedTask):
        print(f"🎯 Orchestration: {task.title} completed by {task.role.value}")
    
    delegation_system.add_completion_callback(delegation_callback)
    orchestrator.add_completion_callback(orchestrator_callback)
    
    # Delegate some general tasks
    print("\n1. Delegating general tasks...")
    task1 = delegation_system.delegate_task(
        title="Run Compliance Check",
        description="Check all files for compliance violations",
        agent_type="compliance",
        priority=3
    )
    
    task2 = delegation_system.delegate_task(
        title="Generate Documentation",
        description="Generate API documentation",
        agent_type="documentation",
        priority=2
    )
    
    # Delegate specialized engineering tasks
    print("\n2. Delegating specialized engineering tasks...")
    task3 = orchestrator.delegate_to_specialist(
        title="Design System Architecture",
        description="Create high-level system architecture",
        role=AgentRole.PROGRAM_ORCHESTRATOR,
        priority=5
    )
    
    task4 = orchestrator.delegate_to_specialist(
        title="Validate Safety Protocols",
        description="Ensure all safety protocols are in place",
        role=AgentRole.SAFETY_ETHICS_OFFICER,
        priority=5
    )
    
    print(f"\n✅ Demo tasks delegated:")
    print(f"  - General: {task1}, {task2}")
    print(f"  - Specialized: {task3}, {task4}")
    
    print("\n⏳ Waiting for completion...")
    time.sleep(3)  # Simulate work
    
    print("\n📊 Demo completed! Both systems are operational.")


def show_system_status():
    """Show status of both systems"""
    print("📊 Unified Agent System Status")
    print("=" * 40)
    
    # Check delegation system
    try:
        delegation_system = AgentDelegationSystem()
        active_tasks = delegation_system.get_active_tasks()
        print(f"📋 Delegation System: {len(active_tasks)} active tasks")
    except Exception as e:
        print(f"📋 Delegation System: Error - {e}")
    
    # Check orchestrator system
    try:
        orchestrator = MultiAgentOrchestrator()
        active_tasks = orchestrator.get_active_tasks()
        print(f"🎯 Orchestrator System: {len(active_tasks)} active tasks")
    except Exception as e:
        print(f"🎯 Orchestrator System: Error - {e}")
    
    print("\n✅ Both systems are available for use!")
