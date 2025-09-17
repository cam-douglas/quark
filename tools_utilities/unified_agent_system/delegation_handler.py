"""
Delegation System Handler

Handles the general agent delegation system functionality.

Author: Quark AI
Date: 2025-01-27
"""

import json
import time
from typing import Dict, List, Optional, Any

# Import from the agent delegation package
import sys
from pathlib import Path
agent_delegation_path = Path(__file__).parent.parent.parent / "agent_delegation"
sys.path.insert(0, str(agent_delegation_path))

from core_system import AgentDelegationSystem
from agent_types import DelegatedTask, TaskStatus


def run_delegation_system(args):
    """Run the general agent delegation system"""
    # Initialize delegation system
    delegation_system = AgentDelegationSystem()
    
    # Add completion callback
    def completion_callback(task: DelegatedTask):
        print(f"🎉 Task completed: {task.title} ({task.agent_type})")
        if task.status == TaskStatus.COMPLETED:
            print(f"✅ Success: {task.result}")
        else:
            print(f"❌ Failed: {task.error_message}")
    
    delegation_system.add_completion_callback(completion_callback)
    
    if args.delegate:
        title, description, agent_type, priority = args.delegate
        task_id = delegation_system.delegate_task(
            title=title,
            description=description,
            agent_type=agent_type,
            priority=int(priority)
        )
        print(f"Delegated task '{title}' with ID: {task_id}")
        
        # Wait for completion (in real usage, this would be non-blocking)
        while True:
            task = delegation_system.get_task_status(task_id)
            if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            time.sleep(1)
    
    elif args.status:
        task = delegation_system.get_task_status(args.status)
        if task:
            print(f"Task: {task.title}")
            print(f"Status: {task.status.value}")
            print(f"Agent: {task.agent_type}")
            if task.result:
                print(f"Result: {task.result}")
        else:
            print("Task not found")
    
    elif args.list:
        tasks = delegation_system.get_all_tasks()
        for task in tasks:
            print(f"{task.id}: {task.title} ({task.status.value}) - {task.agent_type}")
    
    elif args.report:
        report = delegation_system.generate_report()
        print(json.dumps(report, indent=2, default=str))
    
    elif args.cancel:
        success = delegation_system.cancel_task(args.cancel)
        print(f"Task cancellation: {'Success' if success else 'Failed'}")
