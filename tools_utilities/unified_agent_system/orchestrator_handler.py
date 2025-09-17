"""
Orchestrator System Handler

Handles the specialized engineering role orchestrator functionality.

Author: Quark AI
Date: 2025-01-27
"""

import json
import time
from typing import Dict, List, Optional, Any

# Import from the orchestrator package
import sys
from pathlib import Path
orchestrator_path = Path(__file__).parent / "multi_agent_orchestrator"
sys.path.insert(0, str(orchestrator_path))

from core_orchestrator import MultiAgentOrchestrator
from orchestrator_types import SpecializedTask, AgentRole, TaskStatus


def run_orchestrator_system(args):
    """Run the specialized engineering role orchestrator"""
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Add completion callback
    def completion_callback(task: SpecializedTask):
        print(f"🎉 {task.role.value} completed: {task.title}")
        if task.status == TaskStatus.COMPLETED:
            print(f"✅ Deliverables: {task.deliverables}")
            if task.compliance_checked:
                print("✅ Compliance check passed")
        else:
            print(f"❌ Failed: {task.error_message}")
    
    orchestrator.add_completion_callback(completion_callback)
    
    if args.example:
        # Run example orchestration
        print("🚀 Running Multi-Agent Orchestration Example...")
        
        # Delegate tasks to different specialists
        task1 = orchestrator.delegate_to_specialist(
            title="Design Neurulation Architecture",
            description="Create the foundational architecture for neurulation modeling",
            role=AgentRole.PROGRAM_ORCHESTRATOR,
            priority=5,
            deliverables=["/docs/architecture.md", "/docs/roadmap.md"]
        )
        
        task2 = orchestrator.delegate_to_specialist(
            title="Model BMP Gradient Dynamics",
            description="Implement BMP gradient computation for neural tube formation",
            role=AgentRole.DEVELOPMENTAL_NEUROBIOLOGIST,
            priority=4,
            deliverables=["/biology/neurulation_model.cc3d"]
        )
        
        task3 = orchestrator.delegate_to_specialist(
            title="Validate Safety Guardrails",
            description="Ensure all biological constraints are properly enforced",
            role=AgentRole.SAFETY_ETHICS_OFFICER,
            priority=5,
            deliverables=["/safety/guardrails.py"]
        )
        
        print(f"Delegated 3 tasks to specialized agents:")
        print(f"  - Program Orchestrator: {task1}")
        print(f"  - Developmental Neurobiologist: {task2}")
        print(f"  - Safety & Ethics Officer: {task3}")
        
        # Wait for completion
        print("\n⏳ Waiting for task completion...")
        while True:
            active_tasks = orchestrator.get_active_tasks()
            if not active_tasks:
                break
            print(f"Active agents: {len(active_tasks)}")
            time.sleep(2)
        
        print("\n📊 Final Orchestration Report:")
        report = orchestrator.generate_orchestration_report()
        print(json.dumps(report, indent=2, default=str))
    
    elif args.delegate:
        title, description, role_str, priority = args.delegate
        try:
            role = AgentRole(role_str)
            task_id = orchestrator.delegate_to_specialist(
                title=title,
                description=description,
                role=role,
                priority=int(priority)
            )
            print(f"Delegated task '{title}' to {role.value} with ID: {task_id}")
        except ValueError:
            print(f"Invalid role: {role_str}")
            print("Available roles:", [r.value for r in AgentRole])
    
    elif args.status:
        task = orchestrator.get_task_status(args.status)
        if task:
            print(f"Task: {task.title}")
            print(f"Role: {task.role.value}")
            print(f"Status: {task.status.value}")
            print(f"Deliverables: {task.deliverables}")
            if task.result:
                print(f"Result: {task.result}")
        else:
            print("Task not found")
    
    elif args.list:
        tasks = orchestrator.get_all_tasks()
        for task in tasks:
            print(f"{task.id}: {task.title} ({task.role.value}) - {task.status.value}")
    
    elif args.role_tasks:
        try:
            role = AgentRole(args.role_tasks)
            tasks = orchestrator.get_tasks_by_role(role)
            print(f"Tasks for {role.value}:")
            for task in tasks:
                print(f"  {task.id}: {task.title} - {task.status.value}")
        except ValueError:
            print(f"Invalid role: {args.role_tasks}")
    
    elif args.report:
        report = orchestrator.generate_orchestration_report()
        print(json.dumps(report, indent=2, default=str))
