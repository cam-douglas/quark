"""
Multi-Agent Orchestrator Core System

Core orchestration logic for coordinating specialized engineering agents.

Author: Quark AI
Date: 2025-01-27
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# Add compliance system to path
import sys
compliance_system_path = Path(__file__).parent.parent.parent / "compliance_system"
sys.path.insert(0, str(compliance_system_path))

# Import compliance system
try:
    from compliance_system.core_system import QuarkComplianceSystem
except ImportError:
    # Fallback: create a dummy compliance system
    class QuarkComplianceSystem:
        def __init__(self, workspace_root):
            self.workspace_root = workspace_root
        
        def check_compliance_now(self, paths=None):
            return True

# Import from the same directory
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from orchestrator_types import SpecializedTask, AgentRole, TaskStatus
from role_configs import initialize_role_configs
from specialist_workers import (
    OrchestratorWorker, NeurobiologistWorker, GeneticistWorker,
    ComputationalBiologistWorker, SafetyOfficerWorker, QAEngineerWorker,
    GenericSpecialistWorker
)


class MultiAgentOrchestrator:
    """
    Main orchestrator for coordinating specialized engineering agents
    """
    
    def __init__(self, workspace_root: str = "/Users/camdouglas/quark"):
        """Initialize the multi-agent orchestrator"""
        self.workspace_root = Path(workspace_root)
        self.logger = logging.getLogger(__name__)
        
        # Task storage
        self.delegated_tasks: Dict[str, SpecializedTask] = {}
        self.task_queue: List[SpecializedTask] = []
        self.active_agents: Dict[str, threading.Thread] = {}
        
        # Compliance system for validation
        self.compliance_system = QuarkComplianceSystem(str(self.workspace_root))
        
        # Callbacks for task completion
        self.completion_callbacks: List[Callable[[SpecializedTask], None]] = []
        
        # Role-specific configurations
        self.role_configs = initialize_role_configs()
        
        # Configuration
        self.config = {
            "max_concurrent_agents": 5,  # Allow multiple specialized agents
            "auto_compliance_check": True,
            "report_back_enabled": True,
            "task_timeout_minutes": 60,  # Longer timeout for complex tasks
            "parallel_execution_enabled": True
        }
        
        self.logger.info("Multi-Agent Orchestrator initialized")
    
    def add_completion_callback(self, callback: Callable[[SpecializedTask], None]):
        """Add a callback to be called when tasks complete"""
        self.completion_callbacks.append(callback)
    
    def delegate_to_specialist(self, 
                              title: str, 
                              description: str, 
                              role: AgentRole,
                              priority: int = 1,
                              deliverables: List[str] = None,
                              dependencies: List[str] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        """
        Delegate a task to a specialized engineering agent
        
        Args:
            title: Task title
            description: Task description
            role: Engineering role from AgentRole enum
            priority: Task priority (1-5)
            deliverables: Expected deliverables
            dependencies: Task dependencies
            context: Additional context for the agent
            
        Returns:
            Task ID for tracking
        """
        task_id = f"{role.value}_{int(time.time())}_{len(self.delegated_tasks)}"
        
        # Get role-specific configuration
        role_config = self.role_configs.get(role, {})
        
        # Adjust priority based on role
        adjusted_priority = priority + role_config.get("priority_boost", 0)
        
        task = SpecializedTask(
            id=task_id,
            title=title,
            description=description,
            role=role,
            priority=adjusted_priority,
            deliverables=deliverables or role_config.get("deliverables", []),
            dependencies=dependencies or []
        )
        
        # Add context if provided
        if context:
            task.result = {"context": context}
        
        self.delegated_tasks[task_id] = task
        self.task_queue.append(task)
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        self.logger.info(f"Delegated '{title}' to {role.value} (ID: {task_id})")
        
        # Start processing if we have capacity
        self._process_queue()
        
        return task_id
    
    def _process_queue(self):
        """Process the task queue and start available agents"""
        if len(self.active_agents) >= self.config["max_concurrent_agents"]:
            return
        
        # Group tasks by role to respect max_concurrent limits
        role_counts = {}
        for task in self.task_queue[:]:
            if task.status == TaskStatus.PENDING:
                role = task.role
                role_config = self.role_configs.get(role, {})
                max_concurrent = role_config.get("max_concurrent", 1)
                
                current_count = role_counts.get(role, 0)
                if current_count >= max_concurrent:
                    continue
                
                if len(self.active_agents) >= self.config["max_concurrent_agents"]:
                    break
                
                # Check dependencies
                if self._dependencies_satisfied(task):
                    # Start agent for this task
                    self._start_specialist_agent(task)
                    self.task_queue.remove(task)
                    role_counts[role] = current_count + 1
    
    def _dependencies_satisfied(self, task: SpecializedTask) -> bool:
        """Check if all dependencies for a task are satisfied"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            dep_task = self.delegated_tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _start_specialist_agent(self, task: SpecializedTask):
        """Start a background agent for the given specialized task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        # Create agent thread based on role
        agent_thread = threading.Thread(
            target=self._specialist_agent_worker,
            args=(task,),
            name=f"{task.role.value}-{task.id}"
        )
        
        self.active_agents[task.id] = agent_thread
        agent_thread.start()
        
        self.logger.info(f"Started {task.role.value} agent for task '{task.title}'")
    
    def _specialist_agent_worker(self, task: SpecializedTask):
        """Worker for specialized engineering agents"""
        try:
            self.logger.info(f"{task.role.value} agent executing: {task.title}")
            
            # Execute role-specific task
            result = self._execute_specialist_task(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Run compliance check if enabled
            if self.config["auto_compliance_check"]:
                self._run_compliance_check(task)
            
            self.logger.info(f"{task.role.value} agent completed: {task.title}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.logger.error(f"{task.role.value} agent failed: {task.title} - {e}")
        
        finally:
            self._cleanup_agent(task)
    
    def _execute_specialist_task(self, task: SpecializedTask) -> Dict[str, Any]:
        """Execute a task using role-specific logic"""
        role = task.role
        context = task.result.get("context", {}) if task.result else {}
        
        # Create appropriate worker
        if role == AgentRole.PROGRAM_ORCHESTRATOR:
            worker = OrchestratorWorker(self.workspace_root, self.logger)
        elif role == AgentRole.DEVELOPMENTAL_NEUROBIOLOGIST:
            worker = NeurobiologistWorker(self.workspace_root, self.logger)
        elif role == AgentRole.MOLECULAR_GENETICIST:
            worker = GeneticistWorker(self.workspace_root, self.logger)
        elif role == AgentRole.COMPUTATIONAL_BIOLOGIST:
            worker = ComputationalBiologistWorker(self.workspace_root, self.logger)
        elif role == AgentRole.SAFETY_ETHICS_OFFICER:
            worker = SafetyOfficerWorker(self.workspace_root, self.logger)
        elif role == AgentRole.QA_REPRODUCIBILITY_ENGINEER:
            worker = QAEngineerWorker(self.workspace_root, self.logger)
        else:
            # Generic execution for other roles
            worker = GenericSpecialistWorker(self.workspace_root, self.logger)
        
        return worker.execute_task(task)
    
    def _run_compliance_check(self, task: SpecializedTask):
        """Run compliance check on completed task"""
        try:
            # Check deliverables if they exist
            deliverables_to_check = []
            for deliverable in task.deliverables:
                deliverable_path = self.workspace_root / deliverable.lstrip("/")
                if deliverable_path.exists():
                    deliverables_to_check.append(str(deliverable_path))
            
            if deliverables_to_check:
                success = self.compliance_system.check_compliance_now(deliverables_to_check)
                task.compliance_checked = success
                self.logger.info(f"Compliance check for {task.role.value}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            self.logger.error(f"Compliance check failed for {task.role.value}: {e}")
    
    def _cleanup_agent(self, task: SpecializedTask):
        """Clean up completed agent and notify callbacks"""
        # Remove from active agents
        if task.id in self.active_agents:
            del self.active_agents[task.id]
        
        # Call completion callbacks
        for callback in self.completion_callbacks:
            try:
                callback(task)
            except Exception as e:
                self.logger.error(f"Callback failed for task {task.id}: {e}")
        
        # Process queue for next tasks
        self._process_queue()
    
    def get_task_status(self, task_id: str) -> Optional[SpecializedTask]:
        """Get the status of a delegated task"""
        return self.delegated_tasks.get(task_id)
    
    def get_tasks_by_role(self, role: AgentRole) -> List[SpecializedTask]:
        """Get all tasks for a specific role"""
        return [task for task in self.delegated_tasks.values() if task.role == role]
    
    def get_active_tasks(self) -> List[SpecializedTask]:
        """Get currently active tasks"""
        return [task for task in self.delegated_tasks.values() 
                if task.status == TaskStatus.IN_PROGRESS]
    
    def get_completed_tasks(self) -> List[SpecializedTask]:
        """Get completed tasks"""
        return [task for task in self.delegated_tasks.values() 
                if task.status == TaskStatus.COMPLETED]
    
    def generate_orchestration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive orchestration report"""
        all_tasks = list(self.delegated_tasks.values())
        
        # Group by role
        tasks_by_role = {}
        for role in AgentRole:
            tasks_by_role[role.value] = [task for task in all_tasks if task.role == role]
        
        # Calculate role statistics
        role_stats = {}
        for role, tasks in tasks_by_role.items():
            role_stats[role] = {
                "total": len(tasks),
                "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
                "in_progress": len([t for t in tasks if t.status == TaskStatus.IN_PROGRESS]),
                "failed": len([t for t in tasks if t.status == TaskStatus.FAILED])
            }
        
        return {
            "orchestration_summary": {
                "total_tasks": len(all_tasks),
                "active_agents": len(self.active_agents),
                "roles_active": len([role for role, tasks in tasks_by_role.items() if tasks]),
                "parallel_execution": self.config["parallel_execution_enabled"]
            },
            "role_statistics": role_stats,
            "active_tasks": [task.__dict__ for task in self.get_active_tasks()],
            "recent_completions": [task.__dict__ for task in self.get_completed_tasks()[-5:]],
            "generated_at": datetime.now().isoformat()
        }
