"""
Agent Delegation Core System

Core orchestration logic for delegating tasks to background agents.

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
compliance_system_path = Path(__file__).parent.parent / "compliance_system"
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

# Handle both relative and absolute imports
try:
    from .agent_types import DelegatedTask, TaskStatus
    from .agent_workers import (
        ClineAgentWorker, ComplianceAgentWorker, 
        TestingAgentWorker, DocumentationAgentWorker
    )
except ImportError:
    # Import from the same directory
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from agent_types import DelegatedTask, TaskStatus
    from agent_workers import (
        ClineAgentWorker, ComplianceAgentWorker, 
        TestingAgentWorker, DocumentationAgentWorker
    )


class AgentDelegationSystem:
    """
    Main system for delegating tasks to background agents
    """
    
    def __init__(self, workspace_root: str = "/Users/camdouglas/quark"):
        """Initialize the delegation system"""
        self.workspace_root = Path(workspace_root)
        self.logger = logging.getLogger(__name__)
        
        # Task storage
        self.delegated_tasks: Dict[str, DelegatedTask] = {}
        self.task_queue: List[DelegatedTask] = []
        self.active_agents: Dict[str, threading.Thread] = {}
        
        # Compliance system for validation
        self.compliance_system = QuarkComplianceSystem(str(self.workspace_root))
        
        # Callbacks for task completion
        self.completion_callbacks: List[Callable[[DelegatedTask], None]] = []
        
        # Configuration
        self.config = {
            "max_concurrent_agents": 3,
            "auto_compliance_check": True,
            "report_back_enabled": True,
            "task_timeout_minutes": 30
        }
        
        self.logger.info("Agent Delegation System initialized")
    
    def add_completion_callback(self, callback: Callable[[DelegatedTask], None]):
        """Add a callback to be called when tasks complete"""
        self.completion_callbacks.append(callback)
    
    def delegate_task(self, 
                     title: str, 
                     description: str, 
                     agent_type: str,
                     priority: int = 1,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Delegate a task to a background agent
        
        Args:
            title: Task title
            description: Task description
            agent_type: Type of agent ("cline", "compliance", "testing", "documentation")
            priority: Task priority (1-5)
            context: Additional context for the agent
            
        Returns:
            Task ID for tracking
        """
        task_id = f"{agent_type}_{int(time.time())}_{len(self.delegated_tasks)}"
        
        task = DelegatedTask(
            id=task_id,
            title=title,
            description=description,
            agent_type=agent_type,
            priority=priority
        )
        
        # Add context if provided
        if context:
            task.result = {"context": context}
        
        self.delegated_tasks[task_id] = task
        self.task_queue.append(task)
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        self.logger.info(f"Delegated task '{title}' to {agent_type} agent (ID: {task_id})")
        
        # Start processing if we have capacity
        self._process_queue()
        
        return task_id
    
    def _process_queue(self):
        """Process the task queue and start available agents"""
        if len(self.active_agents) >= self.config["max_concurrent_agents"]:
            return
        
        for task in self.task_queue[:]:
            if task.status == TaskStatus.PENDING:
                if len(self.active_agents) >= self.config["max_concurrent_agents"]:
                    break
                
                # Start agent for this task
                self._start_agent(task)
                self.task_queue.remove(task)
    
    def _start_agent(self, task: DelegatedTask):
        """Start a background agent for the given task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        # Create agent thread based on type
        if task.agent_type == "cline":
            agent_thread = threading.Thread(
                target=self._cline_agent_worker,
                args=(task,),
                name=f"ClineAgent-{task.id}"
            )
        elif task.agent_type == "compliance":
            agent_thread = threading.Thread(
                target=self._compliance_agent_worker,
                args=(task,),
                name=f"ComplianceAgent-{task.id}"
            )
        elif task.agent_type == "testing":
            agent_thread = threading.Thread(
                target=self._testing_agent_worker,
                args=(task,),
                name=f"TestingAgent-{task.id}"
            )
        elif task.agent_type == "documentation":
            agent_thread = threading.Thread(
                target=self._documentation_agent_worker,
                args=(task,),
                name=f"DocumentationAgent-{task.id}"
            )
        else:
            self.logger.error(f"Unknown agent type: {task.agent_type}")
            task.status = TaskStatus.FAILED
            task.error_message = f"Unknown agent type: {task.agent_type}"
            return
        
        self.active_agents[task.id] = agent_thread
        agent_thread.start()
        
        self.logger.info(f"Started {task.agent_type} agent for task '{task.title}'")
    
    def _cline_agent_worker(self, task: DelegatedTask):
        """Worker for Cline autonomous coding agent"""
        try:
            worker = ClineAgentWorker(self.workspace_root, self.logger)
            result = worker.execute_task(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            self.logger.info(f"Cline agent completed: {task.title}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.logger.error(f"Cline agent failed: {task.title} - {e}")
        
        finally:
            self._cleanup_agent(task)
    
    def _compliance_agent_worker(self, task: DelegatedTask):
        """Worker for compliance checking agent"""
        try:
            worker = ComplianceAgentWorker(self.workspace_root, self.logger, self.compliance_system)
            result = worker.execute_task(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.compliance_checked = True
            
            self.logger.info(f"Compliance agent completed: {task.title}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.logger.error(f"Compliance agent failed: {task.title} - {e}")
        
        finally:
            self._cleanup_agent(task)
    
    def _testing_agent_worker(self, task: DelegatedTask):
        """Worker for testing agent"""
        try:
            worker = TestingAgentWorker(self.workspace_root, self.logger)
            result = worker.execute_task(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED if result.get("test_results", {}).get("tests_passed", False) else TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            self.logger.info(f"Testing agent completed: {task.title}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.logger.error(f"Testing agent failed: {task.title} - {e}")
        
        finally:
            self._cleanup_agent(task)
    
    def _documentation_agent_worker(self, task: DelegatedTask):
        """Worker for documentation agent"""
        try:
            worker = DocumentationAgentWorker(self.workspace_root, self.logger)
            result = worker.execute_task(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED if result.get("documentation_generated", False) else TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            self.logger.info(f"Documentation agent completed: {task.title}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.logger.error(f"Documentation agent failed: {task.title} - {e}")
        
        finally:
            self._cleanup_agent(task)
    
    def _cleanup_agent(self, task: DelegatedTask):
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
    
    def get_task_status(self, task_id: str) -> Optional[DelegatedTask]:
        """Get the status of a delegated task"""
        return self.delegated_tasks.get(task_id)
    
    def get_all_tasks(self) -> List[DelegatedTask]:
        """Get all delegated tasks"""
        return list(self.delegated_tasks.values())
    
    def get_active_tasks(self) -> List[DelegatedTask]:
        """Get currently active tasks"""
        return [task for task in self.delegated_tasks.values() 
                if task.status == TaskStatus.IN_PROGRESS]
    
    def get_completed_tasks(self) -> List[DelegatedTask]:
        """Get completed tasks"""
        return [task for task in self.delegated_tasks.values() 
                if task.status == TaskStatus.COMPLETED]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or in-progress task"""
        task = self.delegated_tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            if task in self.task_queue:
                self.task_queue.remove(task)
            return True
        
        elif task.status == TaskStatus.IN_PROGRESS:
            # Note: Cannot easily cancel running threads, but mark as cancelled
            task.status = TaskStatus.CANCELLED
            return True
        
        return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all tasks"""
        all_tasks = self.get_all_tasks()
        
        return {
            "summary": {
                "total_tasks": len(all_tasks),
                "pending": len([t for t in all_tasks if t.status == TaskStatus.PENDING]),
                "in_progress": len([t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS]),
                "completed": len([t for t in all_tasks if t.status == TaskStatus.COMPLETED]),
                "failed": len([t for t in all_tasks if t.status == TaskStatus.FAILED]),
                "cancelled": len([t for t in all_tasks if t.status == TaskStatus.CANCELLED])
            },
            "active_agents": len(self.active_agents),
            "tasks": [task.__dict__ for task in all_tasks],
            "generated_at": datetime.now().isoformat()
        }
