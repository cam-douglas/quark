"""
Cline-Quark Task Integration Core

Core integration layer between Cline autonomous coding and Quark State System.
Provides seamless integration with existing task loader while maintaining
sprint-batch-task management structure.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import your existing task management system
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))

from state.quark_state_system.task_management.task_loader import (
    get_tasks, next_actions, mark_task_complete, get_sprint_summary
)
from brain.modules.cline_integration.cline_adapter import (
    ClineAdapter, CodingTask, TaskComplexity, ClineTaskType
)


class QuarkClineIntegration:
    """
    Core integration between Cline autonomous coding and Quark State System
    
    Enhances existing task loader with autonomous execution while preserving
    sprint-batch-task management structure.
    """
    
    def __init__(self):
        """Initialize Quark-Cline integration"""
        self.cline_adapter = ClineAdapter()
        self.logger = logging.getLogger(__name__)
        
        # Configuration for autonomous execution
        self.config = {
            "auto_execute_enabled": True,
            "max_concurrent_tasks": 2,
            "biological_validation_required": True,
            "foundation_layer_priority": True
        }
        
        # Track autonomous executions
        self.execution_history: List[Dict[str, Any]] = []
        
        self.logger.info("Quark-Cline integration initialized")
    
    def can_execute_autonomously(self, task: Dict[str, Any]) -> bool:
        """
        Determine if a task can be executed autonomously by Cline
        
        Args:
            task: Task dictionary from existing task loader
            
        Returns:
            Boolean indicating if task is suitable for autonomous execution
        """
        if not self.config["auto_execute_enabled"]:
            return False
        
        # Check task properties
        title = task.get("title", "").lower()
        description = task.get("description", "").lower()
        
        # Tasks suitable for autonomous execution
        autonomous_keywords = [
            "implement", "create", "develop", "build", "generate",
            "morphogen", "gradient", "bmp", "wnt", "fgf", "ventricular",
            "validation", "testing", "integration"
        ]
        
        # Tasks that should NOT be executed autonomously
        manual_keywords = [
            "review", "decide", "plan", "design", "architecture",
            "strategy", "meeting", "discussion", "approval"
        ]
        
        # Check if task contains autonomous-friendly keywords
        has_autonomous_keywords = any(keyword in title or keyword in description 
                                    for keyword in autonomous_keywords)
        
        # Check if task requires manual intervention
        has_manual_keywords = any(keyword in title or keyword in description 
                                for keyword in manual_keywords)
        
        # Foundation Layer tasks get priority for autonomous execution
        is_foundation_layer = any(term in title or term in description 
                                for term in ["foundation", "morphogen", "gradient", "shh", "bmp"])
        
        # Decision logic
        if has_manual_keywords:
            return False
        
        if is_foundation_layer and self.config["foundation_layer_priority"]:
            return True
        
        return has_autonomous_keywords
    
    async def execute_next_autonomous_task(self) -> Optional[Dict[str, Any]]:
        """
        Execute the next available task autonomously using Cline
        
        Returns:
            Execution result dictionary or None if no suitable tasks
        """
        # Import task conversion module
        from .task_converter import convert_quark_task_to_cline_task
        
        # Get next actions from existing system
        next_tasks = next_actions(limit=10)
        
        if not next_tasks:
            self.logger.info("No pending tasks available for execution")
            return None
        
        # Find first task suitable for autonomous execution
        suitable_task = None
        for task in next_tasks:
            if self.can_execute_autonomously(task):
                suitable_task = task
                break
        
        if not suitable_task:
            self.logger.info("No tasks suitable for autonomous execution")
            return None
        
        self.logger.info(f"Executing task autonomously: {suitable_task.get('title')}")
        
        try:
            # Convert to Cline task
            cline_task = convert_quark_task_to_cline_task(suitable_task)
            
            # Execute via Cline
            result = await self.cline_adapter.autonomous_code_generation(cline_task)
            
            # Process results
            execution_result = {
                "quark_task_id": suitable_task.get("id"),
                "quark_task_title": suitable_task.get("title"),
                "cline_result": result,
                "timestamp": datetime.now().isoformat(),
                "success": result.success,
                "biological_compliance": result.biological_compliance
            }
            
            # If successful, mark task as complete in Quark system
            if result.success:
                task_id = suitable_task.get("id")
                if task_id:
                    mark_task_complete(task_id)
                    self.logger.info(f"Task {task_id} marked as complete in Quark system")
            
            # Store execution history
            self.execution_history.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Autonomous execution failed: {e}")
            return {
                "quark_task_id": suitable_task.get("id"),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_specific_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute a specific task by ID using Cline
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Execution result dictionary
        """
        # Import task conversion module
        from .task_converter import convert_quark_task_to_cline_task
        
        # Find task in current task list
        all_tasks = get_tasks()
        target_task = None
        
        for task in all_tasks:
            if task.get("id") == task_id or task_id in task.get("title", ""):
                target_task = task
                break
        
        if not target_task:
            self.logger.error(f"Task {task_id} not found")
            return None
        
        if not self.can_execute_autonomously(target_task):
            self.logger.warning(f"Task {task_id} not suitable for autonomous execution")
            return None
        
        # Convert and execute
        cline_task = convert_quark_task_to_cline_task(target_task)
        result = await self.cline_adapter.autonomous_code_generation(cline_task)
        
        # Mark complete if successful
        if result.success:
            mark_task_complete(task_id)
        
        return {
            "quark_task_id": task_id,
            "success": result.success,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_autonomous_execution_status(self) -> Dict[str, Any]:
        """
        Get status of autonomous execution capabilities
        
        Returns:
            Status dictionary with execution metrics and available tasks
        """
        # Get current tasks from your system
        all_tasks = get_tasks()
        pending_tasks = get_tasks(status="pending")
        
        # Analyze autonomous execution potential
        autonomous_ready = [
            task for task in pending_tasks 
            if self.can_execute_autonomously(task)
        ]
        
        # Get current sprint summary
        sprint_summary = get_sprint_summary()
        
        return {
            "total_tasks": len(all_tasks),
            "pending_tasks": len(pending_tasks),
            "autonomous_ready_tasks": len(autonomous_ready),
            "autonomous_ready_list": [
                {
                    "id": task.get("id"),
                    "title": task.get("title"),
                    "phase": task.get("phase"),
                    "priority": task.get("priority")
                }
                for task in autonomous_ready[:5]  # Show top 5
            ],
            "execution_history_count": len(self.execution_history),
            "recent_executions": self.execution_history[-3:],  # Last 3
            "sprint_summary": sprint_summary,
            "config": self.config
        }


# Convenience functions for easy integration

async def execute_foundation_layer_tasks_autonomously(max_tasks: int = 3) -> List[Dict[str, Any]]:
    """
    Execute Foundation Layer tasks autonomously using existing Quark task system
    
    Args:
        max_tasks: Maximum number of tasks to execute
        
    Returns:
        List of execution results
    """
    integration = QuarkClineIntegration()
    results = []
    
    for _ in range(max_tasks):
        result = await integration.execute_next_autonomous_task()
        if result:
            results.append(result)
        else:
            break  # No more suitable tasks
    
    return results

async def execute_task_by_name(task_name: str) -> Optional[Dict[str, Any]]:
    """Execute a specific task by name using Cline"""
    integration = QuarkClineIntegration()
    
    # Find task ID from name
    all_tasks = get_tasks()
    for task in all_tasks:
        if task_name.lower() in task.get("title", "").lower():
            return await integration.execute_specific_task(task.get("id"))
    
    return None
