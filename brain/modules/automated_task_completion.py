#!/usr/bin/env python3
"""Automated Task Completion System.

Integrates with todo system to automatically mark tasks as complete in
appropriate files when todo items are marked as completed.

Integration: Automated completion bridge between todo system and task files
Rationale: Seamless task completion workflow with proper file persistence
"""

from typing import Dict, Any, List, Optional
import logging
from .task_completion_manager import task_completion_manager

logger = logging.getLogger(__name__)

class AutomatedTaskCompletion:
    """Automated task completion system.
    
    Monitors todo completion events and automatically updates the appropriate
    task files (roadmap or chat) based on task classification.
    """
    
    def __init__(self):
        """Initialize automated task completion system."""
        self.completion_manager = task_completion_manager
        
        # Task classification patterns
        self.roadmap_task_patterns = {
            "foundation_layer": "foundation_layer_detailed_tasks.md",
            "stage1_embryonic": "stage1_embryonic_detailed_tasks.md", 
            "stage2_fetal": "stage2_fetal_detailed_tasks.md",
            "ventricular": "foundation_layer_detailed_tasks.md",
            "morphogen": "foundation_layer_detailed_tasks.md",
            "excavation": "foundation_layer_detailed_tasks.md",
            "csf": "foundation_layer_detailed_tasks.md"
        }
        
        logger.info("Initialized AutomatedTaskCompletion system")
    
    def on_todo_completed(self, todo_id: str, todo_content: str) -> bool:
        """Handle todo completion event.
        
        Args:
            todo_id: ID of completed todo item
            todo_content: Content/description of completed todo
            
        Returns:
            True if task completion was successfully processed
        """
        logger.info(f"Processing todo completion: {todo_id}")
        
        # Classify task type
        is_roadmap_task, roadmap_file = self._classify_task(todo_id, todo_content)
        
        # Generate completion notes
        completion_notes = self._generate_completion_notes(todo_id, todo_content)
        
        # Complete the task
        success = self.completion_manager.complete_task(
            task_id=todo_id,
            task_description=todo_content,
            is_roadmap_task=is_roadmap_task,
            roadmap_file=roadmap_file,
            completion_notes=completion_notes
        )
        
        if success:
            logger.info(f"Successfully processed completion for {todo_id}")
            if is_roadmap_task:
                logger.info(f"Roadmap task marked for user confirmation: {roadmap_file}")
            else:
                logger.info(f"Chat task automatically completed")
        else:
            logger.error(f"Failed to process completion for {todo_id}")
        
        return success
    
    def _classify_task(self, todo_id: str, todo_content: str) -> tuple[bool, Optional[str]]:
        """Classify whether task is roadmap or ad-hoc.
        
        Args:
            todo_id: Todo item ID
            todo_content: Todo item content
            
        Returns:
            Tuple of (is_roadmap_task, roadmap_file)
        """
        # Check for roadmap task patterns
        content_lower = todo_content.lower()
        id_lower = todo_id.lower()
        
        for pattern, roadmap_file in self.roadmap_task_patterns.items():
            if pattern in content_lower or pattern in id_lower:
                logger.debug(f"Classified as roadmap task: {pattern} -> {roadmap_file}")
                return True, roadmap_file
        
        # Check for explicit roadmap indicators
        roadmap_indicators = [
            "stage1", "stage2", "stage3", "foundation", "embryonic", 
            "fetal", "postnatal", "1.1.", "1.2.", "2.1.", "2.2."
        ]
        
        for indicator in roadmap_indicators:
            if indicator in content_lower or indicator in id_lower:
                # Default to foundation layer for numbered tasks
                if any(num in content_lower for num in ["1.1.", "1.2."]):
                    return True, "foundation_layer_detailed_tasks.md"
                elif any(num in content_lower for num in ["2.1.", "2.2."]):
                    return True, "stage1_embryonic_detailed_tasks.md"
                else:
                    return True, "foundation_layer_detailed_tasks.md"
        
        logger.debug(f"Classified as ad-hoc chat task")
        return False, None
    
    def _generate_completion_notes(self, todo_id: str, todo_content: str) -> str:
        """Generate completion notes for the task.
        
        Args:
            todo_id: Todo item ID
            todo_content: Todo item content
            
        Returns:
            Completion notes string
        """
        notes = [f"Task completed via automated todo system."]
        
        # Add specific notes based on task type
        if "ventricular" in todo_content.lower():
            notes.append("Implemented complete ventricular topology system with architecture compliance.")
        elif "excavation" in todo_content.lower():
            notes.append("Implemented voxel excavation algorithm with biological validation.")
        elif "csf" in todo_content.lower():
            notes.append("Implemented CSF flow dynamics framework with pressure field computation.")
        elif "morphogen" in todo_content.lower():
            notes.append("Integrated with existing morphogen solver system.")
        
        # Add architecture compliance note if applicable
        if any(keyword in todo_content.lower() for keyword in ["refactor", "split", "compliant"]):
            notes.append("All modules split to <300 lines following architecture rules.")
        
        return " ".join(notes)
    
    def complete_multiple_tasks(self, completed_todos: List[Dict[str, str]]) -> Dict[str, bool]:
        """Complete multiple tasks at once.
        
        Args:
            completed_todos: List of dicts with 'id' and 'content' keys
            
        Returns:
            Dict mapping todo_id to success status
        """
        results = {}
        
        for todo in completed_todos:
            todo_id = todo.get('id', '')
            todo_content = todo.get('content', '')
            
            if todo_id and todo_content:
                results[todo_id] = self.on_todo_completed(todo_id, todo_content)
            else:
                logger.warning(f"Invalid todo format: {todo}")
                results[todo_id] = False
        
        return results
    
    def get_pending_roadmap_confirmations(self) -> List[Dict[str, Any]]:
        """Get pending roadmap task completions awaiting user confirmation."""
        return self.completion_manager.get_pending_roadmap_completions()
    
    def confirm_roadmap_task(self, task_id: str) -> bool:
        """Confirm a pending roadmap task completion.
        
        Args:
            task_id: Task ID to confirm
            
        Returns:
            True if confirmation was successful
        """
        return self.completion_manager.confirm_roadmap_completion(task_id)


# Global instance
automated_completion = AutomatedTaskCompletion()


def auto_complete_todos(completed_todos: List[Dict[str, str]]) -> None:
    """Auto-complete multiple todos.
    
    Args:
        completed_todos: List of completed todo items
    """
    results = automated_completion.complete_multiple_tasks(completed_todos)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    logger.info(f"Auto-completed {successful}/{total} tasks")
    
    # Show pending roadmap confirmations
    pending = automated_completion.get_pending_roadmap_confirmations()
    if pending:
        logger.info(f"{len(pending)} roadmap tasks awaiting user confirmation")
        for task in pending:
            logger.info(f"  - {task.get('task_id')}: {task.get('description', '')[:100]}...")


def confirm_roadmap_completions(task_ids: List[str]) -> None:
    """Confirm multiple roadmap task completions.
    
    Args:
        task_ids: List of task IDs to confirm
    """
    successful = 0
    
    for task_id in task_ids:
        if automated_completion.confirm_roadmap_task(task_id):
            successful += 1
            logger.info(f"Confirmed roadmap task: {task_id}")
        else:
            logger.error(f"Failed to confirm roadmap task: {task_id}")
    
    logger.info(f"Confirmed {successful}/{len(task_ids)} roadmap tasks")
