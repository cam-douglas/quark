#!/usr/bin/env python3
"""Task Completion Integration with Quark State System.

Integrates the automated task completion system with existing Quark state
system components including task loader, goal manager, and command handlers.

Integration: Bridge between new completion system and existing state infrastructure
Rationale: Seamless integration without breaking existing workflows
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

# Import existing Quark state system components
from . import task_loader, goal_manager
from .task_management.task_completion import mark_task_complete as existing_mark_complete
from .command_handlers import handle_complete_task as existing_handle_complete

# Import new task completion system
import sys
brain_modules_path = Path(__file__).resolve().parents[2] / "brain" / "modules"
sys.path.append(str(brain_modules_path))

try:
    from task_completion_manager import task_completion_manager
    from automated_task_completion import automated_completion
except ImportError:
    # Fallback if modules not available
    task_completion_manager = None
    automated_completion = None
    logging.warning("Task completion modules not available - using existing system only")

logger = logging.getLogger(__name__)

class QuarkTaskCompletionBridge:
    """Bridge between new completion system and existing Quark state system.
    
    Provides seamless integration that enhances existing functionality
    without breaking current workflows.
    """
    
    def __init__(self):
        """Initialize task completion bridge."""
        self.enhanced_completion_available = (
            task_completion_manager is not None and 
            automated_completion is not None
        )
        
        if self.enhanced_completion_available:
            logger.info("Enhanced task completion system available")
        else:
            logger.info("Using existing task completion system only")
    
    def complete_task(self, task_id: str, task_description: Optional[str] = None,
                     is_roadmap_task: bool = None, completion_notes: Optional[str] = None) -> bool:
        """Complete a task using enhanced or existing system.
        
        Args:
            task_id: Task identifier
            task_description: Task description (auto-detected if not provided)
            is_roadmap_task: Whether roadmap task (auto-detected if not provided)
            completion_notes: Additional completion notes
            
        Returns:
            True if task completed successfully
        """
        try:
            # Try enhanced completion system first
            if self.enhanced_completion_available:
                return self._complete_task_enhanced(
                    task_id, task_description, is_roadmap_task, completion_notes)
            else:
                # Fallback to existing system
                return self._complete_task_existing(task_id)
        
        except Exception as e:
            logger.error(f"Task completion failed for {task_id}: {e}")
            # Always try existing system as fallback
            return self._complete_task_existing(task_id)
    
    def _complete_task_enhanced(self, task_id: str, task_description: Optional[str],
                               is_roadmap_task: Optional[bool], 
                               completion_notes: Optional[str]) -> bool:
        """Complete task using enhanced completion system."""
        # Auto-detect task details if not provided
        if task_description is None or is_roadmap_task is None:
            task_details = self._get_task_details(task_id)
            if task_details:
                task_description = task_description or task_details.get('description', task_id)
                is_roadmap_task = is_roadmap_task if is_roadmap_task is not None else task_details.get('is_roadmap', True)
        
        # Set defaults
        task_description = task_description or task_id
        is_roadmap_task = is_roadmap_task if is_roadmap_task is not None else True
        
        # Determine roadmap file if roadmap task
        roadmap_file = None
        if is_roadmap_task:
            roadmap_file = self._determine_roadmap_file(task_id, task_description)
        
        # Complete using enhanced system
        success = task_completion_manager.complete_task(
            task_id=task_id,
            task_description=task_description,
            is_roadmap_task=is_roadmap_task,
            roadmap_file=roadmap_file,
            completion_notes=completion_notes
        )
        
        # Also complete in existing system for compatibility
        if success:
            try:
                self._complete_task_existing(task_id)
            except Exception as e:
                logger.warning(f"Existing system completion failed for {task_id}: {e}")
        
        return success
    
    def _complete_task_existing(self, task_id: str) -> bool:
        """Complete task using existing Quark system."""
        try:
            # Try task_loader first
            if hasattr(task_loader, 'mark_task_complete'):
                return task_loader.mark_task_complete(task_id)
            
            # Try task_management module
            if existing_mark_complete:
                # Get task from registry
                tasks = task_loader.get_tasks() if hasattr(task_loader, 'get_tasks') else []
                for task in tasks:
                    if task.get('id') == task_id:
                        return existing_mark_complete(task_id, tasks)
                
                logger.warning(f"Task {task_id} not found in registry")
                return False
            
            # Try command handler
            if existing_handle_complete:
                return existing_handle_complete(task_id)
            
            logger.error("No existing completion method available")
            return False
            
        except Exception as e:
            logger.error(f"Existing system completion failed: {e}")
            return False
    
    def _get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task details from existing system."""
        try:
            # Try to get from task loader
            if hasattr(task_loader, 'get_tasks'):
                tasks = task_loader.get_tasks()
                for task in tasks:
                    if task.get('id') == task_id:
                        return {
                            'description': task.get('title', task.get('description', task_id)),
                            'is_roadmap': task.get('source', '').endswith('.md'),
                            'source': task.get('source', ''),
                            'priority': task.get('priority', 'medium')
                        }
            
            # Try goal manager
            if hasattr(goal_manager, 'next_goal'):
                goal = goal_manager.next_goal()
                if goal and goal.get('id') == task_id:
                    return {
                        'description': goal.get('title', goal.get('description', task_id)),
                        'is_roadmap': True,  # Goals are typically roadmap tasks
                        'source': goal.get('source', ''),
                        'priority': goal.get('priority', 'high')
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get task details for {task_id}: {e}")
            return None
    
    def _determine_roadmap_file(self, task_id: str, task_description: str) -> Optional[str]:
        """Determine which roadmap file a task belongs to."""
        task_lower = task_description.lower()
        id_lower = task_id.lower()
        
        # Foundation layer tasks
        foundation_keywords = [
            "foundation", "morphogen", "ventricular", "excavation", "csf",
            "shh", "bmp", "spatial", "1.1.", "1.2."
        ]
        if any(keyword in task_lower or keyword in id_lower for keyword in foundation_keywords):
            return "foundation_layer_detailed_tasks.md"
        
        # Stage 1 embryonic tasks
        stage1_keywords = [
            "stage1", "embryonic", "neural_tube", "2.1.", "2.2.", "2.3."
        ]
        if any(keyword in task_lower or keyword in id_lower for keyword in stage1_keywords):
            return "stage1_embryonic_detailed_tasks.md"
        
        # Stage 2 fetal tasks
        stage2_keywords = [
            "stage2", "fetal", "3.1.", "3.2.", "3.3."
        ]
        if any(keyword in task_lower or keyword in id_lower for keyword in stage2_keywords):
            return "stage2_fetal_detailed_tasks.md"
        
        # Default to foundation layer
        return "foundation_layer_detailed_tasks.md"
    
    def get_completion_status(self) -> Dict[str, Any]:
        """Get status of completion system."""
        status = {
            "enhanced_completion_available": self.enhanced_completion_available,
            "existing_system_available": True,
            "pending_roadmap_completions": 0
        }
        
        if self.enhanced_completion_available:
            try:
                pending = automated_completion.get_pending_roadmap_confirmations()
                status["pending_roadmap_completions"] = len(pending)
                status["pending_tasks"] = [
                    {
                        "task_id": task.get("task_id"),
                        "description": task.get("description", "")[:100] + "...",
                        "target_file": task.get("target_file")
                    }
                    for task in pending[:5]  # Show first 5
                ]
            except Exception as e:
                logger.warning(f"Could not get pending completions: {e}")
        
        return status
    
    def confirm_roadmap_completions(self, task_ids: List[str]) -> Dict[str, bool]:
        """Confirm multiple roadmap task completions.
        
        Args:
            task_ids: List of task IDs to confirm
            
        Returns:
            Dict mapping task_id to success status
        """
        results = {}
        
        if not self.enhanced_completion_available:
            logger.warning("Enhanced completion system not available for confirmations")
            return {task_id: False for task_id in task_ids}
        
        for task_id in task_ids:
            try:
                success = automated_completion.confirm_roadmap_task(task_id)
                results[task_id] = success
                
                if success:
                    logger.info(f"Confirmed roadmap completion: {task_id}")
                else:
                    logger.warning(f"Failed to confirm roadmap completion: {task_id}")
                    
            except Exception as e:
                logger.error(f"Error confirming {task_id}: {e}")
                results[task_id] = False
        
        return results


# Global bridge instance
completion_bridge = QuarkTaskCompletionBridge()


def complete_task(task_id: str, task_description: Optional[str] = None,
                 is_roadmap_task: Optional[bool] = None, 
                 completion_notes: Optional[str] = None) -> bool:
    """Complete a task using the integrated completion system.
    
    Args:
        task_id: Task identifier
        task_description: Task description (auto-detected if not provided)
        is_roadmap_task: Whether roadmap task (auto-detected if not provided)
        completion_notes: Additional completion notes
        
    Returns:
        True if task completed successfully
    """
    return completion_bridge.complete_task(
        task_id, task_description, is_roadmap_task, completion_notes)


def get_completion_status() -> Dict[str, Any]:
    """Get status of the completion system."""
    return completion_bridge.get_completion_status()


def confirm_roadmap_completions(task_ids: List[str]) -> Dict[str, bool]:
    """Confirm multiple roadmap task completions."""
    return completion_bridge.confirm_roadmap_completions(task_ids)


def show_pending_confirmations() -> None:
    """Show pending roadmap confirmations."""
    status = get_completion_status()
    
    pending_count = status.get("pending_roadmap_completions", 0)
    if pending_count == 0:
        print("✅ No pending roadmap confirmations")
        return
    
    print(f"📋 {pending_count} roadmap tasks awaiting confirmation:")
    print("=" * 50)
    
    for task in status.get("pending_tasks", []):
        print(f"🔹 {task['task_id']}")
        print(f"   {task['description']}")
        print(f"   → {task['target_file']}")
        print()
    
    if pending_count > 5:
        print(f"... and {pending_count - 5} more tasks")
        print()
    
    print("To confirm completions:")
    print("python quark_state_system_launcher.py confirm-completions <task_id1> <task_id2> ...")
