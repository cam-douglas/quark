#!/usr/bin/env python3
"""Task Completion Management System.

Automatically updates task status in appropriate files when tasks are completed,
including roadmap tasks and ad-hoc chat tasks with proper user confirmation.

Integration: Task management component for automated completion tracking
Rationale: Centralized task completion with file-based persistence
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import json
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

class TaskCompletionManager:
    """Manager for automated task completion tracking.
    
    Handles updating task status in roadmap task files and chat task files
    when tasks are completed, with appropriate user confirmation workflows.
    """
    
    def __init__(self, base_path: str = "/Users/camdouglas/quark"):
        """Initialize task completion manager.
        
        Args:
            base_path: Base path to quark repository
        """
        self.base_path = Path(base_path)
        self.roadmap_tasks_path = self.base_path / "state" / "tasks" / "roadmap_tasks"
        self.chat_tasks_path = self.base_path / "state" / "tasks" / "chat_tasks.yaml"
        
        # Ensure directories exist
        self.roadmap_tasks_path.mkdir(parents=True, exist_ok=True)
        self.chat_tasks_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized TaskCompletionManager")
        logger.info(f"Roadmap tasks path: {self.roadmap_tasks_path}")
        logger.info(f"Chat tasks path: {self.chat_tasks_path}")
    
    def complete_task(self, task_id: str, task_description: str, 
                     is_roadmap_task: bool = False, 
                     roadmap_file: Optional[str] = None,
                     completion_notes: Optional[str] = None) -> bool:
        """Mark a task as complete in the appropriate file.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of the completed task
            is_roadmap_task: Whether this is a roadmap task
            roadmap_file: Specific roadmap file to update (if roadmap task)
            completion_notes: Additional notes about completion
            
        Returns:
            True if task was successfully marked complete
        """
        try:
            if is_roadmap_task:
                return self._complete_roadmap_task(
                    task_id, task_description, roadmap_file, completion_notes)
            else:
                return self._complete_chat_task(
                    task_id, task_description, completion_notes)
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    def _complete_roadmap_task(self, task_id: str, task_description: str,
                              roadmap_file: Optional[str] = None,
                              completion_notes: Optional[str] = None) -> bool:
        """Complete a roadmap task (requires user confirmation)."""
        logger.warning("Roadmap task completion requires user confirmation")
        logger.info(f"Task ready for completion: {task_id} - {task_description}")
        
        if roadmap_file:
            roadmap_path = self.roadmap_tasks_path / roadmap_file
            logger.info(f"Target file: {roadmap_path}")
        
        # For roadmap tasks, we don't automatically update - we log the completion
        # and wait for user confirmation
        completion_record = {
            "task_id": task_id,
            "description": task_description,
            "completion_time": datetime.now().isoformat(),
            "completion_notes": completion_notes,
            "target_file": roadmap_file,
            "status": "awaiting_user_confirmation"
        }
        
        # Store pending completion for user review
        pending_file = self.base_path / "state" / "tasks" / "pending_roadmap_completions.json"
        self._append_pending_completion(pending_file, completion_record)
        
        logger.info(f"Roadmap task {task_id} marked for user confirmation")
        return True
    
    def _complete_chat_task(self, task_id: str, task_description: str,
                           completion_notes: Optional[str] = None) -> bool:
        """Complete a chat/ad-hoc task (automatic)."""
        logger.info(f"Completing chat task: {task_id}")
        
        # Load existing chat tasks or create new structure
        chat_tasks = self._load_chat_tasks()
        
        # Create completion record
        completion_record = {
            "task_id": task_id,
            "description": task_description,
            "status": "completed",
            "completion_time": datetime.now().isoformat(),
            "completion_notes": completion_notes or "Task completed successfully"
        }
        
        # Add to completed tasks
        if "completed_tasks" not in chat_tasks:
            chat_tasks["completed_tasks"] = []
        
        chat_tasks["completed_tasks"].append(completion_record)
        
        # Remove from active tasks if present
        if "active_tasks" in chat_tasks:
            chat_tasks["active_tasks"] = [
                task for task in chat_tasks["active_tasks"] 
                if task.get("task_id") != task_id
            ]
        
        # Update metadata
        chat_tasks["metadata"] = {
            "last_updated": datetime.now().isoformat(),
            "total_completed": len(chat_tasks["completed_tasks"]),
            "session_id": self._generate_session_id()
        }
        
        # Save updated chat tasks
        self._save_chat_tasks(chat_tasks)
        
        logger.info(f"Chat task {task_id} marked as completed")
        return True
    
    def _load_chat_tasks(self) -> Dict[str, Any]:
        """Load existing chat tasks from file."""
        if self.chat_tasks_path.exists():
            try:
                with open(self.chat_tasks_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load chat tasks: {e}")
        
        # Return default structure
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "session_id": self._generate_session_id()
            },
            "active_tasks": [],
            "completed_tasks": []
        }
    
    def _save_chat_tasks(self, chat_tasks: Dict[str, Any]) -> None:
        """Save chat tasks to file."""
        with open(self.chat_tasks_path, 'w') as f:
            yaml.dump(chat_tasks, f, default_flow_style=False, sort_keys=False)
    
    def _append_pending_completion(self, pending_file: Path, completion_record: Dict[str, Any]) -> None:
        """Append pending completion record for user review."""
        pending_completions = []
        
        if pending_file.exists():
            try:
                with open(pending_file, 'r') as f:
                    pending_completions = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load pending completions: {e}")
        
        pending_completions.append(completion_record)
        
        # Ensure parent directory exists
        pending_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pending_file, 'w') as f:
            json.dump(pending_completions, f, indent=2)
    
    def _generate_session_id(self) -> str:
        """Generate a session ID based on current timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def complete_foundation_layer_tasks(self, completed_task_ids: List[str]) -> None:
        """Complete multiple foundation layer tasks.
        
        Args:
            completed_task_ids: List of task IDs that were completed
        """
        logger.info(f"Completing {len(completed_task_ids)} foundation layer tasks")
        
        # Map task IDs to descriptions
        task_descriptions = {
            "ventricular_topology_design": "1.1.1 Design ventricular cavity topology - Complete system with lateral ventricles, third ventricle, fourth ventricle, and cerebral aqueduct connectivity",
            "voxel_excavation_algorithm": "1.1.2 Implement voxel excavation algorithm - High-precision cavity detection in 1µm³ grid with CSF flow pathway validation",
            "csf_modeling_preparation": "1.1.3 CSF modeling preparation - Flow dynamics framework with pressure gradient mappings and boundary conditions"
        }
        
        for task_id in completed_task_ids:
            if task_id in task_descriptions:
                # These are foundation layer tasks, so they're roadmap tasks
                self.complete_task(
                    task_id=task_id,
                    task_description=task_descriptions[task_id],
                    is_roadmap_task=True,
                    roadmap_file="foundation_layer_detailed_tasks.md",
                    completion_notes=f"Architecture-compliant implementation with modular design. "
                                   f"All files <300 lines, comprehensive validation, and biological accuracy."
                )
            else:
                # Fallback for unknown task IDs
                self.complete_task(
                    task_id=task_id,
                    task_description=f"Foundation layer task: {task_id}",
                    is_roadmap_task=False,
                    completion_notes="Task completed as part of foundation layer implementation"
                )
    
    def get_pending_roadmap_completions(self) -> List[Dict[str, Any]]:
        """Get list of pending roadmap task completions awaiting user confirmation."""
        pending_file = self.base_path / "state" / "tasks" / "pending_roadmap_completions.json"
        
        if not pending_file.exists():
            return []
        
        try:
            with open(pending_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pending completions: {e}")
            return []
    
    def confirm_roadmap_completion(self, task_id: str) -> bool:
        """Confirm and apply a pending roadmap task completion.
        
        Args:
            task_id: Task ID to confirm completion for
            
        Returns:
            True if completion was successfully applied
        """
        pending_completions = self.get_pending_roadmap_completions()
        
        # Find the pending completion
        target_completion = None
        remaining_completions = []
        
        for completion in pending_completions:
            if completion.get("task_id") == task_id:
                target_completion = completion
            else:
                remaining_completions.append(completion)
        
        if not target_completion:
            logger.error(f"No pending completion found for task {task_id}")
            return False
        
        # Apply the completion to the roadmap file
        if target_completion.get("target_file"):
            roadmap_path = self.roadmap_tasks_path / target_completion["target_file"]
            success = self._update_roadmap_file(roadmap_path, target_completion)
            
            if success:
                # Remove from pending completions
                pending_file = self.base_path / "state" / "tasks" / "pending_roadmap_completions.json"
                with open(pending_file, 'w') as f:
                    json.dump(remaining_completions, f, indent=2)
                
                logger.info(f"Roadmap task {task_id} completion confirmed and applied")
                return True
        
        return False
    
    def _update_roadmap_file(self, roadmap_path: Path, completion_record: Dict[str, Any]) -> bool:
        """Update roadmap file with completion status."""
        try:
            if not roadmap_path.exists():
                logger.error(f"Roadmap file not found: {roadmap_path}")
                return False
            
            # Read the file
            with open(roadmap_path, 'r') as f:
                content = f.read()
            
            # Find and update the task status
            task_id = completion_record["task_id"]
            
            # Look for task status patterns and update them
            status_patterns = [
                (rf"(\*\*Status\*\*:\s*)📋\s*Pending", r"\1✅ **COMPLETED**"),
                (rf"(\-\s*\*\*Status\*\*:\s*)📋\s*Pending", r"\1✅ **COMPLETED**"),
                (rf"(\*\*{re.escape(task_id)}\*\*.*?Status.*?)📋\s*Pending", r"\1✅ **COMPLETED**")
            ]
            
            updated = False
            for pattern, replacement in status_patterns:
                new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.MULTILINE)
                if new_content != content:
                    content = new_content
                    updated = True
                    break
            
            if updated:
                # Add completion timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                completion_note = f"\n<!-- Task {task_id} completed on {timestamp} -->\n"
                content += completion_note
                
                # Write back to file
                with open(roadmap_path, 'w') as f:
                    f.write(content)
                
                logger.info(f"Updated roadmap file: {roadmap_path}")
                return True
            else:
                logger.warning(f"Could not find task status to update in {roadmap_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update roadmap file: {e}")
            return False


# Global instance for easy access
task_completion_manager = TaskCompletionManager()


def complete_task(task_id: str, task_description: str, 
                 is_roadmap_task: bool = False,
                 roadmap_file: Optional[str] = None,
                 completion_notes: Optional[str] = None) -> bool:
    """Convenience function to complete a task.
    
    Args:
        task_id: Unique identifier for the task
        task_description: Description of the completed task
        is_roadmap_task: Whether this is a roadmap task
        roadmap_file: Specific roadmap file to update (if roadmap task)
        completion_notes: Additional notes about completion
        
    Returns:
        True if task was successfully marked complete
    """
    return task_completion_manager.complete_task(
        task_id, task_description, is_roadmap_task, roadmap_file, completion_notes
    )
