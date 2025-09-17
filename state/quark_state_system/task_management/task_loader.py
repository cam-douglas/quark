#!/usr/bin/env python3
"""Central Task Loader - Simplified main interface following sprint-batch-task-management rules.

Coordinates task loading, roadmap integration, and sprint management modules.

Integration: Main interface for QuarkDriver and AutonomousAgent task operations.
Rationale: Simplified coordinator that delegates to specialized modules.
"""

from pathlib import Path
from typing import List, Optional, Dict

# Import specialized modules
from .task_storage import _TASKS, load_non_roadmap_tasks_only
from .roadmap_integration import extract_tasks_from_active_roadmaps
from .sprint_management import (
    add_sprint_structure_to_task, format_task_for_display, get_current_phase_summary
)
from .task_completion import mark_task_complete as _mark_task_complete

# Agile phase/step labelling helper
from ..agile_utils import format_phase_step

# Re-export task directory for compatibility
_TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"

def get_tasks(status: Optional[str] = None, priority: Optional[str] = None) -> List[Dict]:
    """Get tasks filtered by status and/or priority."""
    filtered = _TASKS[:]

    if status:
        filtered = [t for t in filtered if t.get("status") == status]
    if priority:
        filtered = [t for t in filtered if t.get("priority") == priority]

    return filtered

def next_actions(limit: int = 5) -> List[Dict]:
    """Get the next priority actions with sprint structure."""
    # Get pending tasks and apply sprint structure
    pending_tasks = get_tasks(status="pending")

    # Add sprint structure to each task
    structured_tasks = []
    for i, task in enumerate(pending_tasks):
        structured_task = add_sprint_structure_to_task(task.copy(), i)
        structured_tasks.append(structured_task)

    # Sort by phase, then batch, then step
    structured_tasks.sort(key=lambda t: (t.get("phase", 1), t.get("batch", "A"), t.get("step", 1)))

    return structured_tasks[:limit]

def task_exists(title: str) -> bool:
    """Check if a task with the given title already exists."""
    return any(title in t.get("title", "") for t in _TASKS)

def add_task(task: Dict) -> bool:
    """Add task to registry with sprint structure if not duplicate."""
    if task_exists(task.get("title", "")):
        return False

    # Ensure task title carries a Phase/Step label for Agile visibility
    title = task.get("title", "")
    if "phase" not in title.lower() or "step" not in title.lower():
        placeholder = format_phase_step(0, 0, 0, 0)
        task["title"] = f"{placeholder} — {title}"

    _TASKS.append(task)

    # Do NOT write any files here - all file writing is handled separately
    # Roadmap tasks → in-progress_tasks.yaml (written by generate_tasks_from_active_roadmaps)
    # Chat tasks → written only when explicitly needed

    return True

def generate_tasks_from_active_roadmaps() -> int:
    """Generate tasks from active roadmaps with sprint-batch-task-management structure."""
    from .task_storage import write_in_progress_tasks_file

    # Clear existing roadmap tasks
    global _TASKS
    _TASKS = [t for t in _TASKS if not t.get("source", "").endswith("_rules")]

    # Extract tasks from roadmaps
    roadmap_tasks = extract_tasks_from_active_roadmaps()

    # Add sprint structure to each task
    structured_tasks = []
    for i, task in enumerate(roadmap_tasks):
        structured_task = add_sprint_structure_to_task(task, i)
        structured_tasks.append(structured_task)
        _TASKS.append(structured_task)

    # Write to in-progress tasks file
    if structured_tasks:
        write_in_progress_tasks_file(structured_tasks)

    return len(structured_tasks)

def mark_task_complete(task_id: str) -> bool:
    """Mark a task as complete and handle archiving."""
    return _mark_task_complete(task_id, _TASKS)

def get_sprint_summary() -> str:
    """Get current sprint/phase summary following cursor rules."""
    pending_tasks = get_tasks(status="pending")
    return get_current_phase_summary(pending_tasks)

def format_tasks_for_display(tasks: List[Dict]) -> List[str]:
    """Format tasks for display following sprint-batch-task-management rules."""
    return [format_task_for_display(task) for task in tasks]

def sync_with_roadmaps(status_map: Dict):
    """Sync tasks with roadmap status (compatibility function)."""
    # This function maintains compatibility with existing code
    # The actual syncing is now handled by generate_tasks_from_active_roadmaps
    pass

def reset_all():
    """Clear in-memory task list. Only roadmap tasks will be regenerated."""
    global _TASKS
    _TASKS.clear()

    # Note: We no longer generate or clear old priority files
    # Only in-progress_tasks.yaml is used for roadmap tasks

# Initialize tasks on module import (only non-roadmap tasks)
load_non_roadmap_tasks_only()
