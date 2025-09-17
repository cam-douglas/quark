#!/usr/bin/env python3
"""Task Completion Module - Handle task completion and archiving workflow.

Manages the complete lifecycle of task completion including archiving and roadmap updates.

Integration: Supports QuarkDriver and AutonomousAgent task completion workflow.
Rationale: Centralized task completion logic with proper archiving and roadmap synchronization.
"""

from typing import Dict, List
import datetime
from pathlib import Path
import yaml

def mark_task_complete(task_id: str, tasks_registry: List[Dict]) -> bool:
    """Mark a task as complete and move it to archive."""
    from .task_storage import write_archive_file, write_in_progress_tasks_file
    from .roadmap_integration import update_roadmap_with_done_tag

    # Find the task
    task_to_complete = None
    for task in tasks_registry:
        if task.get("id") == task_id:
            task_to_complete = task
            break

    if not task_to_complete:
        print(f"❌ Task not found: {task_id}")
        return False

    # Mark as completed
    task_to_complete["status"] = "completed"
    task_to_complete["completed_date"] = datetime.datetime.now().isoformat()

    # Load existing archive
    archive_file = Path(__file__).resolve().parents[1] / "tasks" / "tasks_archive.yaml"
    archive_data = []

    if archive_file.exists():
        try:
            existing_archive = yaml.safe_load(archive_file.read_text()) or []
            if isinstance(existing_archive, list):
                archive_data = existing_archive
            elif isinstance(existing_archive, dict) and "archived_tasks" in existing_archive:
                archive_data = existing_archive["archived_tasks"]
        except Exception as e:
            print(f"⚠️  Warning: Could not read existing archive: {e}")

    archive_data.append(task_to_complete)

    # Write updated archive
    write_archive_file(archive_data)

    # Update roadmap file with DONE tag
    update_roadmap_with_done_tag(task_to_complete)

    # Get remaining pending roadmap tasks and regenerate in-progress file
    remaining_roadmap_tasks = [
        t for t in tasks_registry
        if t.get("source", "").endswith("_rules") and t.get("status") == "pending"
    ]
    write_in_progress_tasks_file(remaining_roadmap_tasks)

    print(f"✅ Task marked complete and archived: {task_to_complete.get('title', 'Unknown')}")
    return True

def get_completion_summary(tasks: List[Dict]) -> Dict:
    """Get summary of task completion status."""
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t.get("status") == "completed"])
    pending_tasks = len([t for t in tasks if t.get("status") == "pending"])
    in_progress_tasks = len([t for t in tasks if t.get("status") == "in_progress"])

    return {
        "total": total_tasks,
        "completed": completed_tasks,
        "pending": pending_tasks,
        "in_progress": in_progress_tasks,
        "completion_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    }
