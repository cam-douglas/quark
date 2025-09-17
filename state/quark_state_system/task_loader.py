#!/usr/bin/env python3
"""Central Task Loader - Simplified main interface following sprint-batch-task-management rules.

Coordinates task loading, roadmap integration, and sprint management.
Separates roadmap recommendations from detailed task documentation.

Integration: Main interface for QuarkDriver and AutonomousAgent task operations.
Rationale: Simplified coordinator that delegates to specialized modules with clear separation.
"""

from pathlib import Path
from typing import List, Optional, Dict
import yaml

# Global task registry
_TASKS: List[Dict] = []
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
    # Get in-progress tasks first, then pending if none in-progress
    in_progress_tasks = get_tasks(status="in-progress")

    if in_progress_tasks:
        # Sort by phase, then batch, then step if available
        in_progress_tasks.sort(key=lambda t: (
            t.get("phase", 1),
            t.get("batch", "A"),
            t.get("step", 1)
        ))
        return in_progress_tasks[:limit]
    else:
        # No in-progress tasks, return empty list
        return []

def task_exists(title: str) -> bool:
    """Check if a task with the given title already exists."""
    return any(title in t.get("title", "") for t in _TASKS)

def add_task(task: Dict) -> bool:
    """Add task to registry if not duplicate."""
    if task_exists(task.get("title", "")):
        return False

    _TASKS.append(task)
    return True

def save_tasks_to_yaml():
    """Save all tasks to chat_tasks.yaml, auto-regenerating on every request."""
    from pathlib import Path

    # Save all tasks to chat_tasks.yaml
    yaml_dir = Path(__file__).resolve().parents[1] / "tasks"
    yaml_dir.mkdir(exist_ok=True)

    yaml_file = yaml_dir / "chat_tasks.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(_TASKS, f, default_flow_style=False, sort_keys=False)

def generate_tasks_from_active_roadmaps() -> int:
    """Generate tasks from active roadmaps."""
    import re
    from pathlib import Path

    # Import roadmap controller
    try:
        from management.rules.roadmap.roadmap_controller import get_all_roadmaps, status_snapshot
    except ImportError:
        return 0

    roadmaps = get_all_roadmaps()
    tasks_added = 0

    for roadmap in roadmaps:
        if roadmap.get("format") != "markdown":
            continue

        roadmap_path = Path(roadmap["path"])
        if not roadmap_path.exists():
            continue

        # Skip archive and backup files (check full path)
        path_str = str(roadmap_path).lower()
        if any(skip_word in path_str for skip_word in ["archive", "backup", "superseded", "deprecated"]):
            continue

        # Extract stage/priority from filename
        filename = roadmap_path.stem.lower()
        if "stage1" in filename or "embryonic" in filename:
            priority = "high"
            stage = "Stage1 Embryonic"
        elif "stage2" in filename or "fetal" in filename:
            priority = "high"
            stage = "Stage2 Fetal"
        elif "stage3" in filename:
            priority = "medium"
            stage = "Stage3 Early Post-Natal"
        elif "stage4" in filename:
            priority = "medium"
            stage = "Stage4 Childhood"
        elif "stage5" in filename:
            priority = "low"
            stage = "Stage5 Adolescence"
        elif "stage6" in filename:
            priority = "low"
            stage = "Stage6 Adult"
        else:
            priority = "medium"
            stage = roadmap.get("title", "General")

        try:
            content = roadmap_path.read_text(encoding="utf-8", errors="ignore")

            # Extract tasks from bullet points under ALL task sections
            in_task_section = False
            current_section = None

            for line in content.split('\n'):
                line = line.strip()

                # Check for any of the three task sections
                if any(section in line for section in ["Engineering Milestones", "Biological Goals", "SOTA ML Practices"]):
                    in_task_section = True
                    current_section = line
                    continue
                elif line.startswith('#') or (line.startswith('**') and in_task_section):
                    # End of current task section
                    if not any(section in line for section in ["Engineering Milestones", "Biological Goals", "SOTA ML Practices"]):
                        in_task_section = False
                        current_section = None
                        continue

                if in_task_section and line.startswith('* '):
                    # Extract task from bullet point
                    task_text = line[2:].strip()

                    # Extract category in brackets if present
                    category_match = re.match(r'\[([^\]]+)\]\s*(.*)', task_text)
                    if category_match:
                        category = category_match.group(1)
                        task_title = category_match.group(2).strip()
                    else:
                        category = "general"
                        task_title = task_text

                    # Determine task status based on document status and line markers
                    task_status = "pending"  # default

                    # Skip tasks that are already marked as done
                    if any(done_marker in task_text for done_marker in ["✅", "✓", "[x]", "[X]", "DONE", "COMPLETED"]):
                        continue  # Skip completed tasks - they shouldn't be in active task list

                    # Check if document has "In Progress" status
                    if "📋 In Progress" in content:
                        # Tasks in "In Progress" documents are in-progress unless marked done
                        task_status = "in-progress"

                    # Create task dict
                    task = {
                        "title": f"{stage} ▶ {task_title}",
                        "description": task_text,
                        "status": task_status,
                        "priority": priority,
                        "category": category,
                        "stage": stage,
                        "source": str(roadmap_path)
                    }

                    # Add task if not duplicate
                    if add_task(task):
                        tasks_added += 1

        except Exception:
            # Skip problematic files
            continue

    # Always save tasks to YAML (auto-regenerate on every request)
    save_tasks_to_yaml()

    return tasks_added

def mark_task_complete(task_title: str) -> bool:
    """Mark a task as complete, move to archive, and update roadmap file."""
    from pathlib import Path

    # Find the task
    task_to_complete = None
    for task in _TASKS:
        if task.get('title') == task_title:
            task_to_complete = task
            break

    if not task_to_complete:
        return False

    # Mark task as completed
    task_to_complete['status'] = 'completed'
    task_to_complete['completed_date'] = str(Path(__file__).stat().st_mtime)  # Simple timestamp

    # Move to archive YAML
    archive_yaml_path = _TASK_DIR / "completed_tasks_archive.yaml"

    # Load existing archive or create new
    archived_tasks = []
    if archive_yaml_path.exists():
        try:
            with open(archive_yaml_path, 'r') as f:
                archived_tasks = yaml.safe_load(f) or []
        except Exception:
            archived_tasks = []

    # Add completed task to archive
    archived_tasks.append(task_to_complete)

    # Save updated archive
    with open(archive_yaml_path, 'w') as f:
        yaml.dump(archived_tasks, f, default_flow_style=False, sort_keys=False)

    # Remove from active tasks
    _TASKS.remove(task_to_complete)

    # Update roadmap file to mark task as done
    _mark_task_done_in_roadmap(task_to_complete)

    # Check if all tasks in this roadmap are done
    _check_and_update_roadmap_status(task_to_complete['source'])

    # Save updated active tasks
    save_tasks_to_yaml()

    return True

def _mark_task_done_in_roadmap(task: dict):
    """Mark a task as done in the roadmap markdown file."""
    from pathlib import Path

    roadmap_path = Path(task['source'])
    if not roadmap_path.exists():
        return

    try:
        content = roadmap_path.read_text(encoding="utf-8", errors="ignore")

        # Find the task line and mark it as done
        task_description = task['description']

        # Look for the exact task line and prepend with ✅
        lines = content.split('\n')
        updated_lines = []

        for line in lines:
            if task_description in line and line.strip().startswith('* '):
                # Mark as done if not already marked
                if not any(done_marker in line for done_marker in ["✅", "✓", "[x]", "[X]"]):
                    line = line.replace('* ', '* ✅ ', 1)
            updated_lines.append(line)

        # Write back to file
        roadmap_path.write_text('\n'.join(updated_lines), encoding="utf-8")

    except Exception as e:
        print(f"Warning: Could not update roadmap file {roadmap_path}: {e}")

def _check_and_update_roadmap_status(roadmap_path: str):
    """Check if all tasks in roadmap are done and update status if needed."""
    from pathlib import Path

    roadmap_file = Path(roadmap_path)
    if not roadmap_file.exists():
        return

    try:
        content = roadmap_file.read_text(encoding="utf-8", errors="ignore")

        # Count total tasks and done tasks in all sections
        in_task_section = False
        total_tasks = 0
        done_tasks = 0

        for line in content.split('\n'):
            line = line.strip()

            # Check for task sections
            if any(section in line for section in ["Engineering Milestones", "Biological Goals", "SOTA ML Practices"]):
                in_task_section = True
                continue
            elif line.startswith('#') or (line.startswith('**') and in_task_section):
                if not any(section in line for section in ["Engineering Milestones", "Biological Goals", "SOTA ML Practices"]):
                    in_task_section = False
                    continue

            # Count tasks
            if in_task_section and line.startswith('* '):
                total_tasks += 1
                if any(done_marker in line for done_marker in ["✅", "✓", "[x]", "[X]"]):
                    done_tasks += 1

        # Update roadmap status if all tasks are done
        if total_tasks > 0 and done_tasks == total_tasks:
            # Replace "In Progress" with "Complete"
            updated_content = content.replace("📋 In Progress", "✅ Complete")
            roadmap_file.write_text(updated_content, encoding="utf-8")
            print(f"✅ Roadmap {roadmap_file.name} marked as Complete - all {total_tasks} tasks done!")

    except Exception as e:
        print(f"Warning: Could not check roadmap status for {roadmap_file}: {e}")

def sync_with_roadmaps(status_map: Dict):
    """Sync tasks with roadmap status (compatibility function)."""
    pass

def reset_all():
    """Clear in-memory task list."""
    global _TASKS
    _TASKS.clear()

# Minimal functions for compatibility
def get_sprint_summary() -> str:
    return "Sprint summary not available"

def format_tasks_for_display(tasks: List[Dict]) -> List[str]:
    return [t.get("title", "No title") for t in tasks]

def complete_task_by_number(task_number: int) -> bool:
    """Complete a task by its display number (1-based)."""
    in_progress_tasks = get_tasks(status="in-progress")

    if task_number < 1 or task_number > len(in_progress_tasks):
        return False

    task_to_complete = in_progress_tasks[task_number - 1]
    return mark_task_complete(task_to_complete['title'])

def list_in_progress_tasks() -> List[str]:
    """List current in-progress tasks with numbers for easy completion."""
    in_progress_tasks = get_tasks(status="in-progress")

    task_list = []
    for i, task in enumerate(in_progress_tasks, 1):
        task_list.append(f"{i}. {task['title']}")

    return task_list
