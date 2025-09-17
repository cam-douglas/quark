#!/usr/bin/env python3
"""Task Storage Module - YAML file operations and task persistence.

Handles all YAML file reading/writing operations for the task management system.

Integration: Core storage layer for QuarkDriver and AutonomousAgent task management.
Rationale: Centralized task persistence with performance optimization.
"""

from pathlib import Path
from typing import List, Dict
import yaml
import concurrent.futures as _cf
from tools_utilities.scripts.performance_utils import read_text_cached

# Task directory and file paths
_TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"
_PRIORITY_FILES = {
    "archive": _TASK_DIR / "tasks_archive.yaml",
    "chat": _TASK_DIR / "chat_tasks.yaml",
}



# Global task registry
_TASKS: List[Dict] = []

def _load_yaml(fp: Path, prio: str):
    """Helper to read YAML file and append priority metadata."""
    try:
        text = fp.read_text()
    except FileNotFoundError:
        return []
    raw_data = yaml.safe_load(read_text_cached(fp)) or []

    # Handle different YAML formats
    if isinstance(raw_data, dict) and "tasks" in raw_data:
        # Format: {generated: date, tasks: [list of strings or dicts]}
        task_list = raw_data["tasks"]
    elif isinstance(raw_data, list):
        # Format: [list of strings or dicts]
        task_list = raw_data
    else:
        return []

    # Convert strings to task dictionaries
    data = []
    for i, item in enumerate(task_list):
        if isinstance(item, str):
            # Convert string to task dict
            task_dict = {
                "id": f"{prio}_{i}_{hash(item) % 10000}",
                "title": item,
                "priority": prio,
                "status": "pending"
            }
        elif isinstance(item, dict):
            # Already a dict, just add missing fields
            task_dict = item.copy()
            task_dict.setdefault("priority", prio)
            task_dict.setdefault("status", "pending")
            task_dict.setdefault("id", f"{prio}_{i}_{hash(str(item)) % 10000}")
        else:
            continue
        data.append(task_dict)

    return data

def load_all_tasks():
    """Load all tasks from YAML files in parallel."""
    global _TASKS
    _TASKS.clear()

    with _cf.ThreadPoolExecutor(max_workers=min(4, len(_PRIORITY_FILES))) as _executor:
        futures = {
            _executor.submit(_load_yaml, fp, prio): prio for prio, fp in _PRIORITY_FILES.items()
        }
        for fut in futures:
            _TASKS.extend(fut.result())

def _write_tasks_to_file(prio: str):
    """Write tasks to specific file (only for chat and archive)."""
    if prio not in _PRIORITY_FILES:
        return  # Only write files we actually use

    fp = _PRIORITY_FILES[prio]
    data = [t for t in _TASKS if t.get("priority") == prio]
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(yaml.safe_dump(data, sort_keys=False))

def _flush_all():
    """Write only chat and archive files - NO priority files."""
    # Only write the files we actually want to keep
    for prio in ["chat", "archive"]:
        if prio in _PRIORITY_FILES:
            _write_tasks_to_file(prio)

def write_in_progress_tasks_file(roadmap_tasks: List[Dict]):
    """Write all roadmap tasks to in-progress_tasks.yaml file with comprehensive organization."""
    import datetime

    if not roadmap_tasks:
        print("⚠️  No in-progress roadmap tasks found")
        return

    # Create in-progress tasks file
    tasks_file = _TASK_DIR / "in-progress_tasks.yaml"

    # Group tasks by roadmap file first, then by section
    tasks_by_roadmap = {}
    for task in roadmap_tasks:
        roadmap_name = task.get("roadmap_file", "unknown.md")
        section_subtitle = task.get("section_subtitle", "Unknown Section")
        original_header = task.get("original_header", section_subtitle)

        if roadmap_name not in tasks_by_roadmap:
            tasks_by_roadmap[roadmap_name] = {}

        if section_subtitle not in tasks_by_roadmap[roadmap_name]:
            tasks_by_roadmap[roadmap_name][section_subtitle] = {
                "original_header": original_header,
                "tasks": []
            }

        tasks_by_roadmap[roadmap_name][section_subtitle]["tasks"].append(task)

    # Create the consolidated data structure with comprehensive organization
    consolidated_data = {
        "generated": datetime.datetime.now().isoformat(),
        "description": "All in-progress tasks from active roadmap files, organized by sub-headings",
        "total_tasks": len(roadmap_tasks),
        "active_roadmaps": list(tasks_by_roadmap.keys()),
        "tasks_by_roadmap": tasks_by_roadmap,
        "all_tasks": roadmap_tasks
    }

    # Write to file
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(yaml.safe_dump(consolidated_data, sort_keys=False, default_flow_style=False))

    print(f"✅ Written {len(roadmap_tasks)} in-progress tasks to {tasks_file}")
    print("   Organized by roadmap files and sub-headings:")
    for roadmap_name, sections in tasks_by_roadmap.items():
        print(f"   📄 {roadmap_name}:")
        for section_name, section_data in sections.items():
            task_count = len(section_data["tasks"])
            print(f"      - {section_data['original_header']}: {task_count} tasks")

def write_archive_file(archived_tasks: List[Dict]):
    """Write completed tasks to tasks_archive.yaml."""
    import datetime

    archive_file = _TASK_DIR / "tasks_archive.yaml"

    archive_structure = {
        "last_updated": datetime.datetime.now().isoformat(),
        "total_archived_tasks": len(archived_tasks),
        "archived_tasks": archived_tasks
    }

    archive_file.parent.mkdir(parents=True, exist_ok=True)
    archive_file.write_text(yaml.safe_dump(archive_structure, sort_keys=False, default_flow_style=False))

# Initialize tasks on module import (but don't load from old priority files for roadmap tasks)
# Only load chat tasks and other non-roadmap tasks
def load_non_roadmap_tasks_only():
    """Load only non-roadmap tasks (chat tasks, etc.) to avoid old priority file contamination."""
    global _TASKS
    _TASKS.clear()

    # Only load chat tasks and other non-roadmap task files
    non_roadmap_files = {
        "chat": _PRIORITY_FILES["chat"],
        "archive": _PRIORITY_FILES["archive"]
    }

    with _cf.ThreadPoolExecutor(max_workers=2) as _executor:
        futures = {
            _executor.submit(_load_yaml, fp, prio): prio for prio, fp in non_roadmap_files.items()
        }
        for fut in futures:
            _TASKS.extend(fut.result())

load_non_roadmap_tasks_only()
