# state/quark_state_system/task_loader.py
"""Central loader for consolidated task YAML files.

Reads the machine-readable task lists in ``state/tasks/`` and exposes helper
functions for the state-system and autonomous agent.
"""
from pathlib import Path
import yaml
from typing import List, Optional, Dict

_TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"
_PRIORITY_FILES = {
    "high": _TASK_DIR / "tasks_high.yaml",
    "medium": _TASK_DIR / "tasks_medium.yaml",
    "low": _TASK_DIR / "tasks_low.yaml",
    "archive": _TASK_DIR / "tasks_archive.yaml",
}

#: Load all task entries (cached at import time)
_TASKS: List[Dict] = []
for prio, fp in _PRIORITY_FILES.items():
    if fp.exists():
        data = yaml.safe_load(fp.read_text()) or []
        for d in data:
            d.setdefault("priority", prio)
        _TASKS.extend(data)

def get_tasks(priority: Optional[str] = None, status: Optional[str] = None):
    """Yield tasks filtered by priority and/or status."""
    for task in _TASKS:
        if priority and task.get("priority") != priority:
            continue
        if status and task.get("status") != status:
            continue
        yield task

def next_actions(limit: int = 3):
    """Return next *limit* highest-priority pending tasks (high→low)."""
    ordered = sorted(
        (t for t in _TASKS if t.get("status") == "pending"),
        key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "medium"), 3),
    )
    return ordered[:limit]
