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
    "chat": _TASK_DIR / "chat_tasks.yaml",
}

#: Load all task entries (cached at import time)
_TASKS: List[Dict] = []
for prio, fp in _PRIORITY_FILES.items():
    if fp.exists():
        data = yaml.safe_load(fp.read_text()) or []
        for d in data:
            d.setdefault("priority", prio)
        _TASKS.extend(data)

# -- internal helper to persist tasks list to YAML ---------------------------

def _write_tasks_to_file(prio: str):
    fp = _PRIORITY_FILES[prio]
    data = [t for t in _TASKS if t.get("priority") == prio]
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(yaml.safe_dump(data, sort_keys=False))

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

# ---------------------------------------------------------------------------
# Phase-2: roadmap → task sync helpers
# ---------------------------------------------------------------------------

def _task_exists(title: str) -> bool:
    return any(t.get("title") == title for t in _TASKS)


def sync_with_roadmaps(snapshot: Dict[str, str]):
    """Ensure each roadmap item in *snapshot* has a corresponding task.

    Mapping e.g. {"Pillar 3 – Hierarchical Processing": "planned"}
    Generates high/medium/low priority tasks depending on status.
    """
    priority_map = {"progress": "high", "planned": "medium", "done": None}

    for title, status in snapshot.items():
        prio = priority_map.get(status, "low")
        if prio is None:
            continue  # done
        if _task_exists(title):
            continue

        task = {"id": f"roadmap-{len(_TASKS)+1}", "title": title, "status": "pending", "priority": prio}
        _TASKS.append(task)

        # append to YAML file
        fp = _PRIORITY_FILES[prio]
        data = []
        if fp.exists():
            data = yaml.safe_load(fp.read_text()) or []
        data.append(task)
        fp.write_text(yaml.safe_dump(data, sort_keys=False))
