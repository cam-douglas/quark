"""GoalManager – single source for roadmap-driven goals.

API:
    next_goal()   -> Dict | None  (highest-priority pending task)
    complete(id)  -> None          (mark task as completed)

On every call, it refreshes tasks by syncing with roadmap status so the
view is always up-to-date.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
from __future__ import annotations
from typing import Dict, Optional

from . import task_loader
from state.quark_state_system.advanced_planner import plan as llm_plan
from brain.tools.task_bridge import TASK_BRIDGE
from management.rules.roadmap.roadmap_controller import status_snapshot


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _refresh():
    """Sync tasks with roadmap status each time we query goals."""
    task_loader.sync_with_roadmaps(status_snapshot())


def next_goal() -> Optional[Dict]:
    """Return the highest-priority pending task or None."""
    _refresh()
    tasks = task_loader.next_actions(limit=1)
    if not tasks:
        return None
    task = tasks[0]
    # Auto-break large task into sub-goals using planner
    if len(task["title"].split()) > 12:
        try:
            sub = llm_plan(task["title"])
            if sub:
                # replace with first subtask and push remainder back into pool
                task_loader.push_front([s["title"] for s in sub[1:]])  # type: ignore
                task["title"] = sub[0]["title"]
        except Exception:  # pragma: no cover – fallback silently
            pass
    return task


def complete(task_id: str) -> None:
    """Mark a task as completed in memory and YAML files."""
    for task in task_loader._TASKS:  # pylint: disable=protected-access
        if task.get("id") == task_id:
            task["status"] = "completed"
            TASK_BRIDGE.mark_done(task["title"])  # propagate DONE to roadmap/chat files
            break
    # Persist change by rewriting YAML it belongs to
    prio = task.get("priority", "medium")
    fp = task_loader._PRIORITY_FILES[prio]  # type: ignore[attr-defined]
    data = [t for t in task_loader._TASKS if t.get("priority") == prio]
    fp.write_text(task_loader.yaml.safe_dump(data, sort_keys=False))

# ------------------------ Runtime helpers ----------------------------------

_CURRENT: Optional[Dict] = None  # cached goal


def poll_goal() -> Optional[Dict]:
    """Non-blocking fetch used by simulators each timestep."""
    global _CURRENT  # noqa: PLW0603
    if _CURRENT and _CURRENT.get("status") == "completed":
        _CURRENT = None
    if not _CURRENT:
        _CURRENT = next_goal()
    return _CURRENT


def urgency_scalar(goal: Dict) -> float:
    """Return a simple urgency value based on priority."""
    return {"high": 1.0, "medium": 0.6, "low": 0.2, "chat": 0.8}.get(goal.get("priority", "medium"), 0.5)


# ------------------------ Status logging ------------------------------------

import json
import time
import threading
import pathlib


def _status_writer():
    log_dir = pathlib.Path("state/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fp = log_dir / "goal_status.json"
    while True:
        snap = {
            "current": _CURRENT,
            "stats": {
                "pending": len(list(task_loader.get_tasks(status="pending"))),
                "completed": len(list(task_loader.get_tasks(status="completed"))),
            },
            "ts": time.time(),
        }
        fp.write_text(json.dumps(snap, indent=2))
        time.sleep(60)  # every minute


def start_background_logger():
    t = threading.Thread(target=_status_writer, daemon=True)
    t.start()
