"""Chat task helper – allows adding chat-specified tasks into the pipeline."""
from __future__ import annotations
from typing import Dict

from . import task_loader


def _next_chat_id() -> str:
    existing = [t for t in task_loader._TASKS if t.get("id", "").startswith("chat-")]  # type: ignore
    return f"chat-{len(existing)+1}"


def add_chat_task(title: str, priority: str = "medium") -> Dict:
    """Append a chat-driven task (if not duplicate) and persist to YAML.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
    if any(t.get("title") == title for t in task_loader._TASKS):
        return {"duplicate": True}

    task = {
        "id": _next_chat_id(),
        "title": title,
        "priority": priority,
        "status": "pending",
        "source": "chat",
    }
    task_loader._TASKS.append(task)  # type: ignore
    task_loader._write_tasks_to_file("chat")  # type: ignore
    return task
