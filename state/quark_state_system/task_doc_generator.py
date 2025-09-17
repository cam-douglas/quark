"""Task Doc Generator – produces a Markdown snapshot of all active roadmap tasks.

Usage:
    from state.quark_state_system.task_doc_generator import generate_tasks_markdown
    filepath = generate_tasks_markdown()
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import List, Dict

from importlib import import_module

# Lazy import to avoid circular dependency at module level
_task_loader = import_module("state.quark_state_system.task_loader")

ROOT_DIR = Path(__file__).resolve().parents[2]  # /quark
DOCS_DIR = ROOT_DIR / "state" / "tasks" / "roadmap_tasks"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

HEADER = """# Quark Roadmap Tasks Snapshot
Generated automatically by `update tasks doc` command.

| Field | Value |
|-------|-------|
| Generated | {generated} |
| Total tasks | {total} |

---
"""

def _rel_link(roadmap_path: str) -> str:
    """Return a relative Markdown link from docs/roadmap_tasks directory."""
    p = Path(roadmap_path).as_posix()
    # ensure it is relative to docs directory
    rel = os.path.relpath(p, DOCS_DIR)
    return rel

def _format_task(task: Dict) -> str:
    """Return one Markdown bullet for a single task dict from task_loader."""
    title = task.get("title", "Untitled task")
    roadmap_file = task.get("roadmap_file", "")
    phase = task.get("phase")
    batch = task.get("batch")
    step = task.get("step")
    label = f"Phase {phase} ▸ Batch {batch} ▸ Step {step}" if phase else "Unscheduled"
    link_part = f" ([roadmap]({_rel_link(roadmap_file)}))" if roadmap_file else ""
    return f"- {label}: {title}{link_part}"

def generate_tasks_markdown(limit_per_section: int | None = None) -> Path:
    """Generate Markdown file listing all pending/in-progress tasks.

    Args:
        limit_per_section: optional cap on tasks per section.

    Returns:
        Path to generated Markdown file.
    """
    # Import inside function to avoid heavy load at import time
    tasks_pending = list(_task_loader.get_tasks(status="pending"))
    tasks_in_progress = list(_task_loader.get_tasks(status="in-progress"))

    sections: List[tuple[str, List[Dict]]] = [
        ("In-Progress", tasks_in_progress),
        ("Pending", tasks_pending),
    ]

    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    filename = f"ROADMAP_TASKS_{_dt.date.today().isoformat()}.md"
    filepath = DOCS_DIR / filename

    with filepath.open("w", encoding="utf-8") as fh:
        fh.write(HEADER.format(generated=timestamp, total=len(tasks_pending)+len(tasks_in_progress)))

        for section, items in sections:
            if not items:
                continue
            fh.write(f"\n## {section} Tasks\n\n")
            if limit_per_section:
                items = items[:limit_per_section]
            for task in items:
                fh.write(_format_task(task) + "\n")

    return filepath

# ---- New helper for ad-hoc task lists from chat -------------------

def generate_tasks_markdown_from_list(tasks: List[str], *, section_title: str = "Suggested Tasks") -> Path:
    """Generate a Markdown file from an explicit list of task strings.

    Args:
        tasks: List of task title strings (plain text).
        section_title: Heading for the section in the doc.

    Returns:
        Path to generated Markdown file.
    """
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    filename = f"roadmap_tasks_{_dt.date.today().isoformat()}_{int(_dt.datetime.utcnow().timestamp())}.md"
    filepath = DOCS_DIR / filename

    with filepath.open("w", encoding="utf-8") as fh:
        fh.write(HEADER.format(generated=timestamp, total=len(tasks)))
        fh.write(f"\n## {section_title}\n\n")
        for t in tasks:
            fh.write(f"- {t}\n")

    return filepath

__all__ = ["generate_tasks_markdown", "generate_tasks_markdown_from_list"]

# If run as script, produce a tasks file
if __name__ == "__main__":
    path = generate_tasks_markdown()
    print(f"✅ Generated tasks doc at {path}")
