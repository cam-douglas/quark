# state/quark_state_system/task_loader.py
"""Central loader for consolidated task YAML files.

Reads the machine-readable task lists in ``state/tasks/`` and exposes helper
functions for the state-system and autonomous agent.
"""
from pathlib import Path
# local cached read
from utilities.performance_utils import read_text_cached
import yaml
from typing import List, Optional, Dict

# Agile phase/step labelling helper
from state.quark_state_system.agile_utils import format_phase_step

_TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"
_PRIORITY_FILES = {
    "high": _TASK_DIR / "tasks_high.yaml",
    "medium": _TASK_DIR / "tasks_medium.yaml",
    "low": _TASK_DIR / "tasks_low.yaml",
    "archive": _TASK_DIR / "tasks_archive.yaml",
    "chat": _TASK_DIR / "chat_tasks.yaml",
}

# ---------------------------------------------------------------------------
# 🚀  PERFORMANCE: Parallel YAML loading (wall-clock ↓ on cold start)
# ---------------------------------------------------------------------------

import concurrent.futures as _cf

_TASKS: List[Dict] = []


def _load_yaml(fp: Path, prio: str):
    """Helper to read *fp* and append priority metadata."""
    try:
        text = fp.read_text()
    except FileNotFoundError:
        return []
    data = yaml.safe_load(read_text_cached(fp)) or []
    for d in data:
        d.setdefault("priority", prio)
    return data


with _cf.ThreadPoolExecutor(max_workers=min(4, len(_PRIORITY_FILES))) as _executor:
    futures = {
        _executor.submit(_load_yaml, fp, prio): prio for prio, fp in _PRIORITY_FILES.items()
    }
    for fut in futures:
        _TASKS.extend(fut.result())

# -- internal helper to persist tasks list to YAML ---------------------------

def _write_tasks_to_file(prio: str):
    fp = _PRIORITY_FILES[prio]
    data = [t for t in _TASKS if t.get("priority") == prio]
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(yaml.safe_dump(data, sort_keys=False))

# write every priority YAML from current _TASKS
def _flush_all():
    for prio in _PRIORITY_FILES:
        _write_tasks_to_file(prio)

# ---------------------------------------------------------------------------
# Public helper – wipe current registry & YAMLs (used for full regeneration)
# ---------------------------------------------------------------------------


def reset_all():
    """Clear in-memory task list and empty all priority YAML files **except** chat tasks.

    Per user rule, ``chat_tasks.yaml`` must never be deleted or overwritten
    without explicit permission. We therefore preserve its contents while
    resetting the other priority files.
    """

    # First, load current chat-specific tasks so we can restore them.
    chat_tasks_fp = _PRIORITY_FILES["chat"]
    preserved_chat_tasks = []
    if chat_tasks_fp.exists():
        import yaml as _yaml
        preserved_chat_tasks = _yaml.safe_load(chat_tasks_fp.read_text()) or []

    # Clear in-memory registry and wipe non-chat YAML files
    _TASKS.clear()
    for prio, fp in _PRIORITY_FILES.items():
        if prio == "chat":
            # Skip deletion to respect user preference
            continue
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text("[]\n")

    # Re-append preserved chat tasks to in-memory registry
    for t in preserved_chat_tasks:
        t.setdefault("priority", "chat")
        _TASKS.append(t)

    # Ensure chat tasks file stays untouched (but rewrite if it didn't exist)
    if not chat_tasks_fp.exists():
        chat_tasks_fp.parent.mkdir(parents=True, exist_ok=True)
        import yaml as _yaml
        chat_tasks_fp.write_text(_yaml.safe_dump(preserved_chat_tasks, sort_keys=False))

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

def task_exists(title: str) -> bool:
    """Public helper to check if a task with *title* already exists."""
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
        if task_exists(title):
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

# ---------------------------------------------------------------------------
# New public helper – allows external modules to add fine-grained tasks while
# ensuring no duplicates and persisting to the correct YAML priority file.
# ---------------------------------------------------------------------------

def add_task(task: Dict):
    """Add *task* dict to registry if not duplicate. Persist to YAML."""
    if task_exists(task.get("title", "")):
        return False
    prio = task.get("priority", "medium")

    # ------------------------------------------------------------------
    # Ensure task title carries a Phase/Step label for Agile visibility
    # If caller already supplied a label we leave it untouched; otherwise
    # we prepend a generic placeholder that can be updated later by planners.
    # ------------------------------------------------------------------
    title = task.get("title", "")
    if "phase" not in title.lower() or "step" not in title.lower():
        placeholder = format_phase_step(0, 0, 0, 0)
        task["title"] = f"{placeholder} — {title}"

    _TASKS.append(task)
    _write_tasks_to_file(prio)
    _flush_all()
    return True

# ---------------------------------------------------------------------------
# Granular task generation from *master_roadmap.md* (naive parser)
# ---------------------------------------------------------------------------

import re

try:
    from state.quark_state_system.advanced_planner import plan as _adv_plan  # type: ignore
except ImportError:
    _adv_plan = None  # fallback later

_PILLAR_HDR = re.compile(r"^Pillar\s+(\d+)\s+–([^:]+):\s+🚧\s+In Progress", re.I)


def _smart_split(text: str):
    """Split *text* on commas/semicolons/and but ignore those inside parentheses."""
    parts = []
    buf = ""
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        # split tokens only when not inside parens
        if depth == 0 and (ch == "," or ch == ";"):
            parts.append(buf.strip())
            buf = ""
            i += 1
            continue
        buf += ch
        i += 1
    if buf.strip():
        parts.append(buf.strip())

    # further split on ' and ' when not inside parentheses
    final = []
    for p in parts:
        if " and " in p and "(" not in p:  # rough heuristic
            final.extend([x.strip() for x in p.split(" and ") if x.strip()])
        else:
            final.append(p)
    return final


def generate_tasks_from_master(master_path: Path, default_priority: str = "medium") -> int:
    """Parse *master_path* for 🚧 In Progress pillar blocks and add fine-grained
    bullet-list tasks below them. Returns number of tasks created."""

    if not master_path.exists():
        return 0

    added = 0
    current_pillar: str | None = None
    capture = False
    buf = master_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    i = 0
    while i < len(buf):
        ln = buf[i]
        hdr = _PILLAR_HDR.match(ln.strip())
        if hdr:
            current_pillar = hdr.group(0).split(":")[0]
            # capture subsequent lines until blank or next header
            i += 1
            segment_lines = []
            while i < len(buf) and buf[i].strip() and not buf[i].startswith("Pillar"):
                segment_lines.append(buf[i].strip())
                i += 1
            # combine lines and split by comma / semicolon / "and"
            segment_text = " ".join(segment_lines)
            tokens = _smart_split(segment_text)
            for tok in tokens:
                item = tok.strip(" .")
                if not item:
                    continue
                if _adv_plan:
                    try:
                        subplans = _adv_plan(item)
                    except Exception as e:
                        subplans = []
                        logger.warning("advanced_planner failed: %s", e)
                else:
                    subplans = []

                if not subplans:
                    # fallback heuristic (design/impl/test/doc)
                    subplans = [
                        {"title": f"{item} – Design"},
                        {"title": f"{item} – Implementation"},
                        {"title": f"{item} – Unit tests"},
                        {"title": f"{item} – Documentation"},
                    ]

                for sp in subplans:
                    task_title = f"{current_pillar} ▶ {sp['title']}"
                    task = {
                        "id": f"roadmap-step-{len(_TASKS)+1}",
                        "title": task_title,
                        "status": "pending",
                        "priority": sp.get("priority", default_priority),
                    }
                    if add_task(task):
                        added += 1
            continue  # skip normal increment
        i += 1
# end while
    _flush_all()
    return added
