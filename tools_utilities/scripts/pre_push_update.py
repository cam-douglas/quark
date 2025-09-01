#!/usr/bin/env python3
"""Pre-push update script – run automatically by the Git pre-push hook.

Actions:
1. Regenerate ROADMAPS_INDEX.md
2. Validate links in master_roadmap.md (abort with non-zero exit if broken)
3. Sync roadmap snapshot → YAML tasks
4. Update README.md Roadmap Status section (abstract + bullet summary)

Integration: Not simulator-integrated; repository tooling for indexing, validation, or CI.
Rationale: Executed by developers/CI to maintain repo health; not part of runtime simulator loop.
"""
from __future__ import annotations
import subprocess as sp
import sys
from pathlib import Path
import re
import textwrap
import datetime
import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root
RD_DIR = ROOT / "management" / "rules" / "roadmap"
MASTER = RD_DIR / "MASTER_ROADMAP.md"
INDEX = RD_DIR / "ROADMAPS_INDEX.md"
README = ROOT / "README.md"


# ---------------------------------------------------------------------------
# Util helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    print(f"[pre-push] {msg}")


def run(cmd):
    log("$ " + " ".join(cmd))
    sp.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 1. Regenerate index
# ---------------------------------------------------------------------------

def regenerate_index():
    script = ROOT / "tools_utilities" / "scripts" / "generate_roadmap_index.py"
    run([sys.executable, str(script)])


# ---------------------------------------------------------------------------
# 2. Link validation for master roadmap
# ---------------------------------------------------------------------------

link_pat = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

def validate_links():
    broken = []
    for match in link_pat.finditer(MASTER.read_text(encoding="utf-8", errors="ignore")):
        target = match.group(2)
        # skip external URLs
        if target.startswith("http"):
            continue
        path = (MASTER.parent / target).resolve() if not Path(target).is_absolute() else ROOT / target
        if not path.exists():
            broken.append(target)
    if broken:
        log("Broken links found:")
        for b in broken:
            print(" -", b)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 3. Sync roadmap snapshot → YAML tasks
# ---------------------------------------------------------------------------

def sync_tasks():
    from management.rules.roadmap.roadmap_controller import status_snapshot
    from state.quark_state_system import task_loader
    before = len(list(task_loader.get_tasks()))
    task_loader.sync_with_roadmaps(status_snapshot())
    # granular tasks from master roadmap
    master = RD_DIR / "master_roadmap.md"
    added = task_loader.generate_tasks_from_master(master)
    after = len(list(task_loader.get_tasks()))
    log(f"Tasks synced (before={before}, after={after}, added_from_master={added})")


# ---------------------------------------------------------------------------
# 4. README update
# ---------------------------------------------------------------------------

START_TAG = "<!-- ROADMAP_STATUS_START -->"
END_TAG = "<!-- ROADMAP_STATUS_END -->"

def roadmap_summary() -> str:
    from management.rules.roadmap.roadmap_controller import status_snapshot
    snap = status_snapshot()
    done = sum(1 for s in snap.values() if s == "done")
    progress = sum(1 for s in snap.values() if s == "progress")
    planned = sum(1 for s in snap.values() if s == "planned")
    from importlib import import_module
    task_loader = import_module('state.quark_state_system.task_loader')
    pending_tasks = sum(1 for _ in task_loader.get_tasks(status="pending"))
    abstract = textwrap.dedent(f"""
    **Roadmap Status**  
    Date: {datetime.date.today()}  
    Roadmaps: {len(snap)} (✅ {done} · 🚧 {progress} · 📋 {planned})  
    Pending tasks: {pending_tasks}
    """).strip()
    bullets = """\
* Use `python QUARK_STATE_SYSTEM.py \\"update roadmap\\"` after editing roadmap files.  
* Top priorities surfaced via `recommendations` command.  
* Chat-specified tasks live in `chat_tasks.yaml`.
"""
    return abstract + "\n\n" + bullets


def update_readme():
    rd_text = README.read_text(encoding="utf-8")
    if START_TAG not in rd_text:
        rd_text += f"\n\n{START_TAG}\n{END_TAG}\n"
    new_block = roadmap_summary()
    rd_text = re.sub(f"{START_TAG}.*?{END_TAG}", f"{START_TAG}\n{new_block}\n{END_TAG}", rd_text, flags=re.S)
    README.write_text(rd_text, encoding="utf-8")
    log("README roadmap status updated")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    try:
        regenerate_index()
        validate_links()
        sync_tasks()
        update_readme()
        log("Pre-push update completed successfully ✅")
    except Exception as e:  # pragma: no cover
        log(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()