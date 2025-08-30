"""Generate QUARK_STATE.md from canonical sources.

Sources:
- Roadmap status via management.rules.roadmaps.roadmap_controller.get_roadmap_status_map
- Task YAMLs via state.quark_state_system.task_loader

Produces/overwrites QUARK_STATE.md located in same directory.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

from importlib import import_module

roadmap_ctrl = import_module("management.rules.roadmaps.roadmap_controller")
loader = import_module("state.quark_state_system.task_loader")

REPORT_PATH = Path(__file__).resolve().parent / "QUARK_STATE.md"


def build_report() -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    status_map = roadmap_ctrl.get_roadmap_status_map()
    tasks = list(loader.next_actions(limit=5))

    md = [
        "# 🧠 QUARK STATE REPORT",
        f"_Auto-generated: {timestamp}_\n",
        "## Roadmap Status Summary",
        "| Roadmap | Status |",
        "|---------|--------|",
    ]
    for k, v in sorted(status_map.items()):
        md.append(f"| {k} | {v} |")

    md.append("\n## Next Priority Tasks")
    if tasks:
        for t in tasks:
            md.append(f"- **{t.get('title','Unnamed Task')}** (priority: {t.get('priority')})")
    else:
        md.append("_No pending tasks found_")

    return "\n".join(md) + "\n"


def main() -> None:
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Written {REPORT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
