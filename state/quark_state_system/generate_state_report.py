"""
Generate a markdown report summarising Quark’s current state.

• Roadmap status via `management.rules.roadmap.roadmap_controller.get_roadmap_status_map`
  (totals milestones, completed, in-progress)
• Pending tasks via `state.quark_state_system.task_loader`
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

from importlib import import_module
from management.rules.roadmap.roadmap_controller import status_snapshot

loader = import_module("state.quark_state_system.task_loader")

REPORT_PATH = Path(__file__).resolve().parent / "quark_state_system.md"


def build_report() -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    status_map = status_snapshot.get_roadmap_status_map()
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
