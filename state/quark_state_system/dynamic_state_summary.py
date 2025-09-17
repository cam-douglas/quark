#!/usr/bin/env python3
"""Dynamic State Summary - Replaces static quark_state_system.md

This module dynamically generates QUARK's current state from live roadmap files
instead of relying on static documentation that can become outdated.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""

from pathlib import Path
from typing import Dict
import re
import datetime


def get_dynamic_state_summary() -> Dict:
    """Generate current state summary from live roadmap data only."""
    roadmap_dir = Path("management/rules/roadmap")

    # Get active roadmaps
    active_roadmaps = {}
    all_roadmaps = {}

    for roadmap_file in roadmap_dir.glob("*.md"):
        # Skip any files in archive directories
        if "archive" in str(roadmap_file).lower() or "backup" in str(roadmap_file).lower():
            continue
        try:
            content = roadmap_file.read_text(encoding='utf-8')

            # Extract stage info
            stage_match = re.search(r'\*Stage (\d+) - ([^*]+)\*', content)
            if stage_match:
                stage_num = stage_match.group(1)
                stage_name = stage_match.group(2).strip()
            else:
                stage_name = roadmap_file.stem.replace('_rules', '').replace('_', ' ').title()
                stage_num = "Unknown"

            # Extract status
            status_match = re.search(r'\*\*Roadmap Status:\*\*\s*📋\s*(.+)', content)
            status = status_match.group(1).strip() if status_match else "Unknown"

            all_roadmaps[stage_name] = {
                "stage_num": stage_num,
                "status": status,
                "file": roadmap_file.name
            }

            if "In Progress" in status:
                active_roadmaps[stage_name] = all_roadmaps[stage_name]

        except Exception as e:
            print(f"Warning: Could not read {roadmap_file}: {e}")
            continue

    # Get task counts
    from . import task_loader
    try:
        pending_tasks = list(task_loader.get_tasks(status="pending"))
        completed_tasks = list(task_loader.get_tasks(status="completed"))

        task_summary = {
            "pending": len(pending_tasks),
            "completed": len(completed_tasks),
            "total": len(pending_tasks) + len(completed_tasks)
        }
    except Exception:
        task_summary = {"pending": 0, "completed": 0, "total": 0}

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "active_roadmaps": active_roadmaps,
        "all_roadmaps": all_roadmaps,
        "task_summary": task_summary,
        "current_stage": _determine_current_stage(active_roadmaps),
        "next_milestone": _determine_next_milestone(all_roadmaps, active_roadmaps)
    }


def _determine_current_stage(active_roadmaps: Dict) -> str:
    """Determine current development stage from active roadmaps."""
    if not active_roadmaps:
        return "No active stage"

    # Find the highest stage number that's in progress
    max_stage = 0
    current_stage_name = ""

    for stage_name, info in active_roadmaps.items():
        try:
            stage_num = int(info.get("stage_num", 0))
            if stage_num > max_stage:
                max_stage = stage_num
                current_stage_name = stage_name
        except (ValueError, TypeError):
            continue

    return f"Stage {max_stage}: {current_stage_name}" if current_stage_name else "Unknown Stage"


def _determine_next_milestone(all_roadmaps: Dict, active_roadmaps: Dict) -> str:
    """Determine next milestone based on roadmap progression."""
    if not active_roadmaps:
        # Find the first stage that's planned or ready
        for stage_name, info in all_roadmaps.items():
            if info["status"] in ["Planned", "Ready"]:
                return f"Start {stage_name}"
        return "No clear next milestone"

    # Find next stage after current active ones
    active_stage_nums = []
    for info in active_roadmaps.values():
        try:
            stage_num = int(info.get("stage_num", 0))
            active_stage_nums.append(stage_num)
        except (ValueError, TypeError):
            continue

    if active_stage_nums:
        next_stage_num = max(active_stage_nums) + 1
        for stage_name, info in all_roadmaps.items():
            try:
                if int(info.get("stage_num", 0)) == next_stage_num:
                    return f"Prepare for {stage_name}"
            except (ValueError, TypeError):
                continue

    return "Continue current stage development"


def format_state_summary(state: Dict) -> str:
    """Format state summary as readable text."""
    lines = [
        "# 🧠 QUARK DYNAMIC STATE SUMMARY",
        "",
        f"**Generated**: {state['timestamp']}",
        f"**Current Stage**: {state['current_stage']}",
        f"**Next Milestone**: {state['next_milestone']}",
        "",
        "## 📊 TASK SUMMARY",
        f"- Pending: {state['task_summary']['pending']}",
        f"- Completed: {state['task_summary']['completed']}",
        f"- Total: {state['task_summary']['total']}",
        "",
        "## 🗺️ ACTIVE ROADMAPS",
    ]

    if state['active_roadmaps']:
        for stage_name, info in state['active_roadmaps'].items():
            lines.append(f"- **{stage_name}** (Stage {info['stage_num']}): {info['status']}")
    else:
        lines.append("- No active roadmaps")

    lines.extend([
        "",
        "## 📋 ALL ROADMAPS STATUS",
    ])

    for stage_name, info in sorted(state['all_roadmaps'].items(),
                                   key=lambda x: int(x[1].get('stage_num', 999)) if x[1].get('stage_num', '').isdigit() else 999):
        status_emoji = "🚀" if "Progress" in info['status'] else "📋" if info['status'] == "Planned" else "✅"
        lines.append(f"- {status_emoji} **Stage {info['stage_num']}: {stage_name}** - {info['status']}")

    lines.extend([
        "",
        "## 📁 **DETAILED STATE INFORMATION**",
        "",
        "- **[Current Status](current_status.md)** - Live progress and active roadmaps",
        "- **[Integration Status](integration_status.md)** - System integrations and health",
        "- **[Development History](development_history.md)** - Completed milestones and phases",
        "- **[Usage Guide](usage_guide.md)** - Commands and workflows",
        "",
        "---",
        "",
        "*This summary is generated dynamically from live roadmap files.*",
        "*No static documentation is used to ensure accuracy.*"
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    state = get_dynamic_state_summary()
    print(format_state_summary(state))
