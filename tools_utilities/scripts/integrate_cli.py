

"""
Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
# FILE CREATION: cli for resource integration
"""CLI helper: integrate external resources via ResourceManager.

Usage
-----
python -m state.quark_state_system.integrate_cli <resource_path_or_uri>
"""
from __future__ import annotations

import sys
from pathlib import Path
import yaml
from datetime import datetime

from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager

CHAT_TASKS = Path(__file__).resolve().parents[3] / "state" / "tasks" / "chat_tasks.yaml"


def _append_chat_task(title: str):
    CHAT_TASKS.parent.mkdir(parents=True, exist_ok=True)
    if CHAT_TASKS.exists():
        data = yaml.safe_load(CHAT_TASKS.read_text()) or []
    else:
        data = []
    task = {
        "id": f"chat-{len(data)+1}",
        "title": title,
        "priority": "medium",
        "status": "pending",
        "created": datetime.utcnow().isoformat(timespec="seconds"),
    }
    data.append(task)
    CHAT_TASKS.write_text(yaml.safe_dump(data, sort_keys=False))


def main():
    if len(sys.argv) < 2:
        print("Usage: integrate_cli <resource>")
        sys.exit(1)

    resource = sys.argv[1]
    rm = ResourceManager()
    try:
        rid = rm.register_resource(resource)
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        _append_chat_task(f"Integration failed for {resource}: {e}")
        sys.exit(1)

    print(f"✅ Resource integrated (id={rid})")
    _append_chat_task(f"Resource integrated: {resource} (id={rid})")


if __name__ == "__main__":
    main()
