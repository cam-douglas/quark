#!/usr/bin/env python3
"""
🧠 QUARK STATE SYSTEM LAUNCHER - MAIN ENTRY POINT

This is the MAIN ENTRY POINT for QUARK's state checking and implementation system.
All state management flows through this central hub.

USAGE:
    python quark_state_system_launcher.py                    # Show system overview
    python quark_state_system_launcher.py status            # Quick status check
    python quark_state_system_launcher.py sync              # Sync all state files
    python quark_state_system_launcher.py recommendations   # Get QUARK's recommendations
    python quark_state_system_launcher.py help              # Show help information

MODULAR ARCHITECTURE:
    This launcher coordinates specialized modules in state/quark_state_system/:
    - command_handlers.py     - All command implementations
    - system_overview.py      - Help and system information
    - chat_task_manager.py    - Ad-hoc task management
    - task_loader.py          - Main task coordination
    - roadmap_integration.py  - Dynamic roadmap task extraction
    - sprint_management.py    - Phase/Batch/Step organization
"""

import sys
import os
from pathlib import Path

# Ensure both project root and ./state are on PYTHONPATH
sys.path.append(os.getcwd())
STATE_DIR = Path(__file__).resolve().parent / "state"
sys.path.append(str(STATE_DIR))

# Import modular components
from state.quark_state_system.command_handlers import (
    run_quick_status, run_sync, run_autonomous_agent,
    run_prompt_validation, run_tasks_overview, activate_driver_mode,
    run_continuous_automation, handle_list_tasks, handle_complete_task,
    handle_update_roadmap, handle_add_chat_task, handle_show_pending_confirmations,
    handle_confirm_completions, handle_completion_status
)
from state.quark_state_system.system_overview import show_system_overview, show_help
from state.quark_state_system.chat_task_manager import ChatTaskManager

# Import for goal handling
from state.quark_state_system import goal_manager
from state.quark_state_system.task_management import task_loader
from management.rules.roadmap.roadmap_controller import status_snapshot

def main():
    """Main entry point for the QUARK state system launcher."""
    if len(sys.argv) > 1:
        # Support natural-language multi-word commands.
        # If the user prefixes with 'run quark state system', strip it.
        tokens = [t.lower() for t in sys.argv[1:]]
        if tokens[:4] == ["run", "quark", "state", "system"]:
            tokens = tokens[4:]
        command_raw = " ".join(tokens)
        command = command_raw.replace("-", " ").lower().strip()

        if command == "status":
            run_quick_status()
            snap = status_snapshot()
            total = len(snap)
            remaining = sum(1 for s in snap.values() if s not in ("done", "✅"))
            print(f"\n🗺️  Roadmaps: {total} (remaining {remaining})")
            pending = list(task_loader.get_tasks(status="pending"))
            print(f"📋 Tasks pending: {len(pending)}")

        elif command == "recommendations":
            print("🔮 TOP PRIORITY TASKS (Roadmap-driven)")
            # Generate fresh tasks from roadmaps
            added = task_loader.generate_tasks_from_active_roadmaps()
            tasks = task_loader.next_actions()
            for t in tasks:
                # Use formatted_label instead of priority
                label = t.get('formatted_label', 'No Label')
                title = t.get('title', 'No Title')
                task_id = t.get('id', 'no-id')
                print(f"- {label}: {title}")
            if not tasks:
                print("(All roadmap tasks are completed 🎉)")

        elif command in ["sync", "sync quark", "sync quark state"]:
            run_sync()

        elif command in ["proceed", "continue", "evolve", "execute"]:
            nxt = goal_manager.next_goal()
            if nxt:
                print(f"🚀 Proceeding with next roadmap goal: {nxt['title']} (prio={nxt['priority']})")
            else:
                print("✅ No pending roadmap goals – system is up-to-date.")
            run_autonomous_agent()

        elif command == "validate":
            run_prompt_validation()

        elif command == "activate":
            activate_driver_mode()

        elif tokens and tokens[0] == "continuous":
            # Support optional numeric limit: `continuous 3` executes first 3 tasks of phase
            if len(tokens) > 1 and tokens[1].isdigit():
                limit = int(tokens[1])
                print(f"🚀 Running continuous-phase automation (limit={limit})")
                from state.quark_state_system.quark_driver import QuarkDriver
                driver = QuarkDriver(os.getcwd())
                driver.run_phase_tasks(limit)
            else:
                run_continuous_automation()

        elif command == "tasks":
            run_tasks_overview()

        elif command == "help":
            show_help()

        elif command in ["update roadmap", "update quark", "update quark state", "refresh roadmap"]:
            handle_update_roadmap()

        elif command in ["add-chat-task", "add to tasks", "update tasks"]:
            title = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
            handle_add_chat_task(title)

        elif command in ["complete-task", "mark-complete", "task-done"]:
            task_id = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
            handle_complete_task(task_id)

        elif command in ["list-tasks", "show-tasks"]:
            handle_list_tasks()

        elif command in ["show-pending", "pending-confirmations"]:
            handle_show_pending_confirmations()

        elif command in ["confirm-completions", "confirm-tasks"]:
            task_ids = sys.argv[2:] if len(sys.argv) > 2 else []
            handle_confirm_completions(task_ids)

        elif command in ["completion-status", "task-status"]:
            handle_completion_status()

        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python quark_state_system_launcher.py help' for available commands")
    else:
        show_system_overview()

if __name__ == "__main__":
    main()
