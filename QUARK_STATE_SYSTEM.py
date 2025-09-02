#!/usr/bin/env python3
"""
🧠 QUARK STATE SYSTEM - MAIN ENTRY POINT

This is the MAIN ENTRY POINT for QUARK's state checking and implementation system.
All state management flows through this central hub.

USAGE:
    python QUARK_STATE_SYSTEM.py                    # Show system overview
    python QUARK_STATE_SYSTEM.py status            # Quick status check
    python QUARK_STATE_SYSTEM.py sync              # Sync all state files
    python QUARK_STATE_SYSTEM.py recommendations   # Get QUARK's recommendations
    python QUARK_STATE_SYSTEM.py help              # Show help information

SYSTEM ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                    QUARK_STATE_SYSTEM.py                   │
    │                     🚀 MAIN ENTRY POINT                    │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                quark_state_system/                         │
    │                    📁 STATE SYSTEM DIRECTORY               │
    │                                                             │
    │              ┌─────────────────────────────┐               │
    │              │      QUARK_STATE.md         │               │
    │              │        📊 MASTER STATE      │               │
    │              │      (Single source of truth)               │
    │              └─────────────────────────────┘               │
    │                                                             │
    │              ┌─────────────────────────────┐               │
    │              │    check_quark_state.py     │               │
    │              │        🔍 Quick Status     │               │
    │              └─────────────────────────────┘               │
    │                                                             │
    │              ┌─────────────────────────────┐               │
    │              │   quark_recommendations.py  │               │
    │              │      🎯 Smart Guidance      │               │
    │              └─────────────────────────────┘               │
    │                                                             │
    │              ┌─────────────────────────────┐               │
    │              │    sync_quark_state.py      │               │
    │              │      🔄 Auto-Sync           │               │
    │              └─────────────────────────────┘               │
    │                                                             │
    │              ┌─────────────────────────────┐               │
    │              │   QUARK_CURRENT_TASKS.md    │               │
    │              │        📋 Task Status       │               │
    │              └─────────────────────────────┘               │
    │                                                             │
    │              ┌─────────────────────────────┐               │
    │              │      QUARK_ROADMAP.md       │               │
    │              │        🗺️  Development      │               │
    │              └─────────────────────────────┘               │
    └─────────────────────────────────────────────────────────────┘

QUICK START:
    1. Check current status: python QUARK_STATE_SYSTEM.py status
    2. Get recommendations: python QUARK_STATE_SYSTEM.py recommendations
    3. Sync all files: python QUARK_STATE_SYSTEM.py sync
    4. Read full state: cat quark_state_system/QUARK_STATE.md
"""

import sys
import subprocess
from pathlib import Path
import re

# --- NEW IMPORTS ---
# Ensure both project root and ./state are on PYTHONPATH
import os
from pathlib import Path
sys.path.append(os.getcwd())
STATE_DIR = Path(__file__).resolve().parent / "state"
sys.path.append(str(STATE_DIR))

# --- Module alias & lazy import helpers ------------------------------------
import sys as _sys
from importlib import import_module

# Ensure legacy code importing `quark_state_system.*` continues to work
try:
    import state.quark_state_system as _sq
    _sys.modules.setdefault("quark_state_system", _sq)
except ModuleNotFoundError:
    pass

# -- Lazy wrappers ----------------------------------------------------------

def _lazy_import(path: str, attr: str):
    mod = import_module(path)
    return getattr(mod, attr)

# Note: AutonomousAgent and PromptGuardian are only needed in specific
# commands. We defer their import to avoid heavy dependencies during simple
# sync/update commands.
# They are fetched via _lazy_import inside the respective functions.
# goal handling
from state.quark_state_system import goal_manager, ask_quark
from management.rules.roadmap.roadmap_controller import status_snapshot
# Roadmaps
from management.rules.roadmap.roadmap_controller import get_all_roadmaps
from quark_state_system import task_loader
# --- Docs helper ---
from tools_utilities.scripts.doc_utils import INDEX_PATH
# --- END NEW IMPORTS ---

# --- Roadmap Task Template Loader -----------------------------------------
import yaml
TASK_TMPL_PATH = Path("management/configurations/project/roadmap_task_format.yaml")
if TASK_TMPL_PATH.exists():
    with TASK_TMPL_PATH.open("r", encoding="utf-8") as _f:
        _task_cfg = yaml.safe_load(_f)
    ROADMAP_TASK_TMPL: str = _task_cfg["roadmap_task_format"].get("task_template", "")
    ROADMAP_USE_MARKERS: bool = _task_cfg["roadmap_task_format"].get("use_markers", True)
else:
    ROADMAP_TASK_TMPL = "Task: {milestone}\n• Biological Goal: {goal}\n• Acceptance KPI: {kpi}\n• Suggested Method (SOTA ML): {method}"
    ROADMAP_USE_MARKERS = True
# --------------------------------------------------------------------------
# Canonical roadmap path (single source of truth)
MASTER_ROADMAP: Path = Path("management/rules/roadmap/MASTER_ROADMAP.md").resolve()
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Roadmap status updater
# --------------------------------------------------------------------------


def _compute_status(md_text: str) -> str:
    """Return Planned / In Progress / Complete for given roadmap markdown."""
    # Collect task lines (milestones, goals, SOTA practices)
    task_lines = [ln for ln in md_text.splitlines() if ln.strip().startswith("* [")]
    if not task_lines:
        return "Planned"
    done_flags = ["DONE" in ln for ln in task_lines]
    if all(done_flags):
        return "Complete"
    if any(done_flags):
        return "In Progress"
    return "Planned"


status_pat = re.compile(r"\*\*Roadmap Status:\*\*\s*(.+)")


def _mark_done(text: str) -> str:
    """Append DONE to milestone/goal/practice bullet lines missing it."""
    new_lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("* [") and "DONE" not in ln:
            ln = ln + " DONE"
        new_lines.append(ln)
    return "\n".join(new_lines)


def update_roadmap_statuses() -> None:
    """Scan roadmap markdown files and update Roadmap Status tags based on DONE markers."""
    from management.rules.roadmap import roadmap_controller

    for meta in roadmap_controller.get_all_roadmaps():
        if meta["format"] != "markdown":
            continue
        path = Path(meta["path"])
        text = path.read_text(encoding="utf-8")
        new_status = _compute_status(text)
        # Replace or insert status line
        if new_status == "Complete":
            text = _mark_done(text)
        if status_pat.search(text):
            text = status_pat.sub(f"**Roadmap Status:** 📋 {new_status}", text)
        else:
            # insert after canonical header if missing
            lines = text.splitlines()
            insert_idx = 0
            for i, ln in enumerate(lines[:20]):
                if ln.startswith("> **Canonical"):
                    insert_idx = i + 1
                    break
            lines.insert(insert_idx, f"**Roadmap Status:** 📋 {new_status}")
            text = "\n".join(lines)
        path.write_text(text, encoding="utf-8")


# Hook into task sync or CLI
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "update-roadmap-status":
        update_roadmap_statuses()
        print("Roadmap statuses updated.")


def show_system_overview():
    """Display the QUARK state system overview."""
    print("🧠 QUARK STATE SYSTEM - MAIN ENTRY POINT")
    print("=" * 60)
    print()
    print("📁 STATE SYSTEM ORGANIZATION:")
    print("   🚀 QUARK_STATE_SYSTEM.py     - THIS FILE (Main Entry Point)")
    print("   📁 quark_state_system/       - State System Directory")
    print("     ├── 📊 QUARK_STATE.md            - Master State File")
    print("     ├── 🔍 check_quark_state.py      - Quick Status Checker")
    print("     ├── 🎯 quark_recommendations.py  - Intelligent Recommendations")
    print("     ├── 🔄 sync_quark_state.py       - State Synchronization")
    print("     ├── 📋 QUARK_CURRENT_TASKS.md    - Current Task Status")
    print("     ├── 🗺️  QUARK_ROADMAP.md         - Development Roadmap")
    print("     └── 📖 QUARK_STATE_ORGANIZATION.md - Organization Guide")
    print()
    print("🤖 AUTONOMOUS & COMPLIANCE SYSTEMS:")
    print("   🧠 autonomous_agent.py   - Core agent for goal-driven execution")
    print("   🛡️  prompt_guardian.py      - Validates all prompts against rules")
    print("   🔬 compliance_engine.py    - Enforces biological/safety rules")
    print("   📜 biological_constraints.py - Centralized rule definitions")
    print("   🗺️  roadmap_controller.py   - Unifies all project roadmaps")
    print()
    print("💡 QUICK COMMANDS:")
    print("   python QUARK_STATE_SYSTEM.py execute       - RUN THE AUTONOMOUS AGENT")
    print("   python QUARK_STATE_SYSTEM.py status        - Quick status")
    print("   python QUARK_STATE_SYSTEM.py recommendations - Get guidance")
    print("   python QUARK_STATE_SYSTEM.py tasks         - Show immediate tasks & gates")
    print("   python QUARK_STATE_SYSTEM.py sync          - Sync all files")
    print("   python check_quark_state.py                - Direct status check")
    print("   cat quark_state_system/QUARK_STATE.md      - Read full state")
    print()
    print("🎯 ENTRY POINT: This file (QUARK_STATE_SYSTEM.py) is the main entry point")
    print("   for all QUARK state operations. All state files are organized in")
    print("   the quark_state_system/ directory for easy access and maintenance.")

def run_quick_status():
    """Run the quick status checker."""
    print("🔍 QUICK STATUS CHECK")
    print("=" * 40)
    try:
        result = subprocess.run([sys.executable, "quark_state_system/check_quark_state.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running status check: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")

def run_recommendations():
    """Run the recommendations engine."""
    print("🎯 QUARK RECOMMENDATIONS")
    print("=" * 40)
    try:
        print(ask_quark("quark recommendations"))
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")

def run_sync():
    """Run the state synchronization."""
    print("🔄 STATE SYNCHRONIZATION")
    print("=" * 40)
    try:
        result = subprocess.run([sys.executable, "quark_state_system/sync_quark_state.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running sync: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")

def run_autonomous_agent():
    """Initializes and runs the autonomous agent to execute one goal cycle."""
    print("🤖 AUTONOMOUS AGENT ACTIVATION")
    print("=" * 40)
    project_root = os.getcwd()
    try:
        AutonomousAgent = _lazy_import("state.quark_state_system.autonomous_agent", "AutonomousAgent")
        agent = AutonomousAgent(project_root)
        agent.execute_next_goal()
        print("\nAGENT: Goal cycle complete.")
    except Exception as e:
        print(f"❌ Error running autonomous agent: {e}")

def run_prompt_validation():
    """Runs the prompt guardian to validate a sample prompt."""
    print("🛡️ PROMPT GUARDIAN VALIDATION")
    print("=" * 40)
    project_root = os.getcwd()
    try:
        PromptGuardian = _lazy_import("state.quark_state_system.prompt_guardian", "PromptGuardian")
        guardian = PromptGuardian(project_root)
        # This is a sample prompt and action for demonstration purposes.
        # In a real integration, this would come from the user input.
        sample_prompt = "Let's work on the Core Infrastructure & Data Strategy"
        sample_action = { "action_type": "define_schema", "domain": "infrastructure" }
        guardian.validate_prompt(sample_prompt, sample_action)
    except Exception as e:
        print(f"❌ Error running prompt guardian: {e}")


def run_tasks_overview():
    """Show immediate next tasks and gates (entry-point focused)."""
    # Always sync tasks with the latest roadmap snapshot so the list reflects
    # the current *master_roadmap* status before we display anything. This keeps
    # the CLI output aligned with roadmap progress even if the YAML files are
    # stale or the user hasn't run an explicit `update roadmap` recently.
    try:
        from management.rules.roadmap.roadmap_controller import status_snapshot
        from state.quark_state_system import task_loader
        task_loader.sync_with_roadmaps(status_snapshot())
    except Exception as sync_err:
        print(f"⚠️  Task sync failed – proceeding with cached tasks: {sync_err}")
    print("📋 QUARK TASKS (Generated from Roadmap)")
    print("=" * 40)
    try:
        print(ask_quark("quark tasks"))
    except Exception as e:
        print(f"❌ Error generating tasks overview: {e}")


def activate_driver_mode():
    """Explains the new active driver mode."""
    print("🚀 QUARK ACTIVE DRIVER MODE IS NOW THE DEFAULT")
    print("=" * 60)
    print("The Quark State System has been configured for self-determined execution.")
    print("From this point forward, my actions are governed by the logic defined in:")
    print("   ▶ quark_state_system/quark_driver.py")
    print("\nThis means:")
    print("  1. I will always be aware of the current roadmap goal.")
    print("  2. Every prompt you provide will be validated against this goal and all biological/safety rules.")
    print("  3. If a prompt is generic (e.g., 'proceed'), I will autonomously execute the next roadmap task.")
    print("  4. To run all roadmap tasks automatically, use the 'run-continuous' command.")
    print("\nThe system is now active. Please provide your next instruction.")


def run_continuous_automation():
    """Activates the driver's continuous automation mode."""
    project_root = os.getcwd()
    try:
        from quark_state_system.quark_driver import QuarkDriver
        driver = QuarkDriver(project_root)
        driver.run_continuous()
    except Exception as e:
        print(f"❌ Error during continuous automation: {e}")


def show_help():
    """Show detailed help information."""
    print("🧠 QUARK STATE SYSTEM - HELP")
    print("=" * 50)
    print()
    print("COMMANDS:")
    print("   continuous      - Run the Autonomous Agent continuously until all goals are complete.")
    print("   execute         - (OR: proceed, continue, evolve) Activate the Autonomous Agent for one goal cycle")
    print("   validate        - Run a sample validation with the Prompt Guardian")
    print("   status          - Quick status check (same as check_quark_state.py)")
    print("   recommendations - Get QUARK's intelligent recommendations")
    print("   tasks           - Show immediate next tasks and entry-point gates")
    print("   sync            - Synchronize all state files")
    print("   help            - Show this help message")
    print("   (no args)       - Show system overview")
    print()
    print("FILE DESCRIPTIONS:")
    print("   quark_state_system/QUARK_STATE.md        - Master state file (single source of truth)")
    print("   quark_state_system/check_quark_state.py  - Quick status extractor")
    print("   quark_state_system/quark_recommendations.py - Smart recommendation engine")
    print("   quark_state_system/sync_quark_state.py   - Automatic state synchronization")
    print("   quark_state_system/QUARK_CURRENT_TASKS.md - Current task status")
    print("   quark_state_system/QUARK_ROADMAP.md      - Development roadmap")
    print("   quark_state_system/QUARK_STATE_ORGANIZATION.md - Organization guide")
    print()
    print("WORKFLOW:")
    print("   1. Always start with: python QUARK_STATE_SYSTEM.py status")
    print("   2. Get guidance: python QUARK_STATE_SYSTEM.py recommendations")
    print("   3. After changes: python QUARK_STATE_SYSTEM.py sync")
    print("   4. Read details: cat quark_state_system/QUARK_STATE.md")
    print()
    print("💡 TIP: This file (QUARK_STATE_SYSTEM.py) is your ONE-STOP entry point")
    print("   for all QUARK state operations! All state files are organized in")
    print("   the quark_state_system/ directory.")

def main():
    """Main entry point for the QUARK state system."""
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
            tasks = task_loader.next_actions()
            for t in tasks:
                print(f"- [{t['priority'].upper()}] {t['title']} (id={t['id']})")
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
            print(f"\n📚 Documentation index available at: {INDEX_PATH}\n")
        elif command in ["update roadmap", "update quark", "update quark state", "refresh roadmap"]:
            print("🔄 Running full state refresh (roadmap index, task sync, README update)…")
            from subprocess import run, CalledProcessError
            try:
                run([sys.executable, "tools_utilities/scripts/pre_push_update.py"], check=True)
            except CalledProcessError as e:
                print(f"❌ State refresh failed: {e}")
            else:
                print("✅ Quark state refreshed (index, tasks, README).")

        elif command in ["add-chat-task", "add to tasks", "update tasks"]:
            title = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
            if not title:
                print("⚠️  Provide a task title: python QUARK_STATE_SYSTEM.py add-chat-task \"My new task\"")
                sys.exit(1)
            from state.quark_state_system.chat_tasks import add_chat_task
            added = add_chat_task(title)
            if added.get("duplicate"):
                print("ℹ️  Task already exists – duplicate skipped.")
            else:
                print(f"✅ Chat task added: {title} (priority=medium)")
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python QUARK_STATE_SYSTEM.py help' for available commands")
    else:
        show_system_overview()

# --------------------------------------------------------------------------
# Recommendation & Task generation
# --------------------------------------------------------------------------

TASK_DIR = Path("state/tasks")
TASK_DIR.mkdir(parents=True, exist_ok=True)


def _extract_priority_tasks(md_text: str) -> list[str]:
    """Return ordered list of bullet lines from three priority sections."""
    lines = md_text.splitlines()
    capture = False
    tasks = []
    for ln in lines:
        if ln.startswith("**Engineering Milestones") or ln.startswith("**Biological Goals") or ln.startswith("**SOTA ML Practices"):
            capture = True
            continue
        if capture and ln.startswith("**"):
            capture = False  # next section reached
        if capture and ln.strip().startswith("* "):
            tasks.append(ln.strip("* "))
    return tasks


def _write_task_yaml(filename: str, task_lines: list[str]):
    path = TASK_DIR / filename
    import yaml, datetime
    data = {"generated": str(datetime.date.today()), "tasks": task_lines}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def generate_recommendations():
    """Identify in-progress roadmap, surface top tasks, and populate YAML files."""
    from management.rules.roadmap import roadmap_controller

    in_progress = None
    for meta in roadmap_controller.get_all_roadmaps():
        if meta["format"] != "markdown":
            continue
        text = Path(meta["path"]).read_text(encoding="utf-8")
        if "**Roadmap Status:** 📋 In Progress" in text:
            in_progress = (meta["path"], text)
            break
    if not in_progress:
        print("No roadmap marked In Progress.")
        return []

    path, md = in_progress
    tasks = _extract_priority_tasks(md)
    top = tasks[:5]

    # Split remainder into high/medium/low thirds
    remaining = tasks[5:]
    n = len(remaining)
    hi = remaining[: n // 3]
    med = remaining[n // 3 : 2 * n // 3]
    lo = remaining[2 * n // 3 :]

    _write_task_yaml("tasks_high.yaml", hi)
    _write_task_yaml("tasks_medium.yaml", med)
    _write_task_yaml("tasks_low.yaml", lo)

    return top


# CLI helper
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "recommendations":
        top = generate_recommendations()
        for t in top:
            print("-", t)
        sys.exit(0)

# --------------------------------------------------------------------------
# Ad-hoc Chat Task Management
# --------------------------------------------------------------------------

CHAT_TASKS_DIR = TASK_DIR  # reuse state/tasks/
ARCHIVE_PATH = TASK_DIR / "tasks_archive.yaml"


def _unique_chat_filename(title: str) -> Path:
    base = title.lower().replace(" ", "_")[:30]
    i = 1
    while True:
        path = CHAT_TASKS_DIR / f"chat_tasks_{base}_{i}.yaml"
        if not path.exists():
            return path
        i += 1


def create_chat_tasks(title: str, items: list[str]):
    """Create new chat task YAML with provided items."""
    path = _unique_chat_filename(title)
    import yaml, datetime
    data = {"created": str(datetime.date.today()), "title": title, "tasks": items}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    print(f"Chat tasks file created: {path}")


def get_latest_chat_file() -> Path | None:
    files = sorted(CHAT_TASKS_DIR.glob("chat_tasks_*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def task_status():
    """Show remaining chat tasks; archive completed ones."""
    import yaml
    chat_file = get_latest_chat_file()
    if not chat_file:
        print("No chat-tasks file found.")
        return
    data = yaml.safe_load(chat_file.read_text()) or {}
    tasks = data.get("tasks", [])
    # consider lines with DONE suffix as completed
    remaining = [t for t in tasks if "DONE" not in t]
    completed = [t for t in tasks if "DONE" in t]
    # archive completed
    if completed:
        arch = []
        if ARCHIVE_PATH.exists():
            arch = yaml.safe_load(ARCHIVE_PATH.read_text()) or []
        arch.extend(completed)
        ARCHIVE_PATH.write_text(yaml.safe_dump(arch), encoding="utf-8")
    # rewrite chat file with remaining
    data["tasks"] = remaining
    chat_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    print("Remaining chat tasks:")
    for t in remaining:
        print("-", t)


# CLI hooks
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "recommendations":
            top = generate_recommendations(); [print("-", t) for t in top]; sys.exit(0)
        if cmd == "update-roadmap-status":
            update_roadmap_statuses(); sys.exit(0)
        if cmd == "task-status":
            task_status(); sys.exit(0)
        if cmd == "create-chat-tasks":
            title = sys.argv[2]
            items = sys.argv[3:]
            create_chat_tasks(title, items); sys.exit(0)

if __name__ == "__main__":
    main()
