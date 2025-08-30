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

# --- NEW IMPORTS ---
# Add the project root to the path to allow for our custom module imports
import os
sys.path.append(os.getcwd())
from quark_state_system.autonomous_agent import AutonomousAgent
# task handling
from quark_state_system.prompt_guardian import PromptGuardian
# goal handling
from state.quark_state_system import goal_manager
from management.rules.roadmaps.roadmap_controller import status_snapshot
# Roadmaps
from management.rules.roadmaps.roadmap_controller import get_all_roadmaps
from quark_state_system import task_loader
# --- Docs helper ---
from utilities.doc_utils import INDEX_PATH
# --- END NEW IMPORTS ---

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
        result = subprocess.run([sys.executable, "quark_state_system/quark_recommendations.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running recommendations: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")

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
    print("📋 QUARK TASKS & GATES")
    print("=" * 40)
    tasks_file = Path("quark_state_system/QUARK_CURRENT_TASKS.md")
    breakdown_file = Path("tasks/PHASE_TODO_BREAKDOWN.md")

    # Show entry-point immediate tasks from QUARK_CURRENT_TASKS.md
    if tasks_file.exists():
        print("\n— Immediate Next Tasks (from QUARK_CURRENT_TASKS.md):")
        try:
            section = []
            capture = False
            for line in tasks_file.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("#### immediate next tasks"):
                    capture = True
                    continue
                if capture:
                    if line.strip().startswith("---") or line.strip() == "":
                        # stop on section break or blank after we've captured items
                        if section:
                            break
                    if line.strip().startswith("- ENT-"):
                        section.append(line.strip())
            if section:
                for item in section:
                    print(f"  {item}")
            else:
                print("  (No ENT-* tasks found. Check the state file.)")
        except Exception as e:
            print(f"  ⚠️ Could not parse immediate tasks: {e}")
    else:
        print("  ⚠️ quark_state_system/QUARK_CURRENT_TASKS.md not found")

    # Show gates from PHASE_TODO_BREAKDOWN.md
    if breakdown_file.exists():
        print("\n— Entry-Point Gates (from tasks/PHASE_TODO_BREAKDOWN.md):")
        try:
            gates = []
            capture = False
            for line in breakdown_file.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("## entry points"):
                    capture = True
                    continue
                if capture:
                    if line.strip().lower().startswith("tasks:"):
                        # After listing Ready-when, we can stop
                        break
                    if line.strip().startswith("- "):
                        gates.append(line.strip())
            if gates:
                for g in gates:
                    print(f"  {g}")
            else:
                print("  (No gates section found. Check the breakdown doc.)")
        except Exception as e:
            print(f"  ⚠️ Could not parse gates: {e}")
    else:
        print("  ⚠️ tasks/PHASE_TODO_BREAKDOWN.md not found")

    print("\nPaths:")
    print("  • quark_state_system/QUARK_CURRENT_TASKS.md")
    print("  • tasks/PHASE_TODO_BREAKDOWN.md")


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
        command = sys.argv[1].lower()
        
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
        elif command == "sync":
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
        elif command == "continuous":
            run_continuous_automation()
        elif command == "tasks":
            run_tasks_overview()
        elif command == "help":
            show_help()
            print(f"\n📚 Documentation index available at: {INDEX_PATH}\n")
        elif command.replace('-', ' ').lower() == "update roadmap":
            print("🔄 Updating roadmap index & tasks …")
            # regenerate index
            from subprocess import run, CalledProcessError
            try:
                run([sys.executable, "tools_utilities/scripts/generate_roadmap_index.py"], check=True)
            except CalledProcessError as e:
                print(f"❌ Failed to regenerate roadmap index: {e}")

            # sync tasks
            from management.rules.roadmaps.roadmap_controller import status_snapshot
            from state.quark_state_system import task_loader
            task_loader.sync_with_roadmaps(status_snapshot())
            print("✅ Roadmap index regenerated and tasks synced.")

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

if __name__ == "__main__":
    main()
