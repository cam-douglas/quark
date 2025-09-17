#!/usr/bin/env python3
"""Command Handlers Module - All QUARK state system command implementations.

Handles all command processing for the QUARK state system launcher.

Integration: Core command processing for QuarkDriver and AutonomousAgent operations.
Rationale: Centralized command logic with clean separation of concerns.
"""

import sys
import subprocess
import os

# Import QUARK modules
from state.quark_state_system import ask_quark
from state.quark_state_system.dynamic_state_summary import get_dynamic_state_summary, format_state_summary
from state.quark_state_system import task_loader
from management.rules.roadmap.roadmap_controller import status_snapshot

def run_quick_status():
    """Run the dynamic status checker using live roadmap data."""
    print("🔍 DYNAMIC STATUS CHECK")
    print("=" * 40)
    try:
        state = get_dynamic_state_summary()
        print(format_state_summary(state))
    except Exception as e:
        print(f"❌ Error getting dynamic status: {e}")
        # Fallback to basic status
        print("📊 Basic Status:")
        print(f"   Tasks pending: {len(list(task_loader.get_tasks(status='pending')))}")
        print(f"   Tasks completed: {len(list(task_loader.get_tasks(status='completed')))}")

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
        result = subprocess.run([sys.executable, "state/quark_state_system/sync_quark_state.py"],
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
        from importlib import import_module
        AutonomousAgent = import_module("state.quark_state_system.autonomous_agent").AutonomousAgent
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
        from importlib import import_module
        PromptGuardian = import_module("state.quark_state_system.prompt_guardian").PromptGuardian
        guardian = PromptGuardian(project_root)
        # Sample prompt for demonstration
        sample_prompt = "Let's work on the Core Infrastructure & Data Strategy"
        sample_action = {"action_type": "define_schema", "domain": "infrastructure"}
        guardian.validate_prompt(sample_prompt, sample_action)
    except Exception as e:
        print(f"❌ Error running prompt guardian: {e}")

def run_tasks_overview():
    """Show immediate next tasks and gates (entry-point focused)."""
    try:
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
        from importlib import import_module
        QuarkDriver = import_module("state.quark_state_system.quark_driver").QuarkDriver
        driver = QuarkDriver(project_root)
        driver.run_continuous()
    except Exception as e:
        print(f"❌ Error during continuous automation: {e}")

def handle_list_tasks():
    """Handle the list-tasks command."""
    pending_tasks = list(task_loader.get_tasks(status="pending"))
    print(f"📋 In-Progress Tasks ({len(pending_tasks)}):")
    for task in pending_tasks:
        print(f"   ID: {task.get('id', 'no-id')}")
        print(f"   Title: {task.get('title', 'No title')}")
        print(f"   Section: {task.get('section_subtitle', 'Unknown section')}")
        print("   ---")

def handle_complete_task(task_id: str):
    """Handle the complete-task command with enhanced completion system."""
    if not task_id:
        print("⚠️  Provide a task ID: python QUARK_STATE_SYSTEM.py complete-task <task_id>")
        return False

    # Try enhanced completion system first
    try:
        from .task_completion_integration import complete_task
        success = complete_task(task_id)
        if success:
            print(f"✅ Task marked complete and archived: {task_id}")
            
            # Show pending roadmap confirmations if any
            from .task_completion_integration import get_completion_status
            status = get_completion_status()
            if status.get("pending_roadmap_completions", 0) > 0:
                print(f"📋 {status['pending_roadmap_completions']} roadmap tasks awaiting user confirmation")
                print("   Use 'python quark_state_system_launcher.py show-pending' to view")
        else:
            print(f"❌ Failed to complete task: {task_id}")
        return success
    except ImportError:
        # Fallback to existing system
        success = task_loader.mark_task_complete(task_id)
        if success:
            print(f"✅ Task marked complete and archived: {task_id}")
        else:
            print(f"❌ Failed to complete task: {task_id}")
        return success

def handle_update_roadmap():
    """Handle roadmap update commands."""
    print("🔄 Running full state refresh (roadmap index, task sync, README update)…")
    try:
        result = subprocess.run([sys.executable, "tools_utilities/scripts/pre_push_update.py"], check=True)
        print("✅ Quark state refreshed (index, tasks, README).")
    except subprocess.CalledProcessError as e:
        print(f"❌ State refresh failed: {e}")

def handle_add_chat_task(title: str):
    """Handle adding chat tasks."""
    if not title:
        print("⚠️  Provide a task title: python QUARK_STATE_SYSTEM.py add-chat-task \"My new task\"")
        return False

    try:
        from state.quark_state_system.chat_tasks import add_chat_task
        added = add_chat_task(title)
        if added.get("duplicate"):
            print("ℹ️  Task already exists – duplicate skipped.")
        else:
            print(f"✅ Chat task added: {title} (priority=medium)")
        return True
    except Exception as e:
        print(f"❌ Error adding chat task: {e}")
        return False

def handle_show_pending_confirmations():
    """Show pending roadmap task confirmations."""
    try:
        from .task_completion_integration import show_pending_confirmations
        show_pending_confirmations()
    except ImportError:
        print("📋 Enhanced completion system not available")
        print("   No pending confirmations to show")

def handle_confirm_completions(task_ids: list):
    """Confirm roadmap task completions."""
    if not task_ids:
        print("⚠️  Provide task IDs: python quark_state_system_launcher.py confirm-completions <task_id1> <task_id2> ...")
        return False

    try:
        from .task_completion_integration import confirm_roadmap_completions
        results = confirm_roadmap_completions(task_ids)
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"📋 Confirmed {successful}/{total} roadmap task completions")
        
        for task_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {task_id}")
        
        return successful == total
        
    except ImportError:
        print("❌ Enhanced completion system not available")
        print("   Cannot confirm roadmap completions")
        return False

def handle_completion_status():
    """Show completion system status."""
    try:
        from .task_completion_integration import get_completion_status
        status = get_completion_status()
        
        print("🔄 TASK COMPLETION SYSTEM STATUS")
        print("=" * 40)
        print(f"Enhanced completion: {'✅ Available' if status['enhanced_completion_available'] else '❌ Not available'}")
        print(f"Existing system: {'✅ Available' if status['existing_system_available'] else '❌ Not available'}")
        print(f"Pending roadmap confirmations: {status['pending_roadmap_completions']}")
        
        if status.get('pending_tasks'):
            print("\nPending roadmap tasks:")
            for task in status['pending_tasks']:
                print(f"  🔹 {task['task_id']}")
                print(f"     {task['description']}")
        
    except ImportError:
        print("📋 Basic completion system available")
        print("   Enhanced features not available")
