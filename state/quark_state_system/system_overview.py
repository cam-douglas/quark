#!/usr/bin/env python3
"""System Overview Module - Display system information and help.

Provides system overview, help information, and documentation for QUARK state system.

Integration: User interface and documentation for QuarkDriver and AutonomousAgent.
Rationale: Centralized user guidance and system information display.
"""

from tools_utilities.scripts.doc_utils import INDEX_PATH

def show_system_overview():
    """Display the QUARK state system overview."""
    print("🧠 QUARK STATE SYSTEM - MAIN ENTRY POINT")
    print("=" * 60)
    print()
    print("📁 STATE SYSTEM ORGANIZATION:")
    print("   🚀 quark_state_system_launcher.py - Main Entry Point")
    print("   📁 state/quark_state_system/     - State System Modules")
    print("     ├── 📊 quark_state_system.md      - Main State Index")
    print("     ├── 🔍 dynamic_state_summary.py   - Live Status Generator")
    print("     ├── 🎯 quark_recommendations.py   - Intelligent Recommendations")
    print("     ├── 🔄 sync_quark_state.py        - State Synchronization")
    print("     ├── 📋 task_loader.py             - Task Management")
    print("     ├── 🗺️  roadmap_integration.py    - Roadmap Task Extraction")
    print("     └── 📖 usage_guide.md             - Usage Documentation")
    print()
    print("🤖 AUTONOMOUS & COMPLIANCE SYSTEMS:")
    print("   🧠 autonomous_agent.py   - Core agent for goal-driven execution")
    print("   🛡️  prompt_guardian.py      - Validates all prompts against rules")
    print("   🔬 compliance_engine.py    - Enforces biological/safety rules")
    print("   📜 biological_constraints.py - Centralized rule definitions")
    print("   🗺️  roadmap_controller.py   - Unifies all project roadmaps")
    print()
    print("💡 QUICK COMMANDS:")
    print("   python quark_state_system_launcher.py execute       - RUN THE AUTONOMOUS AGENT")
    print("   python quark_state_system_launcher.py status        - Quick status")
    print("   python quark_state_system_launcher.py recommendations - Get guidance")
    print("   python quark_state_system_launcher.py tasks         - Show immediate tasks & gates")
    print("   python quark_state_system_launcher.py sync          - Sync all files")
    print("   cat state/quark_state_system/quark_state_system.md  - Read full state")
    print()
    print("🎯 ENTRY POINT: quark_state_system_launcher.py is the main entry point")
    print("   for all QUARK state operations. All state modules are organized in")
    print("   the state/quark_state_system/ directory for easy access and maintenance.")

def show_help():
    """Show detailed help information."""
    print("🧠 QUARK STATE SYSTEM - HELP")
    print("=" * 50)
    print()
    print("COMMANDS:")
    print("   continuous      - Run the Autonomous Agent continuously until all goals are complete.")
    print("   execute         - (OR: proceed, continue, evolve) Activate the Autonomous Agent for one goal cycle")
    print("   validate        - Run a sample validation with the Prompt Guardian")
    print("   status          - Quick status check")
    print("   recommendations - Get QUARK's intelligent recommendations")
    print("   tasks           - Show immediate next tasks and entry-point gates")
    print("   list-tasks      - Show all in-progress tasks with IDs")
    print("   complete-task   - Mark a task as complete: complete-task <task_id>")
    print("   sync            - Synchronize all state files")
    print("   help            - Show this help message")
    print("   (no args)       - Show system overview")
    print()
    print("FILE DESCRIPTIONS:")
    print("   state/quark_state_system/quark_state_system.md - Master state index")
    print("   state/quark_state_system/current_status.md    - Live status and progress")
    print("   state/quark_state_system/usage_guide.md       - Complete usage guide")
    print("   state/quark_state_system/task_loader.py       - Task management system")
    print("   state/quark_state_system/roadmap_integration.py - Roadmap task extraction")
    print()
    print("WORKFLOW:")
    print("   1. Always start with: python quark_state_system_launcher.py status")
    print("   2. Get guidance: python quark_state_system_launcher.py recommendations")
    print("   3. After changes: python quark_state_system_launcher.py sync")
    print("   4. Read details: cat state/quark_state_system/quark_state_system.md")
    print()
    print("💡 TIP: quark_state_system_launcher.py is your ONE-STOP entry point")
    print("   for all QUARK state operations! All state modules are organized in")
    print("   the state/quark_state_system/ directory.")
    print(f"\n📚 Documentation index available at: {INDEX_PATH}\n")
