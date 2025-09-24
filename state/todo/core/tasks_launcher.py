#!/usr/bin/env python3
"""
Quark Task Management System - Main Entry Point
================================================
Interactive guide for planning, executing, and tracking roadmap tasks.

Usage:
    python quark_tasks.py [command] [options]

Commands:
    plan        - Interactive task planning guide
    execute     - Guide for executing current tasks
    track       - Track progress on active tasks
    review      - Review completed tasks
    create      - Create new task from template
    generate    - Generate tasks from roadmap milestones
    sync        - Sync tasks with roadmap documents
    list        - List all tasks by category/status
    help        - Show detailed help

Run 'python quark_tasks.py help' for detailed command reference.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from state.tasks.core import TaskManager


def show_help():
    """Display help information."""
    print(__doc__)
    print("\n" + "=" * 60)
    print("COMMAND REFERENCE")
    print("=" * 60)
    print("\n📋 Task Planning")
    print("  python quark_tasks.py plan [--category CAT]")
    print("\n🚀 Task Execution")
    print("  python quark_tasks.py execute [--task-id ID]")
    print("\n📊 Progress Tracking")
    print("  python quark_tasks.py track [--status STATUS]")
    print("\n🔄 Roadmap Integration")
    print("  python quark_tasks.py generate [--stage N] [--category CAT]")
    print("  python quark_tasks.py sync [--update-status]")
    print("\n📚 Task Management")
    print("  python quark_tasks.py list [--filter FILTER]")
    print("  python quark_tasks.py create")
    print("  python quark_tasks.py review")
    print("\n📁 File Locations:")
    print("  • Tasks: state/tasks/roadmap_tasks/")
    print("  • Template: state/tasks/validation/templates/TASK_TEMPLATE.md")
    print("  • Core modules: state/tasks/core/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quark Task Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command", nargs="?", default="help",
        choices=["plan", "execute", "track", "review", "create", 
                "generate", "sync", "list", "help"],
        help="Task management command"
    )
    
    parser.add_argument("--category", help="Task category")
    parser.add_argument("--task-id", help="Specific task ID")
    parser.add_argument("--status", help="Filter by status")
    parser.add_argument("--template", help="Template name")
    parser.add_argument("--filter", help="Filter criteria")
    parser.add_argument("--stage", type=int, help="Biological stage (1-6)")
    parser.add_argument("--update-status", action="store_true", 
                       help="Update status")
    
    args = parser.parse_args()
    
    # Initialize and route
    try:
        manager = TaskManager(PROJECT_ROOT)
        
        if args.command == "generate":
            manager.generate_from_roadmap(args.stage, args.category)
        elif args.command == "sync":
            manager.sync_roadmap_status(args.update_status)
        elif args.command == "plan":
            manager.plan_task(args.category)
        elif args.command == "execute":
            manager.execute_task(args.task_id)
        elif args.command == "track":
            manager.track_progress(args.status)
        elif args.command == "review":
            manager.review_tasks()
        elif args.command == "create":
            manager.create_task(args.template)
        elif args.command == "list":
            manager.list_tasks(args.filter)
        else:
            show_help()
            
    except KeyboardInterrupt:
        print("\n⚠️ Cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
