#!/usr/bin/env python3
"""
Quark TODO System - Master Orchestrator
========================================
Unified entry point for all Quark operations.

Usage:
    todo [natural language command]
    
Examples:
    todo what's next
    todo plan new task
    todo validate foundation layer
    todo work on cerebellum task
    todo run tests
    todo commit changes
    
Special Commands:
    todo workflow new_feature     → Execute new feature workflow
    todo workflow daily_standup   → Execute daily standup workflow
    todo history                  → Show command history
    todo help                     → Show all available commands
"""

import sys
import argparse
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from state.todo.core import (
    ContextAnalyzer,
    CommandRouter,
    StateManager,
    WorkflowOrchestrator
)


def main():
    """Main entry point for TODO system."""
    parser = argparse.ArgumentParser(
        description="Quark TODO System - Natural Language Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # We'll handle help ourselves
    )
    
    parser.add_argument(
        'command',
        nargs='*',
        default=['help'],
        help='Natural language command'
    )
    
    args = parser.parse_args()
    
    # Join command parts
    command = ' '.join(args.command)
    
    # Initialize components
    analyzer = ContextAnalyzer()
    router = CommandRouter(PROJECT_ROOT)
    state_manager = StateManager(PROJECT_ROOT / 'state' / 'todo')
    orchestrator = WorkflowOrchestrator(PROJECT_ROOT)
    
    # Special commands
    if command == 'help':
        router._show_help()
        return 0
    
    elif command == 'history':
        print("\n📜 Recent Commands:")
        print("=" * 50)
        for cmd in state_manager.get_recent_commands(10):
            print(f"  • {cmd}")
        return 0
    
    elif command.startswith('workflow'):
        # Execute a workflow
        parts = command.split()
        if len(parts) > 1:
            workflow_name = parts[1]
            return orchestrator.execute_workflow(workflow_name, router)
        else:
            print("\n📋 Available Workflows:")
            for wf in orchestrator.list_workflows():
                print(f"  • todo workflow {wf}")
            return 0
    
    elif command in ['', 'what now', "what's next", 'next']:
        # Show context-aware suggestions
        context = analyzer.get_current_context()
        suggestions = state_manager.get_suggestions()
        
        print("\n🎯 Suggested Next Actions:")
        print("=" * 50)
        
        if context.get('has_changes'):
            print("📝 You have uncommitted changes:")
            for file in context.get('recent_files', [])[:3]:
                print(f"  • {file}")
        
        if suggestions:
            print("\n💡 Based on your recent work:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. todo {suggestion}")
        
        print("\n🔄 Or start a workflow:")
        workflow = orchestrator.suggest_workflow(context)
        if workflow:
            print(f"  • todo workflow {workflow}")
        
        return 0
    
    # Analyze natural language command
    system, action, params = analyzer.analyze_command(command)
    
    # Record command
    state_manager.record_command(command, system, action)
    
    # Update state based on command
    if system == 'task' and action == 'execute' and params.get('task_id'):
        state_manager.set_current_task(params['task_id'])
    elif system == 'validate':
        state_manager.update_validation_time()
    
    # Route to appropriate system
    print(f"\n🚀 Executing: {command}")
    print("-" * 50)
    
    exit_code = router.route_command(system, action, params)
    
    # Post-execution suggestions
    if exit_code == 0 and system == 'task' and action in ['plan', 'create']:
        print("\n💡 Next: todo work on [task-id]")
    elif exit_code == 0 and system == 'validate':
        print("\n💡 Next: todo commit changes")
    
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
