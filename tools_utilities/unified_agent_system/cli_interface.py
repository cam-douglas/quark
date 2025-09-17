"""
CLI Interface for Unified Agent System

Command-line interface for the unified agent system.

Author: Quark AI
Date: 2025-01-27
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add both agent packages to the path
agent_delegation_path = Path(__file__).parent.parent / "agent_delegation"
orchestrator_path = Path(__file__).parent / "multi_agent_orchestrator"
sys.path.insert(0, str(agent_delegation_path))
sys.path.insert(0, str(orchestrator_path))


# Import and run the main CLI interface
try:
    from core_system import AgentDelegationSystem
    from agent_types import DelegatedTask, TaskStatus
    from core_orchestrator import MultiAgentOrchestrator
    from orchestrator_types import SpecializedTask, AgentRole
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback: run the CLI directly
    import subprocess
    cli_path = agent_delegation_path / "core_system.py"
    sys.argv[0] = str(cli_path)
    subprocess.run([sys.executable, str(cli_path)] + sys.argv[1:])
    sys.exit(0)

from delegation_handler import run_delegation_system
from orchestrator_handler import run_orchestrator_system
from demo_system import run_unified_demo, show_system_status


def main():
    """Unified CLI interface for both agent systems"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Agent System - Delegation & Orchestration")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='system', help='Available systems')
    
    # Agent Delegation System
    delegation_parser = subparsers.add_parser('delegation', help='General agent delegation system')
    delegation_parser.add_argument("--delegate", nargs=4, metavar=("TITLE", "DESCRIPTION", "AGENT_TYPE", "PRIORITY"),
                                  help="Delegate a task to a background agent")
    delegation_parser.add_argument("--status", help="Get status of a specific task")
    delegation_parser.add_argument("--list", action="store_true", help="List all tasks")
    delegation_parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    delegation_parser.add_argument("--cancel", help="Cancel a task")
    
    # Multi-Agent Orchestrator
    orchestrator_parser = subparsers.add_parser('orchestrator', help='Specialized engineering role orchestrator')
    orchestrator_parser.add_argument("--delegate", nargs=4, metavar=("TITLE", "DESCRIPTION", "ROLE", "PRIORITY"),
                                    help="Delegate a task to a specialized agent")
    orchestrator_parser.add_argument("--status", help="Get status of a specific task")
    orchestrator_parser.add_argument("--list", action="store_true", help="List all tasks")
    orchestrator_parser.add_argument("--report", action="store_true", help="Generate orchestration report")
    orchestrator_parser.add_argument("--role-tasks", help="List tasks for a specific role")
    orchestrator_parser.add_argument("--example", action="store_true", help="Run example orchestration")
    
    # Unified commands
    parser.add_argument("--demo", action="store_true", help="Run unified demo")
    parser.add_argument("--status-all", action="store_true", help="Show status of both systems")
    
    args = parser.parse_args()
    
    if args.demo:
        run_unified_demo()
    elif args.status_all:
        show_system_status()
    elif args.system == 'delegation':
        run_delegation_system(args)
    elif args.system == 'orchestrator':
        run_orchestrator_system(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
