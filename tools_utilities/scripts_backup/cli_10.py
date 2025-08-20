#!/usr/bin/env python3
"""
SmallMind Command System CLI
Main command-line interface integrating with neuro agents.
"""

import sys
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from ................................................command_database import CommandDatabase
from ................................................natural_language_parser import NaturalLanguageParser
from ................................................command_executor import CommandExecutor, ExecutionContext

class SmallMindCLI:
    """Main CLI for the SmallMind command system."""
    
    def __init__(self):
        self.db = CommandDatabase()
        self.executor = CommandExecutor(self.db)
        self.interactive_mode = False
        self.growth_orchestrator = None
        
    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="SmallMind Command System - Intelligent command discovery and execution",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Natural language commands
  python -m smallmind.commands "train a neural network with 100 epochs"
  python -m smallmind.commands "show me brain simulation tools"
  python -m smallmind.commands "deploy to AWS using GPU"
  
  # Direct command execution
  python -m smallmind.commands 1.3.1  # Execute command by number
  
  # Interactive mode
  python -m smallmind.commands --interactive
  
  # Help and discovery
  python -m smallmind.commands --help-category "Brain Development"
  python -m smallmind.commands --list-commands
  python -m smallmind.commands --stats
            """
        )
        
        # Command input (natural language or number)
        parser.add_argument('command', nargs='?', help='Command to execute (natural language or number)')
        
        # Execution modes
        parser.add_argument('--interactive', '-i', action='store_true', 
                          help='Start interactive command mode')
        parser.add_argument('--dry-run', action='store_true',
                          help='Show what would be executed without running')
        parser.add_argument('--safe-mode', action='store_true', default=True,
                          help='Enable safety checks (default: True)')
        parser.add_argument('--no-safe-mode', action='store_false', dest='safe_mode',
                          help='Disable safety checks')
        
        # Discovery and help
        parser.add_argument('--list-commands', action='store_true',
                          help='List all available commands')
        parser.add_argument('--list-categories', action='store_true',
                          help='List all command categories')
        parser.add_argument('--help-category', metavar='CATEGORY',
                          help='Show help for a specific category')
        parser.add_argument('--search', metavar='QUERY',
                          help='Search commands by keyword')
        parser.add_argument('--stats', action='store_true',
                          help='Show command database statistics')
        
        # Execution parameters
        parser.add_argument('--timeout', type=float, default=300,
                          help='Command timeout in seconds (default: 300)')
        parser.add_argument('--working-dir', metavar='DIR',
                          help='Working directory for command execution')
        parser.add_argument('--env', action='append', metavar='VAR=VALUE',
                          help='Environment variables (can be used multiple times)')
        
        # Output options
        parser.add_argument('--json', action='store_true',
                          help='Output results in JSON format')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        
        # Exponential growth options
        parser.add_argument('--enable-growth', action='store_true',
                          help='Enable exponential growth with curious agents')
        parser.add_argument('--growth-factor', type=float, default=1.2,
                          help='Exponential growth factor (default: 1.2)')
        parser.add_argument('--curious-agents', type=int, default=6,
                          help='Number of curious agents (default: 6)')
        
        args = parser.parse_args()
        
        # Initialize exponential growth if requested
        if args.enable_growth:
            self._initialize_exponential_growth(args)
        
        # Set up execution context
        context = self._create_execution_context(args)
        
        # Handle different modes
        if args.interactive:
            return self._interactive_mode(context)
        elif args.list_commands:
            return self._list_commands(args.json)
        elif args.list_categories:
            return self._list_categories(args.json)
        elif args.help_category:
            return self._help_category(args.help_category, args.json)
        elif args.search:
            return self._search_commands(args.search, args.json)
        elif args.stats:
            return self._show_stats(args.json)
        elif args.command:
            return self._execute_command(args.command, context, args.json, args.verbose)
        else:
            parser.print_help()
            return 0
    
    def _create_execution_context(self, args) -> ExecutionContext:
        """Create execution context from arguments."""
        # Parse environment variables
        env_vars = dict(os.environ)
        if args.env:
            for env_pair in args.env:
                if '=' in env_pair:
                    key, value = env_pair.split('=', 1)
                    env_vars[key] = value
        
        # Set working directory
        working_dir = args.working_dir or str(Path.cwd())
        
        return ExecutionContext(
            working_directory=working_dir,
            environment_vars=env_vars,
            timeout=args.timeout,
            dry_run=args.dry_run,
            safe_mode=args.safe_mode,
            interactive=False,
            resource_limits={
                "max_memory_mb": 4096,
                "max_cpu_percent": 80,
                "max_execution_time": args.timeout
            }
        )
    
    def _interactive_mode(self, context: ExecutionContext) -> int:
        """Run interactive command mode."""
        print("🧠 SmallMind Interactive Command System")
        print("Type 'help' for assistance, 'quit' to exit")
        print("=" * 50)
        
        self.interactive_mode = True
        
        try:
            while True:
                try:
                    user_input = input("\n🤖 smallmind> ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if user_input.lower() in ['clear', 'cls']:
                        os.system('clear' if os.name != 'nt' else 'cls')
                        continue
                    
                    # Execute command
                    result = self.executor.execute_natural_language(user_input, context)
                    self._display_result(result, json_output=False, verbose=True)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        finally:
            self.interactive_mode = False
        
        return 0
    
    def _execute_command(self, command: str, context: ExecutionContext, json_output: bool, verbose: bool) -> int:
        """Execute a single command."""
        # Check if command is a number (direct command ID)
        if command.isdigit() or '.' in command:
            # Try to get command by number
            all_commands = self.db.search_commands("")  # Get all commands
            for cmd in all_commands:
                if cmd.number == command:
                    result = self.executor.execute_command(cmd, {}, context)
                    self._display_result(result, json_output, verbose)
                    return 0 if result.success else 1
            
            print(f"❌ Command number '{command}' not found")
            return 1
        
        # Natural language execution
        result = self.executor.execute_natural_language(command, context)
        self._display_result(result, json_output, verbose)
        return 0 if result.success else 1
    
    def _list_commands(self, json_output: bool) -> int:
        """List all available commands."""
        commands = self.db.search_commands("")  # Get all commands
        
        if json_output:
            command_data = []
            for cmd in commands:
                command_data.append({
                    "number": cmd.number,
                    "name": cmd.name,
                    "description": cmd.description,
                    "category": cmd.category,
                    "complexity": cmd.complexity
                })
            print(json.dumps(command_data, indent=2))
        else:
            print(f"📋 Available Commands ({len(commands)} total)")
            print("=" * 60)
            
            current_category = None
            for cmd in sorted(commands, key=lambda x: x.number):
                if cmd.category != current_category:
                    current_category = cmd.category
                    categories = self.db.get_categories()
                    cat_name = next((c.name for c in categories if c.number == current_category), current_category)
                    print(f"\n📁 {current_category}: {cat_name}")
                    print("-" * 40)
                
                complexity_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(cmd.complexity, "⚪")
                print(f"  {cmd.number:8} {complexity_emoji} {cmd.name}")
                print(f"           {cmd.description}")
        
        return 0
    
    def _list_categories(self, json_output: bool) -> int:
        """List all command categories."""
        categories = self.db.get_categories()
        
        if json_output:
            category_data = []
            for cat in categories:
                commands = self.db.get_commands_by_category(cat.number)
                category_data.append({
                    "number": cat.number,
                    "name": cat.name,
                    "description": cat.description,
                    "parent": cat.parent,
                    "command_count": len(commands)
                })
            print(json.dumps(category_data, indent=2))
        else:
            print("📁 Command Categories")
            print("=" * 50)
            
            # Group by parent
            main_categories = [cat for cat in categories if cat.parent is None]
            for main_cat in sorted(main_categories, key=lambda x: x.number):
                commands = self.db.get_commands_by_category(main_cat.number)
                print(f"\n{main_cat.number}. {main_cat.name} ({len(commands)} commands)")
                print(f"   {main_cat.description}")
                
                # Show subcategories
                subcategories = [cat for cat in categories if cat.parent == main_cat.number]
                for sub_cat in sorted(subcategories, key=lambda x: x.number):
                    sub_commands = self.db.get_commands_by_category(sub_cat.number)
                    print(f"   {sub_cat.number} {sub_cat.name} ({len(sub_commands)} commands)")
        
        return 0
    
    def _help_category(self, category: str, json_output: bool) -> int:
        """Show help for a specific category."""
        help_content = self.executor.generate_help_content(category)
        
        if json_output:
            print(json.dumps({"help_content": help_content}))
        else:
            print(help_content)
        
        return 0
    
    def _search_commands(self, query: str, json_output: bool) -> int:
        """Search commands by keyword."""
        commands = self.db.search_commands(query)
        
        if json_output:
            command_data = []
            for cmd in commands:
                command_data.append({
                    "number": cmd.number,
                    "name": cmd.name,
                    "description": cmd.description,
                    "category": cmd.category,
                    "keywords": cmd.keywords
                })
            print(json.dumps(command_data, indent=2))
        else:
            print(f"🔍 Search Results for '{query}' ({len(commands)} found)")
            print("=" * 60)
            
            if not commands:
                print("No commands found matching your search.")
                
                # Suggest similar commands
                parser = NaturalLanguageParser()
                suggestions = parser.suggest_similar_commands(query, self.db)
                if suggestions:
                    print("\n💡 You might be looking for:")
                    for suggestion in suggestions:
                        print(f"   {suggestion}")
            else:
                for cmd in commands[:20]:  # Limit to 20 results
                    complexity_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(cmd.complexity, "⚪")
                    print(f"\n{cmd.number:8} {complexity_emoji} {cmd.name}")
                    print(f"         {cmd.description}")
                    print(f"         Keywords: {', '.join(cmd.keywords)}")
        
        return 0
    
    def _show_stats(self, json_output: bool) -> int:
        """Show command database statistics."""
        stats = self.db.get_stats()
        execution_stats = self.executor.get_execution_status()
        
        combined_stats = {**stats, **execution_stats}
        
        if json_output:
            print(json.dumps(combined_stats, indent=2))
        else:
            print("📊 SmallMind Command System Statistics")
            print("=" * 50)
            
            print(f"Database:")
            print(f"  Total Commands: {stats['total_commands']}")
            print(f"  Total Categories: {stats['total_categories']}")
            
            print(f"\nBy Category:")
            for cat in stats['categories']:
                print(f"  {cat['name']}: {cat['count']} commands")
            
            print(f"\nBy Complexity:")
            for comp in stats['complexity']:
                print(f"  {comp['complexity'].title()}: {comp['count']} commands")
            
            print(f"\nExecution Status:")
            print(f"  Active Processes: {execution_stats['active_processes']}")
            print(f"  Execution History: {execution_stats['execution_history_count']}")
            print(f"  Safety Checks: {'Enabled' if execution_stats['safety_checks_enabled'] else 'Disabled'}")
        
        return 0
    
    def _display_result(self, result, json_output: bool, verbose: bool):
        """Display execution result."""
        if json_output:
            # Convert result to dict for JSON serialization
            result_dict = {
                "command_id": result.command_id,
                "success": result.success,
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": result.execution_time,
                "start_time": result.start_time,
                "end_time": result.end_time
            }
            if verbose:
                result_dict.update({
                    "resource_usage": result.resource_usage,
                    "working_directory": result.working_directory,
                    "process_id": result.process_id
                })
            print(json.dumps(result_dict, indent=2))
        else:
            # Text output
            if result.success:
                if not self.interactive_mode:
                    print("✅ Command executed successfully")
                if result.stdout:
                    print(result.stdout)
            else:
                if not self.interactive_mode:
                    print("❌ Command failed")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                elif result.stdout:
                    print(result.stdout)
            
            if verbose and result.execution_time > 0:
                print(f"\n⏱️  Execution time: {result.execution_time:.2f}s")
                if result.resource_usage:
                    if "peak_memory_mb" in result.resource_usage:
                        print(f"💾 Peak memory: {result.resource_usage['peak_memory_mb']:.1f} MB")
                    if "peak_cpu_percent" in result.resource_usage:
                        print(f"🖥️  Peak CPU: {result.resource_usage['peak_cpu_percent']:.1f}%")
                
                # Show growth status if active
                if self.growth_orchestrator and self.growth_orchestrator.is_active:
                    status = self.growth_orchestrator.get_exponential_status()
                    print(f"🧠 Growth: {status['current_commands']} commands (+{status['total_discoveries']} discovered)")
    
    def _initialize_exponential_growth(self, args):
        """Initialize exponential growth system."""
        try:
            from ................................................exponential_orchestrator import ExponentialOrchestrator
            
            project_root = Path(args.working_dir) if args.working_dir else Path.cwd()
            self.growth_orchestrator = ExponentialOrchestrator(project_root)
            
            # Configure growth parameters
            self.growth_orchestrator.exponential_factor = args.growth_factor
            
            # Start exponential growth
            self.growth_orchestrator.start_exponential_growth()
            
            print("🧠 Exponential Growth ACTIVATED!")
            print(f"⚡ Growth factor: {args.growth_factor}x")
            print(f"🤖 Curious agents discovering commands...")
            print("📈 Commands will grow exponentially in the background")
            print()
            
        except ImportError as e:
            print(f"⚠️  Could not initialize exponential growth: {e}")
        except Exception as e:
            print(f"❌ Failed to start exponential growth: {e}")
    
    def __del__(self):
        """Cleanup when CLI is destroyed."""
        if self.growth_orchestrator and self.growth_orchestrator.is_active:
            self.growth_orchestrator.stop_exponential_growth()

def main():
    """Entry point for the CLI."""
    try:
        cli = SmallMindCLI()
        return cli.main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
