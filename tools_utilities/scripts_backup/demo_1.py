#!/usr/bin/env python3
"""
SmallMind Command System Demo

Demonstrates the capabilities of the command system including:
- Command discovery and database population
- Natural language parsing
- Command execution with safety checks
- Help system and interactive features
"""

import os, sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Try relative imports first (when run as module)
    from ................................................command_database import CommandDatabase
    from ................................................natural_language_parser import NaturalLanguageParser
    from ................................................command_executor import CommandExecutor, ExecutionContext
    from ................................................neuro_integration import NeuroAgentConnector, SmartCommandDiscovery
except ImportError:
    # Fallback to direct imports (when run as script)
    from command_database import CommandDatabase
    from natural_language_parser import NaturalLanguageParser
    from command_executor import CommandExecutor, ExecutionContext
    from neuro_integration import NeuroAgentConnector, SmartCommandDiscovery

def demo_database():
    """Demonstrate command database functionality."""
    print("📊 Command Database Demo")
    print("=" * 50)
    
    # Initialize database
    db = CommandDatabase()
    
    # Show statistics
    stats = db.get_stats()
    print(f"Total Commands: {stats['total_commands']}")
    print(f"Total Categories: {stats['total_categories']}")
    
    print("\nCategories:")
    for cat in stats['categories']:
        print(f"  {cat['name']}: {cat['count']} commands")
    
    print("\nComplexity Distribution:")
    for comp in stats['complexity']:
        print(f"  {comp['complexity'].title()}: {comp['count']} commands")
    
    # Test search
    print("\n🔍 Search Results for 'neural':")
    results = db.search_commands("neural")
    for cmd in results[:3]:
        print(f"  {cmd.number} - {cmd.name}: {cmd.description}")
    
    db.close()
    print("✅ Database demo completed\n")

def demo_natural_language():
    """Demonstrate natural language parsing."""
    print("🧠 Natural Language Parser Demo")
    print("=" * 50)
    
    db = CommandDatabase()
    parser = NaturalLanguageParser()
    
    test_inputs = [
        "train a neural network with 100 epochs",
        "show me brain simulation commands", 
        "deploy to AWS using GPU instance",
        "help with optimization",
        "list all available models"
    ]
    
    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        intent = parser.parse_user_input(test_input, db)
        
        print(f"  Intent: {intent.intent_type}")
        print(f"  Confidence: {intent.confidence:.2f}")
        print(f"  Commands found: {len(intent.command_ids)}")
        print(f"  Parameters: {intent.parameters}")
        
        if intent.requires_confirmation:
            print(f"  ⚠️  Warning: {intent.safety_warning}")
    
    db.close()
    print("✅ Natural language demo completed\n")

def demo_command_execution():
    """Demonstrate command execution."""
    print("⚡ Command Execution Demo")
    print("=" * 50)
    
    db = CommandDatabase()
    executor = CommandExecutor(db)
    
    # Test help command
    print("Testing help command:")
    result = executor.execute_natural_language("help")
    print(f"Success: {result.success}")
    print(f"Output preview: {result.stdout[:200]}...")
    
    # Test dry run
    print("\nTesting dry run mode:")
    context = ExecutionContext(
        working_directory=str(Path.cwd()),
        environment_vars=dict(os.environ),
        timeout=30.0,
        dry_run=True,
        safe_mode=True,
        interactive=False,
        resource_limits={}
    )
    
    result = executor.execute_natural_language("list models", context)
    print(f"Dry run result: {result.stdout}")
    
    # Test search
    print("\nTesting command search:")
    result = executor.execute_natural_language("search for neural commands")
    print(f"Search success: {result.success}")
    if result.stdout:
        print(f"Results preview: {result.stdout[:300]}...")
    
    db.close()
    print("✅ Command execution demo completed\n")

def demo_neuro_integration():
    """Demonstrate neuro agent integration."""
    print("🔗 Neuro Integration Demo")
    print("=" * 50)
    
    connector = NeuroAgentConnector()
    status = connector.get_status()
    
    print(f"Neuro Available: {status['available']}")
    print(f"Available Commands: {status['commands']}")
    
    if connector.is_available():
        print("\n🔍 Testing smart command discovery...")
        discovery = SmartCommandDiscovery(connector)
        commands = discovery.discover_project_commands()
        
        print(f"Discovered {len(commands)} project-specific commands:")
        for cmd in commands[:5]:
            print(f"  • {cmd['name']}: {cmd['description']}")
    else:
        print("\n⚠️  Neuro agents not available - using fallback discovery")
        discovery = SmartCommandDiscovery(connector)
        commands = discovery.discover_project_commands()
        print(f"Fallback discovered {len(commands)} commands:")
        for cmd in commands[:3]:
            print(f"  • {cmd['name']}: {cmd['description']}")
    
    print("✅ Neuro integration demo completed\n")

def demo_interactive_features():
    """Demonstrate interactive features."""
    print("🎮 Interactive Features Demo")
    print("=" * 50)
    
    db = CommandDatabase()
    
    # Test command completion
    print("Command completion examples:")
    completions = ["neuro", "train", "aws", "help"]
    
    for partial in completions:
        commands = db.search_commands(partial)
        print(f"  '{partial}' -> {len(commands)} matches")
        for cmd in commands[:2]:
            print(f"    • {cmd.name}")
    
    # Test category browsing
    print("\nCategory browsing:")
    categories = db.get_categories()
    main_cats = [cat for cat in categories if cat.parent is None]
    
    for cat in main_cats[:3]:
        commands = db.get_commands_by_category(cat.number)
        print(f"  {cat.number}. {cat.name}: {len(commands)} commands")
    
    db.close()
    print("✅ Interactive features demo completed\n")

def main():
    """Run all demos."""
    print("🚀 SmallMind Command System Demonstration")
    print("=" * 60)
    print()
    
    try:
        # Run all demos
        demo_database()
        demo_natural_language()
        demo_command_execution()
        demo_neuro_integration()
        demo_interactive_features()
        
        print("🎉 All demos completed successfully!")
        print("\nTo try the interactive CLI, run:")
        print("  python -m smallmind.commands --interactive")
        print("\nFor help with specific commands:")
        print("  python -m smallmind.commands 'help neural'")
        print("  python -m smallmind.commands --list-commands")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
