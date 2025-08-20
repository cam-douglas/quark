#!/usr/bin/env python3
"""
Simple test of the command system
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from command_database import CommandDatabase
    print("✅ Successfully imported CommandDatabase")
    
    # Test database initialization
    db = CommandDatabase()
    print("✅ Successfully initialized database")
    
    # Test stats
    stats = db.get_stats()
    print(f"✅ Database stats: {stats['total_commands']} commands, {stats['total_categories']} categories")
    
    # Test simple search
    results = db.search_commands("neural")
    print(f"✅ Search for 'neural' found {len(results)} results")
    
    for cmd in results[:2]:
        print(f"  - {cmd.number}: {cmd.name}")
    
    # Test empty search (get all commands)
    all_commands = db.search_commands("")
    print(f"✅ Total commands in database: {len(all_commands)}")
    
    db.close()
    print("✅ Database test completed successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
