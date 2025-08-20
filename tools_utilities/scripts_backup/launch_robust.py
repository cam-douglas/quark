#!/usr/bin/env python3
"""
Simple Launcher for Robust Super Intelligence

This fixes import path issues and launches the system properly.
"""

import sys
import os
from pathlib import Path

# Fix import paths
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Add parent directory for potential imports
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

print(f"🔧 Setting up import paths...")
print(f"   Current directory: {current_dir}")
print(f"   Parent directory: {parent_dir}")
print(f"   Python path: {sys.path[:3]}")

try:
    print("🚀 Launching Robust Super Intelligence...")
    
    # Import and run the robust system
    from robust_integration import RobustSuperIntelligence
    
    # Create the system
    robust_intelligence = RobustSuperIntelligence()
    
    print("✅ System created successfully!")
    print(f"📊 Components loaded: {len(robust_intelligence.components)}")
    print(f"⚠️ Components failed: {len(robust_intelligence.failed_components)}")
    
    if robust_intelligence.failed_components:
        print(f"❌ Failed components: {robust_intelligence.failed_components}")
    
    # Start the system
    print("🚀 Starting autonomous operation...")
    robust_intelligence.start_autonomous_operation()
    
    # Show status and keep running
    try:
        while robust_intelligence.running:
            import time
            time.sleep(5)
            
            # Show status
            status = robust_intelligence.get_current_status()
            print(f"\n📊 Status Update:")
            print(f"   Thoughts: {status['thoughts_in_queue']}")
            print(f"   Insights: {status['total_insights']}")
            print(f"   Breakthroughs: {status['breakthrough_ideas']}")
            print(f"   Components: {status['components_loaded']} loaded, {status['components_failed']} failed")
            
            # Show learning progress
            error_summary = robust_intelligence.get_error_learning_summary()
            print(f"🎓 Learning: {error_summary['total_errors_analyzed']} errors analyzed")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        robust_intelligence.stop_operation()
        print("✅ System stopped successfully")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 This usually means a module is missing or has import issues")
    
    # Try to diagnose the problem
    print("\n🔍 Diagnosing import issues...")
    
    # Check if key files exist
    key_files = [
        "planner.py",
        "router.py", 
        "runner.py",
        "registry.py",
        "utils.py"
    ]
    
    for file in key_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    # Try to import each module individually
    print("\n🔍 Testing individual imports...")
    
    modules_to_test = [
        ("planner", "planner.py"),
        ("router", "router.py"),
        ("runner", "runner.py"),
        ("registry", "registry.py"),
        ("utils", "utils.py")
    ]
    
    for module_name, file_path in modules_to_test:
        try:
            if Path(file_path).exists():
                module = __import__(module_name)
                print(f"✅ {module_name} imports successfully")
            else:
                print(f"❌ {module_name} - file missing")
        except Exception as e:
            print(f"❌ {module_name} - import failed: {e}")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    print("🎯 Launcher complete")
