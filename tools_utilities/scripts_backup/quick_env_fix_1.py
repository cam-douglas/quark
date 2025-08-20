#!/usr/bin/env python3
"""
Quick Environment Fix for Small-Mind

This script fixes the immediate import issues without trying to install
problematic packages like pybullet.
"""

import sys
import os
from pathlib import Path

def fix_import_issues():
    """Fix the relative import issues in integrate_all.py."""
    print("🔧 Fixing import issues in integrate_all.py...")
    
    integrate_file = Path("models/agent_hub/integrate_all.py")
    if not integrate_file.exists():
        print("❌ integrate_all.py not found")
        return False
    
    # Read the file
    with open(integrate_file, 'r') as f:
        content = f.read()
    
    # Fix the problematic imports
    import_fixes = {
        "from planner import": "from .....................................................planner import",
        "from registry import": "from .....................................................registry import", 
        "from router import": "from .....................................................router import",
        "from runner import": "from .....................................................runner import",
        "from intelligent_feedback import": "from .....................................................intelligent_feedback import",
        "from cloud_training import": "from .....................................................cloud_training import"
    }
    
    original_content = content
    for old_import, new_import in import_fixes.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"  ✅ Fixed: {old_import} → {new_import}")
    
    # Write the fixed content back
    if content != original_content:
        with open(integrate_file, 'w') as f:
            f.write(content)
        print("✅ Import issues fixed in integrate_all.py")
        return True
    else:
        print("ℹ️  No import issues found to fix")
        return True

def create_simple_runner():
    """Create a simple runner script that avoids import issues."""
    print("📝 Creating simple runner script...")
    
    runner_content = '''#!/usr/bin/env python3
"""
Simple Runner for Small-Mind Integration

This script runs the integration without complex imports.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🧠 Small-Mind Simple Integration Runner")
    print("=" * 40)
    
    try:
        # Try to import and run the main integration
        from models.agent_hub.integrate_all import IntegratedSuperIntelligence
        
        print("🚀 Starting Integrated Super Intelligence...")
        intelligence = IntegratedSuperIntelligence()
        intelligence.start()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: make env-setup")
        return 1
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    runner_path = Path("run_integration.py")
    with open(runner_path, 'w') as f:
        f.write(runner_content)
    
    # Make it executable
    runner_path.chmod(0o755)
    print(f"✅ Created simple runner: {runner_path}")
    return str(runner_path)

def main():
    """Main fix routine."""
    print("🧠 Small-Mind Quick Environment Fix")
    print("=" * 40)
    
    # Fix import issues
    if not fix_import_issues():
        print("❌ Failed to fix import issues")
        return 1
    
    # Create simple runner
    runner_path = create_simple_runner()
    
    print("\n🎉 Environment fixes applied!")
    print(f"💡 You can now run: python {runner_path}")
    print("💡 Or try the original: python models/agent_hub/integrate_all.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
