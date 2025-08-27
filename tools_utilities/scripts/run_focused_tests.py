#!/usr/bin/env python3
"""
🧪 FOCUSED REPOSITORY TESTING RUNNER
Purpose: Execute the focused testing agent on core quark repository files only
Inputs: None (auto-discovers core project files)
Outputs: Complete test results with simulation data for core files
Seeds: 42
Dependencies: focused_repo_testing_agent
"""

import sys
import time
from pathlib import Path

# Add the tools directory to the path
tools_dir = Path(__file__).parent / "tools_utilities" / "testing_frameworks"
sys.path.insert(0, str(tools_dir))

from focused_repo_testing_agent import FocusedRepoTestingAgent

def main():
    """Main execution function"""
    print("🚀 Starting Focused Repository Testing...")
    print("🎯 Testing only core project files (excluding external packages)")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Initialize and run the focused testing agent
        agent = FocusedRepoTestingAgent()
        agent.run_comprehensive_testing()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"🎉 COMPLETE! Total execution time: {total_time:.2f} seconds")
        print(f"📁 Check results in: {agent.base_output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


