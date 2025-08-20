#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE REPOSITORY TESTING RUNNER
Purpose: Execute the comprehensive testing agent on the entire quark repository
Inputs: None (auto-discovers all files)
Outputs: Complete test results with simulation data
Seeds: 42
Dependencies: comprehensive_repo_testing_agent
"""

import sys
import time
from pathlib import Path

# Add the tools directory to the path
tools_dir = Path(__file__).parent / "tools_utilities" / "testing_frameworks"
sys.path.insert(0, str(tools_dir))

from comprehensive_repo_testing_agent import ComprehensiveRepoTestingAgent

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Repository Testing...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize and run the testing agent
        agent = ComprehensiveRepoTestingAgent()
        agent.run_comprehensive_testing()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
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


