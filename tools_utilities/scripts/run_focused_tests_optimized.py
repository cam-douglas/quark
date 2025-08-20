#!/usr/bin/env python3
"""
🧪 OPTIMIZED FOCUSED REPOSITORY TESTING RUNNER
Purpose: Execute the focused testing agent with optional report generation
Inputs: Command line arguments to control report generation
Outputs: Test results with optional comprehensive reports
Seeds: 42
Dependencies: focused_repo_testing_agent
"""

import sys
import time
import argparse
from pathlib import Path

# Add the tools directory to the path
tools_dir = Path(__file__).parent / "tools_utilities" / "testing_frameworks"
sys.path.insert(0, str(tools_dir))

from focused_repo_testing_agent import FocusedRepoTestingAgent

def main():
    """Main execution function with command line options"""
    parser = argparse.ArgumentParser(description="Focused Repository Testing Agent")
    parser.add_argument("--reports", "-r", action="store_true", 
                       help="Generate comprehensive reports and visualizations")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick mode: testing only, no reports (default)")
    
    args = parser.parse_args()
    
    # Default to quick mode (no reports) unless explicitly requested
    generate_reports = args.reports
    
    print("🚀 Starting Optimized Focused Repository Testing...")
    print("🎯 Testing core project files (excluding external packages)")
    
    if generate_reports:
        print("📊 Comprehensive reports will be generated")
    else:
        print("⚡ Quick mode: No reports (use --reports to generate them)")
    
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Initialize and run the focused testing agent
        agent = FocusedRepoTestingAgent()
        agent.run_comprehensive_testing(generate_reports=generate_reports)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"🎉 COMPLETE! Total execution time: {total_time:.2f} seconds")
        print(f"📁 Check results in: {agent.base_output_dir}")
        
        if not generate_reports:
            print(f"\n💡 To generate reports later, run:")
            print(f"   python3 generate_reports_from_test_run.py {agent.base_output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

