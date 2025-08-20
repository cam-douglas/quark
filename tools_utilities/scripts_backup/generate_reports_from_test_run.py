#!/usr/bin/env python3
"""
📊 GENERATE REPORTS FROM EXISTING TEST RUN
Purpose: Generate comprehensive reports and visualizations from existing test data
Inputs: Path to existing test run directory
Outputs: Comprehensive reports and visualizations
Seeds: 42
Dependencies: focused_repo_testing_agent
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add the tools directory to the path
tools_dir = Path(__file__).parent / "tools_utilities" / "testing_frameworks"
sys.path.insert(0, str(tools_dir))

from focused_repo_testing_agent import FocusedRepoTestingAgent

def load_existing_test_results(test_run_dir: Path) -> Dict[str, Any]:
    """Load existing test results from a test run directory"""
    test_results = {}
    test_results_dir = test_run_dir / "test_results"
    
    if not test_results_dir.exists():
        raise FileNotFoundError(f"Test results directory not found: {test_results_dir}")
    
    print(f"📂 Loading test results from: {test_results_dir}")
    
    # Load all individual test result files
    result_files = list(test_results_dir.glob("*_result.json"))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                test_results[result_data['file_path']] = result_data
        except Exception as e:
            print(f"⚠️  Warning: Could not load {result_file}: {e}")
    
    print(f"📊 Loaded {len(test_results)} test results")
    return test_results

def reconstruct_test_stats(test_results: Dict[str, Any]) -> Dict[str, int]:
    """Reconstruct test statistics from loaded results"""
    stats = {
        'total_files': len(test_results),
        'files_tested': len(test_results),
        'syntax_valid': 0,
        'syntax_invalid': 0,
        'import_success': 0,
        'import_failed': 0,
        'execution_success': 0,
        'execution_failed': 0,
        'test_files_created': 0,
        'simulation_data_generated': 0
    }
    
    for file_path, result in test_results.items():
        if 'tests' in result:
            # Count syntax results
            syntax_passed = result['tests'].get('syntax', {}).get('passed', False)
            if syntax_passed:
                stats['syntax_valid'] += 1
            else:
                stats['syntax_invalid'] += 1
            
            # Count import results
            import_passed = result['tests'].get('import', {}).get('passed', False)
            if import_passed:
                stats['import_success'] += 1
            else:
                stats['import_failed'] += 1
        
        # Count generated files
        if 'generated_test_file' in result:
            stats['test_files_created'] += 1
        
        if 'simulation_data' in result:
            stats['simulation_data_generated'] += 1
    
    return stats

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate reports from existing test run")
    parser.add_argument("test_run_dir", nargs="?", 
                       help="Path to existing test run directory")
    parser.add_argument("--latest", "-l", action="store_true",
                       help="Use the latest test run automatically")
    
    args = parser.parse_args()
    
    # Find test run directory
    if args.latest or not args.test_run_dir:
        # Find the latest test run
        tests_dir = Path("tests/focused_repo_tests")
        if not tests_dir.exists():
            print("❌ No test runs found. Run tests first with run_focused_tests_optimized.py")
            return False
        
        test_runs = list(tests_dir.glob("test_run_*"))
        if not test_runs:
            print("❌ No test runs found. Run tests first with run_focused_tests_optimized.py")
            return False
        
        # Get the latest test run
        test_run_dir = max(test_runs, key=lambda x: x.stat().st_mtime)
        print(f"🔍 Using latest test run: {test_run_dir}")
    else:
        test_run_dir = Path(args.test_run_dir)
        if not test_run_dir.exists():
            print(f"❌ Test run directory not found: {test_run_dir}")
            return False
    
    try:
        print("📊 GENERATING REPORTS FROM EXISTING TEST RUN")
        print("=" * 50)
        
        # Load existing test results
        test_results = load_existing_test_results(test_run_dir)
        
        if not test_results:
            print("❌ No test results found to generate reports from")
            return False
        
        # Create a temporary agent instance for report generation
        agent = FocusedRepoTestingAgent()
        
        # Override the agent's data with loaded results
        agent.base_output_dir = test_run_dir
        agent.test_results = test_results
        agent.test_stats = reconstruct_test_stats(test_results)
        
        # Generate reports
        print("\n📊 Generating comprehensive reports...")
        agent.generate_visualizations()
        agent.generate_comprehensive_report()
        
        print(f"\n🎉 REPORTS GENERATED SUCCESSFULLY!")
        print(f"📁 Reports saved to: {test_run_dir}")
        print(f"📄 Main report: {test_run_dir / 'reports' / 'focused_test_report.md'}")
        print(f"📈 Visualization: {test_run_dir / 'visualization_outputs' / 'focused_test_results.html'}")
        
        # Print summary
        stats = agent.test_stats
        print(f"\n📊 Summary Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Syntax valid: {stats['syntax_valid']} ({stats['syntax_valid']/stats['total_files']*100:.1f}%)")
        print(f"   Import success: {stats['import_success']} ({stats['import_success']/stats['total_files']*100:.1f}%)")
        print(f"   Test files created: {stats['test_files_created']}")
        print(f"   Simulation data generated: {stats['simulation_data_generated']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
