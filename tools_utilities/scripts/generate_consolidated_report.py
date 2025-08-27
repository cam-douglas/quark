#!/usr/bin/env python3
"""
📊 CONSOLIDATED TESTING REPORT GENERATOR
Purpose: Generate a single comprehensive report from completed testing sessions
Inputs: Path to test run directory (or finds latest automatically)
Outputs: Single consolidated report summarizing all tests
Seeds: 42
Dependencies: pathlib, json, datetime
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

def find_latest_test_run() -> Path:
    """Find the latest test run directory"""
    tests_dir = Path("tests/focused_repo_tests")
    if not tests_dir.exists():
        raise FileNotFoundError("No test runs found. Run tests first.")
    
    test_runs = list(tests_dir.glob("test_run_*"))
    if not test_runs:
        raise FileNotFoundError("No test runs found. Run tests first.")
    
    return max(test_runs, key=lambda x: x.stat().st_mtime)

def analyze_test_results(test_run_dir: Path) -> Dict[str, Any]:
    """Analyze test results from simulation data files"""
    simulation_dir = test_run_dir / "simulation_data"
    failed_tests_dir = test_run_dir / "failed_tests"
    
    results = {
        'total_files': 0,
        'syntax_valid': 0,
        'syntax_invalid': 0,
        'import_success': 0,
        'import_failed': 0,
        'successful_files': [],
        'failed_files': [],
        'file_categories': {},
        'complexity_stats': {
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'avg_complexity': 0
        }
    }
    
    # Analyze simulation data files
    if simulation_dir.exists():
        sim_files = list(simulation_dir.glob("*.json"))
        results['total_files'] = len(sim_files)
        
        complexities = []
        
        for sim_file in sim_files:
            try:
                with open(sim_file, 'r') as f:
                    data = json.load(f)
                
                file_name = sim_file.stem.replace('_simulation', '')
                
                # Check if this is a successful file (has metadata)
                if 'file_metadata' in data and not data.get('error'):
                    results['syntax_valid'] += 1
                    results['successful_files'].append(file_name)
                    
                    # Aggregate complexity stats
                    metadata = data['file_metadata']
                    results['complexity_stats']['total_lines'] += metadata.get('lines_of_code', 0)
                    results['complexity_stats']['total_functions'] += metadata.get('functions_count', 0)
                    results['complexity_stats']['total_classes'] += metadata.get('classes_count', 0)
                    
                    if 'complexity_metrics' in data:
                        complexity = data['complexity_metrics'].get('cyclomatic_complexity_estimate', 0)
                        if complexity > 0:
                            complexities.append(complexity)
                else:
                    results['syntax_invalid'] += 1
                    results['failed_files'].append(file_name)
                    
            except Exception as e:
                results['failed_files'].append(f"{sim_file.name} (parse error)")
        
        if complexities:
            results['complexity_stats']['avg_complexity'] = sum(complexities) / len(complexities)
    
    # Count failed tests
    if failed_tests_dir.exists():
        failed_files = list(failed_tests_dir.glob("*_error.json"))
        results['import_failed'] = len(failed_files)
    
    results['import_success'] = results['syntax_valid'] - results['import_failed']
    
    return results

def generate_consolidated_report(test_run_dir: Path, results: Dict[str, Any]) -> str:
    """Generate a single consolidated report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_run_name = test_run_dir.name
    
    # Calculate percentages
    total = results['total_files']
    syntax_pct = (results['syntax_valid'] / total * 100) if total > 0 else 0
    import_pct = (results['import_success'] / total * 100) if total > 0 else 0
    
    report = f"""# 🧪 CONSOLIDATED TESTING REPORT

**Generated:** {timestamp}  
**Test Run:** {test_run_name}  
**Test Directory:** {test_run_dir.relative_to(Path.cwd())}

## 📊 EXECUTIVE SUMMARY

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Files Tested** | {total:,} | 100.0% |
| **Syntax Valid** | {results['syntax_valid']:,} | {syntax_pct:.1f}% |
| **Import Successful** | {results['import_success']:,} | {import_pct:.1f}% |
| **Failed Tests** | {results['syntax_invalid'] + results['import_failed']:,} | {100 - syntax_pct:.1f}% |

## 📈 CODE COMPLEXITY ANALYSIS

| Metric | Total | Average per File |
|--------|-------|------------------|
| **Lines of Code** | {results['complexity_stats']['total_lines']:,} | {results['complexity_stats']['total_lines'] / max(total, 1):.1f} |
| **Functions** | {results['complexity_stats']['total_functions']:,} | {results['complexity_stats']['total_functions'] / max(total, 1):.1f} |
| **Classes** | {results['complexity_stats']['total_classes']:,} | {results['complexity_stats']['total_classes'] / max(total, 1):.1f} |
| **Avg Complexity** | {results['complexity_stats']['avg_complexity']:.1f} | - |

## ✅ TEST RESULTS OVERVIEW

- **🟢 Syntax Valid Files:** {results['syntax_valid']:,} files passed syntax validation
- **🟢 Import Successful:** {results['import_success']:,} files imported successfully  
- **🔴 Syntax Errors:** {results['syntax_invalid']:,} files have syntax issues
- **🔴 Import Errors:** {results['import_failed']:,} files failed to import

## 📁 OUTPUT STRUCTURE

All test data has been organized in the following structure:
```
{test_run_dir.name}/
├── simulation_data/     # Individual file simulation data
├── failed_tests/        # Details of failed tests
├── reports/            # This consolidated report
└── visualization_outputs/  # Charts and graphs (if generated)
```

## 🚀 NEXT STEPS

1. **Review Failed Tests:** Check `failed_tests/` directory for specific error details
2. **Fix Critical Issues:** Address syntax errors and import failures
3. **Performance Analysis:** Use simulation data for optimization insights
4. **Generate Visualizations:** Run with `--reports` flag for detailed charts

## 📋 TESTING METHODOLOGY

This consolidated report summarizes testing conducted across the entire quark repository:

- **Scope:** Core project files (excluding external packages and dependencies)
- **Tests:** Syntax validation, import testing, complexity analysis
- **Data:** Simulation data generated for performance insights
- **Output:** Consolidated summary (this report) rather than individual test files

---
*Report generated by Consolidated Testing Report Generator*  
*Timestamp: {timestamp}*
"""
    
    return report

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate consolidated testing report")
    parser.add_argument("test_run_dir", nargs="?", 
                       help="Path to test run directory (auto-finds latest if not provided)")
    
    args = parser.parse_args()
    
    try:
        # Find test run directory
        if args.test_run_dir:
            test_run_dir = Path(args.test_run_dir)
            if not test_run_dir.exists():
                print(f"❌ Test run directory not found: {test_run_dir}")
                return False
        else:
            test_run_dir = find_latest_test_run()
            print(f"🔍 Using latest test run: {test_run_dir}")
        
        print("📊 GENERATING CONSOLIDATED TESTING REPORT")
        print("=" * 50)
        
        # Analyze test results
        print("🔍 Analyzing test results...")
        results = analyze_test_results(test_run_dir)
        
        if results['total_files'] == 0:
            print("❌ No test results found to analyze")
            return False
        
        # Generate consolidated report
        print("📝 Generating consolidated report...")
        report_content = generate_consolidated_report(test_run_dir, results)
        
        # Save report
        reports_dir = test_run_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / "CONSOLIDATED_TEST_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Also save JSON summary
        summary_file = reports_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_run_directory': str(test_run_dir),
                'summary_statistics': results
            }, f, indent=2)
        
        print(f"\n🎉 CONSOLIDATED REPORT GENERATED!")
        print(f"📄 Report saved to: {report_file}")
        print(f"📊 JSON summary: {summary_file}")
        
        # Print key statistics
        print(f"\n📊 KEY STATISTICS:")
        print(f"   📁 Total files tested: {results['total_files']:,}")
        print(f"   ✅ Syntax valid: {results['syntax_valid']:,} ({results['syntax_valid']/results['total_files']*100:.1f}%)")
        print(f"   🔗 Import successful: {results['import_success']:,} ({results['import_success']/results['total_files']*100:.1f}%)")
        print(f"   📏 Total lines of code: {results['complexity_stats']['total_lines']:,}")
        print(f"   🔧 Total functions: {results['complexity_stats']['total_functions']:,}")
        print(f"   🏗️  Total classes: {results['complexity_stats']['total_classes']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

