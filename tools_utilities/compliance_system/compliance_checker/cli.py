"""
CLI Interface for Compliance Checker

Command-line interface for the rule compliance checker.

Author: Quark AI
Date: 2025-01-27
"""

import sys
import argparse
from typing import Dict, Any

# Handle both relative and absolute imports
try:
    from .core_checker import RuleComplianceChecker
except ImportError:
    from core_checker import RuleComplianceChecker


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Check Quark project rule compliance")
    parser.add_argument("--paths", nargs="+", help="Specific files/directories to check")
    parser.add_argument("--output", help="Output report to file")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix violations")
    parser.add_argument("--workspace", default="/Users/camdouglas/quark", help="Workspace root path")
    
    args = parser.parse_args()
    
    checker = RuleComplianceChecker(args.workspace)
    report = checker.check_workspace_compliance(args.paths)
    
    # Generate and display report
    report_text = checker.generate_report(report, args.output)
    print(report_text)
    
    # Attempt fixes if requested
    if args.fix and report.violations:
        print("\n" + "="*50)
        print("ATTEMPTING FIXES...")
        fix_results = checker.fix_violations(report.violations)
        
        if fix_results["applied"]:
            print(f"✅ Applied {len(fix_results['applied'])} fixes")
        if fix_results["failed"]:
            print(f"❌ Failed to fix {len(fix_results['failed'])} violations")
            for failed in fix_results["failed"]:
                print(f"  - {failed['violation'].file_path}: {failed['reason']}")
    
    # Exit with error code if violations found
    if not report.compliant:
        sys.exit(1)


if __name__ == "__main__":
    main()
