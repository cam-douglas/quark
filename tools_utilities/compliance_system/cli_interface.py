#!/usr/bin/env python3
"""
CLI interface for the compliance system.

This module handles command-line argument parsing and execution
of compliance system operations.
"""

import sys
import time
import argparse
from typing import List, Optional

# Handle both relative and absolute imports
try:
    from .core_system import QuarkComplianceSystem
except ImportError:
    from core_system import QuarkComplianceSystem


def main():
    """Main CLI interface for unified compliance system"""
    parser = argparse.ArgumentParser(description="Quark Unified Compliance System")
    parser.add_argument("--workspace", default="/Users/camdouglas/quark", help="Workspace root path")
    parser.add_argument("--start", action="store_true", help="Start compliance system")
    parser.add_argument("--stop", action="store_true", help="Stop compliance system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--check", action="store_true", help="Run immediate compliance check")
    parser.add_argument("--paths", nargs="+", help="Specific files/directories to check")
    parser.add_argument("--output", help="Output report to file")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix violations")
    parser.add_argument("--test", action="store_true", help="Run test operation")
    parser.add_argument("--operation", help="Operation name for testing")
    parser.add_argument("--background", action="store_true", help="Run in background")
    
    args = parser.parse_args()
    
    compliance_system = QuarkComplianceSystem(args.workspace)
    
    if args.start:
        compliance_system.start_system(args.background)
        if not args.background:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                compliance_system.stop_system()
    
    elif args.stop:
        compliance_system.stop_system()
    
    elif args.status:
        status = compliance_system.get_system_status()
        print("📊 Compliance System Status:")
        print(f"   Running: {status.get('running', False)}")
        print(f"   Workspace: {status.get('workspace', 'Unknown')}")
        print(f"   Last Update: {status.get('timestamp', 'Unknown')}")
        
        if 'components' in status:
            print("   Components:")
            for component, active in status['components'].items():
                status_icon = "✅" if active else "❌"
                print(f"     {status_icon} {component}")
    
    elif args.check:
        compliance_system.check_compliance_now(args.paths)
    
    elif args.test:
        operation_name = args.operation or "test_operation"
        target_files = args.paths
        
        print(f"🧪 Testing three-phase compliance for '{operation_name}'")
        
        with compliance_system.operation_context(operation_name, target_files):
            print("📝 Simulating operation...")
            time.sleep(2)  # Simulate work
            print("✅ Operation completed")
    
    else:
        # Default: run compliance check
        result = compliance_system.run_compliance_check(args.paths)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result["stdout"])
            print(f"Report saved to {args.output}")
        else:
            print(result["stdout"])
        
        if args.fix and result["returncode"] != 0:
            print("\n💡 Fix suggestions:")
            print("   - Split large files into smaller modules")
            print("   - Add missing docstrings and type hints")
            print("   - Remove prohibited patterns")
        
        # Exit with error code if violations found
        if result["returncode"] != 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
