#!/usr/bin/env python3
"""
Quick Environment Health Check for Small-Mind

This script provides a fast way to check if your environment is healthy
and ready for development.
"""

import sys
import os
import importlib
from pathlib import Path
from typing import List, Dict

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False

def check_virtual_environment() -> bool:
    """Check if virtual environment is active."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✅ Virtual environment: {sys.prefix}")
        return True
    else:
        print("❌ No virtual environment active")
        return False

def check_project_structure() -> List[str]:
    """Check if project structure is correct."""
    issues = []
    project_root = Path.cwd()
    
    # Check for essential directories
    essential_dirs = ["models", "src", "scripts", "docs"]
    for dir_name in essential_dirs:
        if not (project_root / dir_name).exists():
            issues.append(f"Missing directory: {dir_name}")
        else:
            print(f"✅ Directory: {dir_name}")
    
    # Check for virtual environment
    venv_locations = ["aws_env", "env", "venv", ".venv"]
    venv_found = False
    for venv_dir in venv_locations:
        if (project_root / venv_dir / "bin" / "python").exists():
            print(f"✅ Virtual environment found: {venv_dir}")
            venv_found = True
            break
    
    if not venv_found:
        issues.append("No virtual environment found")
    
    return issues

def check_imports() -> List[str]:
    """Check if essential packages can be imported."""
    issues = []
    
    # Essential packages for Small-Mind
    essential_packages = [
        "numpy", "pathlib", "json", "logging", "typing"
    ]
    
    for package in essential_packages:
        try:
            importlib.import_module(package)
            print(f"✅ Package: {package}")
        except ImportError as e:
            issues.append(f"Missing package: {package} - {e}")
    
    return issues

def check_paths() -> List[str]:
    """Check if Python paths are configured correctly."""
    issues = []
    project_root = str(Path.cwd())
    
    if project_root not in sys.path:
        issues.append("Project root not in Python path")
        print("⚠️  Project root not in Python path")
    else:
        print("✅ Project root in Python path")
    
    return issues

def run_health_check() -> Dict[str, any]:
    """Run complete health check."""
    print("🧠 Small-Mind Environment Health Check")
    print("=" * 40)
    
    results = {
        "healthy": True,
        "issues": [],
        "warnings": []
    }
    
    # Check Python version
    if not check_python_version():
        results["healthy"] = False
        results["issues"].append("Python version incompatible")
    
    # Check virtual environment
    if not check_virtual_environment():
        results["warnings"].append("No virtual environment active")
    
    # Check project structure
    structure_issues = check_project_structure()
    results["issues"].extend(structure_issues)
    
    # Check imports
    import_issues = check_imports()
    results["issues"].extend(import_issues)
    
    # Check paths
    path_issues = check_paths()
    results["issues"].extend(path_issues)
    
    # Update health status
    if results["issues"]:
        results["healthy"] = False
    
    # Print summary
    print("\n" + "=" * 40)
    if results["healthy"]:
        print("🎉 Environment is healthy!")
    else:
        print("⚠️  Environment has issues:")
        for issue in results["issues"]:
            print(f"  • {issue}")
    
    if results["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in results["warnings"]:
            print(f"  • {warning}")
    
    return results

def suggest_fixes(results: Dict[str, any]):
    """Suggest fixes for detected issues."""
    if not results["issues"]:
        return
    
    print("\n🔧 Suggested fixes:")
    print("=" * 20)
    
    for issue in results["issues"]:
        if "Python version" in issue:
            print("  • Update Python to version 3.8 or higher")
        elif "virtual environment" in issue.lower():
            print("  • Run: make env-setup")
            print("  • Or: ./scripts/smart_env.sh --fix")
        elif "Missing package" in issue:
            print("  • Run: make install-deps")
            print("  • Or: pip install -r requirements.txt")
        elif "Python path" in issue:
            print("  • Run: make env-setup")
            print("  • Or: export PYTHONPATH=$(pwd):$PYTHONPATH")
    
    print("\n💡 Quick fix command: make env-fix")

if __name__ == "__main__":
    results = run_health_check()
    suggest_fixes(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["healthy"] else 1)
