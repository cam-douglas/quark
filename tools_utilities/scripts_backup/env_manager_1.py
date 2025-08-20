#!/usr/bin/env python3
"""
Intelligent Environment Manager for Small-Mind

This script automatically detects and fixes common environment issues:
- Missing dependencies
- Path problems
- Import errors
- Virtual environment issues
- Package conflicts
"""

import os, sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentManager:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.venv_path = self._find_venv()
        self.issues = []
        self.fixes_applied = []
        
    def _find_venv(self) -> Optional[Path]:
        """Find virtual environment in common locations."""
        possible_paths = [
            self.project_root / "aws_env",
            self.project_root / "env",
            self.project_root / "venv",
            self.project_root / ".venv",
            Path.home() / ".virtualenvs" / self.project_root.name
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "bin" / "python").exists():
                return path
        return None
    
    def diagnose_environment(self) -> Dict[str, List[str]]:
        """Run comprehensive environment diagnostics."""
        logger.info("🔍 Running environment diagnostics...")
        
        diagnostics = {
            "critical": [],
            "warnings": [],
            "info": [],
            "suggestions": []
        }
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            diagnostics["critical"].append(f"Python version {python_version.major}.{python_version.minor} is too old. Need Python 3.8+")
        else:
            diagnostics["info"].append(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
        
        # Check virtual environment
        if not self.venv_path:
            diagnostics["critical"].append("No virtual environment detected")
        else:
            diagnostics["info"].append(f"✅ Virtual environment found at: {self.venv_path}")
            
            # Check if we're actually in the venv
            if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                diagnostics["warnings"].append("Not currently in virtual environment")
        
        # Check common import issues
        import_issues = self._check_imports()
        diagnostics["critical"].extend(import_issues)
        
        # Check path issues
        path_issues = self._check_paths()
        diagnostics["warnings"].extend(path_issues)
        
        # Check package conflicts
        conflict_issues = self._check_package_conflicts()
        diagnostics["warnings"].extend(conflict_issues)
        
        # Check file permissions
        perm_issues = self._check_permissions()
        diagnostics["warnings"].extend(perm_issues)
        
        return diagnostics
    
    def _check_imports(self) -> List[str]:
        """Check for common import issues."""
        issues = []
        
        # Check if we can import common packages
        common_packages = [
            "numpy", "pandas", "torch", "transformers", "requests", "pathlib"
        ]
        
        for package in common_packages:
            try:
                importlib.import_module(package)
            except ImportError as e:
                issues.append(f"Missing package: {package} - {str(e)}")
        
        return issues
    
    def _check_paths(self) -> List[str]:
        """Check for path-related issues."""
        issues = []
        
        # Check if project root is in Python path
        if str(self.project_root) not in sys.path:
            issues.append("Project root not in Python path")
        
        # Check for relative import issues
        if self.venv_path:
            venv_lib = self.venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
            if venv_lib.exists() and str(venv_lib) not in sys.path:
                issues.append("Virtual environment lib not in Python path")
        
        return issues
    
    def _check_package_conflicts(self) -> List[str]:
        """Check for package conflicts."""
        issues = []
        
        # This would require more sophisticated analysis
        # For now, just check if we can run pip
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                issues.append(f"Pip list failed: {result.stderr}")
        except Exception as e:
            issues.append(f"Could not check packages: {str(e)}")
        
        return issues
    
    def _check_permissions(self) -> List[str]:
        """Check for permission issues."""
        issues = []
        
        if self.venv_path:
            # Check if we can write to venv
            test_file = self.venv_path / "test_write.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception:
                issues.append("Cannot write to virtual environment directory")
        
        return issues
    
    def auto_fix_issues(self, diagnostics: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Automatically fix common issues."""
        logger.info("🔧 Applying automatic fixes...")
        
        fixes_applied = []
        
        # Fix path issues
        if "Project root not in Python path" in diagnostics["warnings"]:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
                fixes_applied.append("Added project root to Python path")
        
        # Fix virtual environment activation
        if self.venv_path and "Not currently in virtual environment" in diagnostics["warnings"]:
            # Update environment variables
            os.environ["VIRTUAL_ENV"] = str(self.venv_path)
            os.environ["PATH"] = f"{self.venv_path}/bin:{os.environ.get('PATH', '')}"
            fixes_applied.append("Updated environment variables for virtual environment")
        
        # Install missing packages
        missing_packages = [issue.split(": ")[1].split(" - ")[0] for issue in diagnostics["critical"] 
                           if "Missing package:" in issue]
        
        if missing_packages:
            logger.info(f"Installing missing packages: {missing_packages}")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, 
                              check=True, timeout=300)
                fixes_applied.append(f"Installed missing packages: {missing_packages}")
            except subprocess.TimeoutExpired:
                fixes_applied.append("Package installation timed out")
            except subprocess.CalledProcessError as e:
                fixes_applied.append(f"Failed to install packages: {e}")
        
        return fixes_applied
    
    def create_environment_script(self) -> str:
        """Create a script to properly activate the environment."""
        script_content = f"""#!/bin/bash
# Auto-generated environment activation script
# Generated by env_manager.py

export PROJECT_ROOT="{self.project_root.absolute()}"
export VIRTUAL_ENV="{self.venv_path.absolute() if self.venv_path else ''}"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Activate virtual environment if it exists
if [ -f "$VIRTUAL_ENV/bin/activate" ]; then
    source "$VIRTUAL_ENV/bin/activate"
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment not found at: $VIRTUAL_ENV"
fi

# Add project scripts to PATH
export PATH="$PROJECT_ROOT/scripts:$PATH"

echo "🚀 Environment ready for Small-Mind development"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
"""
        
        script_path = self.project_root / "activate_env.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        return str(script_path)
    
    def generate_requirements(self) -> str:
        """Generate comprehensive requirements file."""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                                  capture_output=True, text=True, check=True)
            
            requirements_path = self.project_root / "requirements.txt"
            requirements_path.write_text(result.stdout)
            
            return str(requirements_path)
        except Exception as e:
            logger.error(f"Failed to generate requirements: {e}")
            return None

def main():
    """Main diagnostic and fix routine."""
    print("🧠 Small-Mind Environment Manager")
    print("=" * 50)
    
    manager = EnvironmentManager()
    
    # Run diagnostics
    diagnostics = manager.diagnose_environment()
    
    # Print results
    for severity, issues in diagnostics.items():
        if issues:
            print(f"\n{severity.upper()}:")
            for issue in issues:
                print(f"  • {issue}")
    
    # Auto-fix issues
    if any(diagnostics.values()):
        print(f"\n🔧 Applying automatic fixes...")
        fixes = manager.auto_fix_issues(diagnostics)
        
        if fixes:
            print("Applied fixes:")
            for fix in fixes:
                print(f"  ✅ {fix}")
        else:
            print("No automatic fixes available")
    
    # Create environment script
    script_path = manager.create_environment_script()
    print(f"\n📝 Created environment script: {script_path}")
    
    # Generate requirements
    req_path = manager.generate_requirements()
    if req_path:
        print(f"📦 Generated requirements: {req_path}")
    
    print(f"\n💡 To activate environment, run: source {script_path}")
    print("💡 For manual fixes, check the diagnostics above")

if __name__ == "__main__":
    main()
