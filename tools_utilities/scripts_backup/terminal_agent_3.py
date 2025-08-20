#!/usr/bin/env python3
"""
Terminal Agent for Small-Mind Project
=====================================

Intelligent terminal assistant that provides:
- Automatic dependency detection and installation suggestions
- Version management and compatibility checking  
- Virtual environment creation and styling
- Smart auto-activation based on project context
- Enhanced terminal UI with colors and symbols

Features:
- 🔍 Dependency scanner (analyzes imports and suggests installations)
- 📦 Version checker (detects outdated packages and suggests updates)
- 🐍 Smart env manager (creates/activates appropriate environments)
- 🎨 Enhanced terminal styling with environment indicators
- 🧠 Context-aware suggestions based on current directory/project
"""

import os, sys
import json
import subprocess
import pathlib
import re
import ast
import importlib.util
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class DependencyInfo:
    name: str
    version: Optional[str] = None
    source: str = "import"  # import, requirements, setup.py, etc.
    installed: bool = False
    latest_version: Optional[str] = None
    suggested_install: Optional[str] = None

@dataclass 
class EnvironmentInfo:
    name: str
    path: pathlib.Path
    python_version: str
    packages: List[str]
    is_active: bool = False
    project_type: Optional[str] = None  # ml, web, data, general

class TerminalAgent:
    """Intelligent terminal agent for dependency and environment management."""
    
    def __init__(self, project_root: pathlib.Path = None):
        self.project_root = project_root or pathlib.Path.cwd()
        self.cache_dir = self.project_root / ".neuro" / "terminal_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment detection patterns
        self.env_patterns = {
            "ml": ["torch", "tensorflow", "sklearn", "pandas", "numpy", "jupyter"],
            "web": ["flask", "django", "fastapi", "requests", "jinja2"],
            "data": ["pandas", "numpy", "matplotlib", "seaborn", "plotly"],
            "neuro": ["networkx", "scipy", "scikit-learn", "brain", "neural"],
            "blockchain": ["web3", "ethereum", "solidity", "crypto"],
            "cloud": ["boto3", "azure", "google-cloud", "docker"]
        }
    
    def scan_dependencies(self, scan_path: pathlib.Path = None) -> List[DependencyInfo]:
        """Scan project for dependencies and their status."""
        scan_path = scan_path or self.project_root
        deps = {}
        
        # Scan Python files for imports
        for py_file in scan_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            try:
                deps.update(self._extract_imports_from_file(py_file))
            except Exception as e:
                print(f"⚠️  Error scanning {py_file}: {e}")
        
        # Scan requirements files
        for req_file in ["requirements.txt", "requirements-dev.txt", "setup.py", "pyproject.toml", "environment.yml"]:
            req_path = scan_path / req_file
            if req_path.exists():
                deps.update(self._extract_from_requirements(req_path))
        
        # Check installation status and versions
        for dep in deps.values():
            self._check_dependency_status(dep)
        
        return list(deps.values())
    
    def _should_skip_file(self, file_path: pathlib.Path) -> bool:
        """Check if file should be skipped during scanning."""
        skip_patterns = [
            "/.venv/", "/env/", "/node_modules/", "/.git/", 
            "/build/", "/dist/", "/__pycache__/", "/logs/",
            "/aws_env/", "/conda/", "/.cursor/"
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _extract_imports_from_file(self, py_file: pathlib.Path) -> Dict[str, DependencyInfo]:
        """Extract import statements from Python file."""
        deps = {}
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name.split('.')[0]
                        deps[name] = DependencyInfo(name=name, source=f"import:{py_file.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        name = node.module.split('.')[0]
                        deps[name] = DependencyInfo(name=name, source=f"from_import:{py_file.name}")
        
        except (SyntaxError, UnicodeDecodeError):
            # Try regex fallback for problematic files
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Basic regex patterns for imports
                import_patterns = [
                    r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
                ]
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        name = match.group(1)
                        deps[name] = DependencyInfo(name=name, source=f"regex:{py_file.name}")
            except:
                pass
        
        return deps
    
    def _extract_from_requirements(self, req_file: pathlib.Path) -> Dict[str, DependencyInfo]:
        """Extract dependencies from requirements/setup files."""
        deps = {}
        
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if req_file.name == "requirements.txt" or "requirements" in req_file.name:
                # Parse requirements.txt format
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle various formats: package==1.0, package>=1.0, package
                        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_-]*)', line)
                        if match:
                            name = match.group(1).lower()
                            version_match = re.search(r'[=<>!]+([0-9.]+)', line)
                            version = version_match.group(1) if version_match else None
                            deps[name] = DependencyInfo(
                                name=name, 
                                version=version, 
                                source=f"requirements:{req_file.name}"
                            )
            
            elif req_file.name == "setup.py":
                # Extract install_requires from setup.py
                install_requires_match = re.search(
                    r'install_requires\s*=\s*\[(.*?)\]', 
                    content, 
                    re.DOTALL
                )
                if install_requires_match:
                    requirements_str = install_requires_match.group(1)
                    for req in re.findall(r'["\']([^"\']+)["\']', requirements_str):
                        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_-]*)', req)
                        if match:
                            name = match.group(1).lower()
                            deps[name] = DependencyInfo(name=name, source="setup.py")
        
        except Exception as e:
            print(f"⚠️  Error parsing {req_file}: {e}")
        
        return deps
    
    def _check_dependency_status(self, dep: DependencyInfo):
        """Check if dependency is installed and get version info."""
        try:
            # Check if module can be imported
            spec = importlib.util.find_spec(dep.name)
            if spec is not None:
                dep.installed = True
                
                # Try to get version
                try:
                    module = importlib.import_module(dep.name)
                    if hasattr(module, '__version__'):
                        dep.version = module.__version__
                    elif hasattr(module, 'VERSION'):
                        dep.version = str(module.VERSION)
                except:
                    pass
                
                # Check latest version with pip (in background, non-blocking)
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "show", dep.name],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        version_match = re.search(r'Version: (.+)', result.stdout)
                        if version_match:
                            dep.version = version_match.group(1)
                except:
                    pass
            else:
                dep.installed = False
                dep.suggested_install = self._suggest_install_command(dep.name)
        
        except Exception:
            dep.installed = False
            dep.suggested_install = self._suggest_install_command(dep.name)
    
    def _suggest_install_command(self, package_name: str) -> str:
        """Suggest appropriate installation command for package."""
        # Common package name mappings
        package_mapping = {
            "cv2": "opencv-python",
            "PIL": "Pillow", 
            "sklearn": "scikit-learn",
            "yaml": "pyyaml",
            "dotenv": "python-dotenv",
            "bs4": "beautifulsoup4",
            "requests_oauthlib": "requests-oauthlib"
        }
        
        # Packages that commonly have build issues
        problematic_packages = {
            "pybullet": "pip install --only-binary=all pybullet || conda install -c conda-forge pybullet",
            "tensorflow": "pip install tensorflow-macos tensorflow-metal  # For Apple Silicon",
            "torch": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "mujoco": "pip install mujoco-py || conda install -c conda-forge mujoco-py",
            "gym": "pip install gymnasium  # Modern replacement for gym",
            "Box2D": "conda install -c conda-forge box2d-py || brew install swig && pip install box2d-py"
        }
        
        actual_name = package_mapping.get(package_name, package_name)
        
        # Return special command for problematic packages
        if actual_name in problematic_packages:
            return problematic_packages[actual_name]
        
        return f"pip install {actual_name}"
    
    def detect_project_type(self, deps: List[DependencyInfo]) -> str:
        """Detect project type based on dependencies."""
        dep_names = [dep.name.lower() for dep in deps]
        
        scores = {}
        for proj_type, keywords in self.env_patterns.items():
            score = sum(1 for keyword in keywords if any(keyword in dep for dep in dep_names))
            if score > 0:
                scores[proj_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    def suggest_environment_setup(self, project_type: str = None) -> Dict:
        """Suggest virtual environment setup based on project."""
        if not project_type:
            deps = self.scan_dependencies()
            project_type = self.detect_project_type(deps)
        
        env_name = f"{self.project_root.name}_{project_type}"
        env_path = self.project_root / "env"
        
        # Suggest Python version based on project type
        python_versions = {
            "ml": "3.9",      # Good for most ML libraries
            "web": "3.11",    # Latest stable for web apps  
            "data": "3.10",   # Good balance for data science
            "neuro": "3.9",   # Compatible with scientific libraries
            "blockchain": "3.10",
            "cloud": "3.11",
            "general": "3.11"
        }
        
        suggested_python = python_versions.get(project_type, "3.11")
        
        setup_commands = [
            f"python{suggested_python} -m venv env",
            "source env/bin/activate" if os.name != 'nt' else "env\\Scripts\\activate",
            "pip install --upgrade pip"
        ]
        
        # Add project-specific packages
        if project_type in self.env_patterns:
            core_packages = self.env_patterns[project_type][:3]  # Top 3 most important
            setup_commands.append(f"pip install {' '.join(core_packages)}")
        
        return {
            "env_name": env_name,
            "env_path": str(env_path),
            "project_type": project_type,
            "python_version": suggested_python,
            "setup_commands": setup_commands,
            "activation_script": self._generate_activation_script(env_path, project_type)
        }
    
    def _generate_activation_script(self, env_path: pathlib.Path, project_type: str) -> str:
        """Generate enhanced activation script with styling."""
        activate_path = env_path / "bin" / "activate"
        
        # Color codes for different project types
        colors = {
            "ml": "🧠 \\033[1;35m",        # Bright magenta
            "web": "🌐 \\033[1;36m",       # Bright cyan  
            "data": "📊 \\033[1;33m",      # Bright yellow
            "neuro": "🧙 \\033[1;95m",     # Bright magenta
            "blockchain": "⛓️ \\033[1;32m", # Bright green
            "cloud": "☁️ \\033[1;34m",     # Bright blue
            "general": "🐍 \\033[1;37m"    # Bright white
        }
        
        symbol, color = colors.get(project_type, colors["general"]).split(" ", 1)
        
        script = f'''#!/bin/bash
# Enhanced activation script for {project_type} environment

# Source the standard activation
source "{activate_path}"

# Set enhanced prompt with color and symbol
export PS1="{symbol} ({color}{env_path.name}\\033[0m) $PS1"

# Set environment variables
export PROJECT_TYPE="{project_type}"
export NEURO_ENV_ACTIVE="true"
export SMALLMIND_ROOT="{self.project_root}"

# Add project-specific aliases and functions
alias pip-deps="python -m neuro.terminal_agent check-deps"
alias env-info="python -m neuro.terminal_agent env-status"
alias smart-install="python -m neuro.terminal_agent smart-install"

echo "{symbol} Activated {project_type} environment: {env_path.name}"
echo "🔧 Type 'pip-deps' to check dependencies"
echo "📋 Type 'env-info' for environment status"
'''
        return script
    
    def generate_dependency_report(self) -> Dict:
        """Generate comprehensive dependency report."""
        deps = self.scan_dependencies()
        project_type = self.detect_project_type(deps)
        
        missing = [dep for dep in deps if not dep.installed]
        installed = [dep for dep in deps if dep.installed]
        
        return {
            "project_root": str(self.project_root),
            "project_type": project_type,
            "total_dependencies": len(deps),
            "installed": len(installed),
            "missing": len(missing),
            "missing_deps": [
                {
                    "name": dep.name,
                    "source": dep.source,
                    "install_command": dep.suggested_install
                } for dep in missing
            ],
            "installed_deps": [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "source": dep.source
                } for dep in installed
            ],
            "quick_install_all": f"pip install {' '.join(dep.suggested_install.split()[-1] for dep in missing)}" if missing else None
        }

def main():
    """CLI interface for terminal agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Small-Mind Terminal Agent")
    parser.add_argument("command", choices=[
        "scan", "check-deps", "env-status", "suggest-env", "smart-install", "report"
    ])
    parser.add_argument("--path", type=pathlib.Path, help="Project path to analyze")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    agent = TerminalAgent(args.path)
    
    if args.command == "scan":
        deps = agent.scan_dependencies()
        if args.json:
            print(json.dumps([{
                "name": dep.name,
                "version": dep.version,
                "installed": dep.installed,
                "source": dep.source
            } for dep in deps], indent=2))
        else:
            print(f"🔍 Found {len(deps)} dependencies:")
            for dep in deps:
                status = "✅" if dep.installed else "❌"
                version = f" v{dep.version}" if dep.version else ""
                print(f"  {status} {dep.name}{version} ({dep.source})")
    
    elif args.command == "check-deps":
        report = agent.generate_dependency_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"📊 Dependency Report for {report['project_type']} project")
            print(f"   📁 {report['project_root']}")
            print(f"   📦 {report['installed']}/{report['total_dependencies']} dependencies installed")
            
            if report['missing_deps']:
                print(f"\n❌ Missing dependencies ({len(report['missing_deps'])}):")
                for dep in report['missing_deps']:
                    print(f"   • {dep['name']} - {dep['install_command']}")
                
                if report['quick_install_all']:
                    print(f"\n🚀 Quick install all: {report['quick_install_all']}")
            else:
                print("\n✅ All dependencies satisfied!")
    
    elif args.command == "suggest-env":
        suggestion = agent.suggest_environment_setup()
        if args.json:
            print(json.dumps(suggestion, indent=2))
        else:
            print(f"🐍 Environment Setup for {suggestion['project_type']} project:")
            print(f"   📁 Path: {suggestion['env_path']}")
            print(f"   🔗 Python: {suggestion['python_version']}")
            print(f"\n🛠️  Setup commands:")
            for cmd in suggestion['setup_commands']:
                print(f"   {cmd}")

if __name__ == "__main__":
    main()
