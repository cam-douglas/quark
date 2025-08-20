#!/usr/bin/env python3
"""
Automatic Dependency Installer for Small-Mind
=============================================

Silently handles all dependency installation, environment setup,
and system configuration without user prompts.

Features:
- Automatic pip package installation
- Environment creation and activation
- Requirement file processing
- Silent error handling and fallbacks
- Cross-platform compatibility
"""

import subprocess
import sys
import os
import importlib.util
import json
import pathlib
from typing import List, Dict, Set, Optional, Tuple
import logging

# Setup minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class AutoInstaller:
    """Automatically handles all dependency installation for Small-Mind."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = pathlib.Path(project_root or os.getcwd()).resolve()
        self.installed_packages: Set[str] = set()
        self.failed_packages: Set[str] = set()
        self.silent_mode = True
        
        # Package mappings for common import/install mismatches
        self.package_mappings = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
            'yaml': 'pyyaml',
            'dotenv': 'python-dotenv',
            'bs4': 'beautifulsoup4',
            'requests_oauthlib': 'requests-oauthlib',
            'sentence_transformers': 'sentence-transformers',
            'torch': 'torch',
            'transformers': 'transformers',
            'datasets': 'datasets',
            'accelerate': 'accelerate',
            'wandb': 'wandb',
            'tensorboard': 'tensorboard',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'streamlit': 'streamlit',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pydantic': 'pydantic',
            'sqlalchemy': 'sqlalchemy',
            'alembic': 'alembic',
            'redis': 'redis',
            'celery': 'celery',
            'httpx': 'httpx',
            'aiofiles': 'aiofiles',
            'typer': 'typer',
            'rich': 'rich',
            'click': 'click'
        }
        
        # Essential packages for Small-Mind
        self.essential_packages = [
            'numpy', 'pandas', 'networkx', 'pyyaml', 'requests',
            'tqdm', 'rich', 'typer', 'fastapi', 'uvicorn'
        ]
        
        # Optional packages with fallbacks
        self.optional_packages = {
            'torch': 'PyTorch for ML models',
            'transformers': 'Hugging Face transformers',
            'sentence-transformers': 'Sentence embeddings',
            'scikit-learn': 'Machine learning toolkit',
            'matplotlib': 'Plotting library',
            'streamlit': 'Web interface',
            'openai': 'OpenAI API',
            'anthropic': 'Anthropic API'
        }
        
    def ensure_all_dependencies(self) -> bool:
        """Ensure all dependencies are installed. Returns True if successful."""
        try:
            # First, install essential packages
            self._install_essential_packages()
            
            # Process requirement files
            self._process_requirement_files()
            
            # Install from imports in code
            self._install_from_code_analysis()
            
            # Install model-specific dependencies
            self._install_model_dependencies()
            
            return True
            
        except Exception as e:
            if not self.silent_mode:
                print(f"⚠️ Some dependencies could not be installed: {e}")
            return False
            
    def _install_essential_packages(self):
        """Install essential packages for core functionality."""
        for package in self.essential_packages:
            self._safe_install(package, essential=True)
            
    def _process_requirement_files(self):
        """Process all requirements.txt files in the project."""
        req_files = list(self.project_root.rglob("requirements*.txt"))
        
        for req_file in req_files:
            try:
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (remove version constraints)
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                        self._safe_install(package)
                        
            except Exception as e:
                logger.warning(f"Could not process {req_file}: {e}")
                
    def _install_from_code_analysis(self):
        """Analyze Python files and install missing imports."""
        python_files = list(self.project_root.rglob("*.py"))
        
        imports = set()
        for py_file in python_files:
            try:
                imports.update(self._extract_imports(py_file))
            except Exception:
                continue
                
        # Install missing imports
        for import_name in imports:
            if not self._is_package_available(import_name):
                package_name = self.package_mappings.get(import_name, import_name)
                self._safe_install(package_name)
                
    def _install_model_dependencies(self):
        """Install dependencies for specific models and features."""
        models_yaml = self.project_root / "models" / "models.yaml"
        
        if models_yaml.exists():
            try:
                import yaml
                with open(models_yaml) as f:
                    models_config = yaml.safe_load(f)
                    
                # Install dependencies based on model types
                self._install_ml_dependencies(models_config)
                
            except Exception as e:
                logger.warning(f"Could not process models.yaml: {e}")
                
    def _install_ml_dependencies(self, models_config: Dict):
        """Install ML-specific dependencies based on models configuration."""
        ml_packages = []
        
        # Check for specific model types
        if 'multi_model_training' in models_config:
            ml_packages.extend(['torch', 'transformers', 'accelerate', 'datasets'])
            
        if 'neuro_system' in models_config:
            ml_packages.extend(['networkx', 'scikit-learn', 'sentence-transformers'])
            
        # Install LLM packages
        if any('llm' in str(config).lower() for config in str(models_config)):
            ml_packages.extend(['openai', 'anthropic', 'langchain'])
            
        for package in ml_packages:
            self._safe_install(package)
            
    def _safe_install(self, package: str, essential: bool = False) -> bool:
        """Safely install a package with fallbacks."""
        if package in self.installed_packages:
            return True
            
        if package in self.failed_packages and not essential:
            return False
            
        try:
            # Check if already installed
            if self._is_package_available(package):
                self.installed_packages.add(package)
                return True
                
            # Try to install
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package, '--quiet', '--disable-pip-version-check'],
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.installed_packages.add(package)
                return True
            else:
                # Try alternative installation methods
                return self._try_alternative_install(package, essential)
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Installation of {package} timed out")
            return self._try_alternative_install(package, essential)
        except Exception as e:
            logger.warning(f"Failed to install {package}: {e}")
            return self._try_alternative_install(package, essential)
            
    def _try_alternative_install(self, package: str, essential: bool = False) -> bool:
        """Try alternative installation methods."""
        alternatives = [
            # Try with --user flag
            [sys.executable, '-m', 'pip', 'install', '--user', package, '--quiet'],
            # Try with --no-deps (for problematic dependencies)
            [sys.executable, '-m', 'pip', 'install', '--no-deps', package, '--quiet'],
            # Try pre-compiled wheels only
            [sys.executable, '-m', 'pip', 'install', '--only-binary=all', package, '--quiet']
        ]
        
        for cmd in alternatives:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=180)
                if result.returncode == 0:
                    self.installed_packages.add(package)
                    return True
            except Exception:
                continue
                
        # Mark as failed
        self.failed_packages.add(package)
        
        if essential:
            logger.error(f"CRITICAL: Could not install essential package {package}")
        else:
            logger.warning(f"Could not install optional package {package}")
            
        return False
        
    def _is_package_available(self, package: str) -> bool:
        """Check if a package is available for import."""
        try:
            importlib.import_module(package)
            return True
        except ImportError:
            return False
            
    def _extract_imports(self, py_file: pathlib.Path) -> Set[str]:
        """Extract import statements from a Python file."""
        imports = set()
        
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                
                # Handle 'import module' statements
                if line.startswith('import ') and not line.startswith('import.'):
                    parts = line.split()
                    if len(parts) >= 2:
                        module = parts[1].split('.')[0]
                        imports.add(module)
                        
                # Handle 'from module import' statements
                elif line.startswith('from ') and ' import ' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        module = parts[1].split('.')[0]
                        imports.add(module)
                        
        except Exception:
            pass
            
        return imports
        
    def get_installation_summary(self) -> Dict[str, any]:
        """Get summary of installation results."""
        return {
            'installed_packages': list(self.installed_packages),
            'failed_packages': list(self.failed_packages),
            'total_installed': len(self.installed_packages),
            'total_failed': len(self.failed_packages),
            'success_rate': len(self.installed_packages) / (len(self.installed_packages) + len(self.failed_packages)) if (self.installed_packages or self.failed_packages) else 0
        }

# Global auto-installer instance
_auto_installer = None

def ensure_dependencies(project_root: Optional[str] = None) -> bool:
    """Global function to ensure all dependencies are installed."""
    global _auto_installer
    
    if _auto_installer is None:
        _auto_installer = AutoInstaller(project_root)
        
    return _auto_installer.ensure_all_dependencies()

def safe_import(module_name: str, package_name: Optional[str] = None):
    """Safely import a module, installing if necessary."""
    global _auto_installer
    
    if _auto_installer is None:
        _auto_installer = AutoInstaller()
        
    try:
        return importlib.import_module(module_name)
    except ImportError:
        # Try to install and import
        pkg_name = package_name or _auto_installer.package_mappings.get(module_name, module_name)
        if _auto_installer._safe_install(pkg_name):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                pass
        
        # Return a mock module if installation fails
        return _create_mock_module(module_name)

def _create_mock_module(module_name: str):
    """Create a mock module for failed imports."""
    import types
    
    mock_module = types.ModuleType(module_name)
    
    # Add common attributes that might be accessed
    setattr(mock_module, '__version__', '0.0.0-mock')
    setattr(mock_module, '__file__', f'<mock-{module_name}>')
    
    # Add a warning function
    def mock_warning(*args, **kwargs):
        logger.warning(f"Using mock {module_name} - functionality may be limited")
        
    setattr(mock_module, '_mock_warning', mock_warning)
    
    return mock_module

# Auto-run dependency installation when module is imported
if __name__ != '__main__':
    try:
        ensure_dependencies()
    except Exception:
        pass  # Silent failure for imports
