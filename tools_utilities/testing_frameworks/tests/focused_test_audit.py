#!/usr/bin/env python3
"""
FOCUSED TEST AUDIT: Check project Python files for corresponding tests
Purpose: Audit actual project Python files and ensure they have visual tests
Inputs: Project Python files (excluding venv, site-packages, etc.)
Outputs: Focused audit report with missing tests
Seeds: 42
Dependencies: pathlib, os, sys
"""

import os, sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

class FocusedTestAuditor:
    """Focused audit of project Python files for corresponding tests"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.test_results = {}
        
    def find_project_python_files(self) -> List[Path]:
        """Find Python files in the actual project (excluding venv, etc.)"""
        python_files = []
        
        # Use find command to get project files only
        import subprocess
        
        try:
            result = subprocess.run([
                'find', '.', '-maxdepth', '3', '-name', '*.py', '-type', 'f'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and not any(exclude in line for exclude in [
                        '__pycache__', 'venv', 'site-packages', '.git', '.cursor'
                    ]):
                        python_files.append(Path(line))
        except Exception as e:
            print(f"⚠️  Error using find command: {e}")
            # Fallback to manual search
            python_files = self._manual_search()
        
        return python_files
    
    def _manual_search(self) -> List[Path]:
        """Manual search for Python files"""
        python_files = []
        
        # Define project directories to search
        project_dirs = ['src', 'tests', 'scripts', '.']
        
        for project_dir in project_dirs:
            dir_path = self.project_root / project_dir
            if dir_path.exists():
                for root, dirs, files in os.walk(dir_path):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if d not in {
                        '__pycache__', 'venv', 'env', '.venv', '.env',
                        'node_modules', '.git', '.vscode', '.idea',
                        'site-packages', 'dist', 'build', '*.egg-info'
                    }]
                    
                    for file in files:
                        if file.endswith('.py'):
                            file_path = Path(root) / file
                            python_files.append(file_path)
        
        return python_files
    
    def find_project_test_files(self) -> List[Path]:
        """Find test files in the project"""
        test_files = []
        
        # Search in tests directories
        test_dirs = ['tests', 'src/core/tests', 'src/training/tests', 'src/config/tests', 'scripts/debug/tests']
        
        for test_dir in test_dirs:
            dir_path = self.project_root / test_dir
            if dir_path.exists():
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = Path(root) / file
                            test_files.append(file_path)
        
        return test_files
    
    def get_component_name(self, file_path: Path) -> str:
        """Extract component name from file path"""
        # Remove .py extension
        name = file_path.stem
        
        # Handle special cases
        if name == '__init__':
            # Use parent directory name
            return file_path.parent.name
        elif name.startswith('test_'):
            # Remove test_ prefix
            return name[5:]
        elif name.endswith('_test'):
            # Remove _test suffix
            return name[:-5]
        
        return name
    
    def find_corresponding_test(self, component_name: str, test_files: List[Path]) -> Path:
        """Find corresponding test file for a component"""
        possible_test_names = [
            f"test_{component_name}.py",
            f"{component_name}_test.py",
            f"test_{component_name}_test.py"
        ]
        
        for test_file in test_files:
            if test_file.name in possible_test_names:
                return test_file
        
        return None
    
    def audit_project_tests(self) -> Dict:
        """Audit project Python files for tests"""
        print("🔍 Starting focused test audit...")
        
        # Find project Python files
        python_files = self.find_project_python_files()
        print(f"📁 Found {len(python_files)} project Python files")
        
        # Find project test files
        test_files = self.find_project_test_files()
        print(f"🧪 Found {len(test_files)} project test files")
        
        # Audit each Python file
        audit_results = {
            'total_files': len(python_files),
            'files_with_tests': 0,
            'files_without_tests': 0,
            'test_coverage': 0.0,
            'missing_tests': [],
            'existing_tests': [],
            'test_locations': {},
            'file_details': []
        }
        
        for py_file in python_files:
            component_name = self.get_component_name(py_file)
            corresponding_test = self.find_corresponding_test(component_name, test_files)
            
            file_detail = {
                'path': str(py_file),
                'component_name': component_name,
                'has_test': corresponding_test is not None,
                'test_path': str(corresponding_test) if corresponding_test else None,
                'suggested_test_path': self.suggest_test_path(py_file)
            }
            
            audit_results['file_details'].append(file_detail)
            
            if corresponding_test:
                audit_results['files_with_tests'] += 1
                audit_results['existing_tests'].append(file_detail)
                audit_results['test_locations'][str(py_file)] = str(corresponding_test)
            else:
                audit_results['files_without_tests'] += 1
                audit_results['missing_tests'].append(file_detail)
        
        # Calculate test coverage
        if audit_results['total_files'] > 0:
            audit_results['test_coverage'] = audit_results['files_with_tests'] / audit_results['total_files']
        
        return audit_results
    
    def suggest_test_path(self, py_file: Path) -> str:
        """Suggest where the test file should be located"""
        # If file is in src/core/, suggest src/core/tests/
        if 'src/core' in str(py_file):
            return str(py_file.parent / 'tests' / f"test_{py_file.stem}.py")
        elif 'src/training' in str(py_file):
            return str(py_file.parent / 'tests' / f"test_{py_file.stem}.py")
        elif 'src/config' in str(py_file):
            return str(py_file.parent / 'tests' / f"test_{py_file.stem}.py")
        elif 'scripts' in str(py_file):
            return str(py_file.parent / 'tests' / f"test_{py_file.stem}.py")
        else:
            return str(Path('tests') / f"test_{py_file.stem}.py")
    
    def create_focused_report(self, audit_results: Dict) -> str:
        """Create a focused audit report"""
        print("📋 Creating focused audit report...")
        
        report = f"""
# 🧪 FOCUSED TEST AUDIT REPORT

## 📊 Summary
- **Total Project Python Files**: {audit_results['total_files']}
- **Files with Tests**: {audit_results['files_with_tests']}
- **Files without Tests**: {audit_results['files_without_tests']}
- **Test Coverage**: {audit_results['test_coverage']:.1%}

## ✅ Files with Tests ({len(audit_results['existing_tests'])})
"""
        
        for item in audit_results['existing_tests']:
            report += f"- **{item['component_name']}**: `{item['path']}` → `{item['test_path']}`\n"
        
        report += f"""
## ❌ Missing Tests ({len(audit_results['missing_tests'])})
"""
        
        for item in audit_results['missing_tests']:
            report += f"- **{item['component_name']}**: `{item['path']}`\n"
            report += f"  - Suggested: `{item['suggested_test_path']}`\n"
        
        report += f"""
## 📈 Priority Recommendations

### 🔴 High Priority (Core Brain Components)
"""
        
        # Prioritize core components
        core_components = ['brain_launcher', 'neural_components', 'developmental_timeline', 
                          'training_orchestrator', 'sleep_consolidation', 'multi_scale_integration',
                          'capacity_progression', 'rules_loader', 'biological_validator']
        
        for component in core_components:
            missing = [item for item in audit_results['missing_tests'] if component in item['component_name'].lower()]
            if missing:
                report += f"- Create test for **{component}**\n"
        
        report += f"""
### 🟡 Medium Priority (Supporting Components)
"""
        
        supporting_components = ['connectome', 'config', 'training', 'debug']
        
        for component in supporting_components:
            missing = [item for item in audit_results['missing_tests'] if component in item['component_name'].lower()]
            if missing:
                report += f"- Create test for **{component}**\n"
        
        report += f"""
### 🟢 Low Priority
- `__init__.py` files (usually don't need tests)
- Simple script files
- Demo files

## 🚀 Next Steps
1. Run the comprehensive test runner: `python tests/comprehensive_test_runner.py`
2. Create missing tests for high-priority components
3. Ensure all tests include visual validation
4. Update test coverage regularly

## 📁 Detailed File List
"""
        
        for item in audit_results['file_details']:
            status = "✅" if item['has_test'] else "❌"
            report += f"{status} `{item['path']}` ({item['component_name']})\n"
        
        return report
    
    def save_focused_report(self, report: str, filename: str = "focused_test_audit_report.md"):
        """Save focused audit report to file"""
        output_path = Path("tests") / filename
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Focused audit report saved to {output_path}")
        return output_path
    
    def run_focused_audit(self):
        """Run the focused audit"""
        print("🚀 Starting Focused Test Audit...")
        
        # Perform audit
        audit_results = self.audit_project_tests()
        
        # Create report
        report = self.create_focused_report(audit_results)
        
        # Save report
        report_path = self.save_focused_report(report)
        
        # Print summary
        print(f"\n📊 Focused Audit Summary:")
        print(f"  📁 Project Python files: {audit_results['total_files']}")
        print(f"  ✅ Files with tests: {audit_results['files_with_tests']}")
        print(f"  ❌ Files without tests: {audit_results['files_without_tests']}")
        print(f"  📈 Test coverage: {audit_results['test_coverage']:.1%}")
        print(f"  📋 Report saved to: {report_path}")
        
        if audit_results['missing_tests']:
            print(f"\n⚠️  Missing tests for {len(audit_results['missing_tests'])} files")
            print("💡 Run the comprehensive test runner to create visual tests")
        
        return audit_results

if __name__ == "__main__":
    auditor = FocusedTestAuditor()
    results = auditor.run_focused_audit()
