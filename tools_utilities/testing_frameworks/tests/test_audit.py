#!/usr/bin/env python3
"""
TEST AUDIT: Check all Python files for corresponding tests
Purpose: Audit all Python files and ensure they have visual tests
Inputs: All Python files in the project
Outputs: Audit report with missing tests
Seeds: 42
Dependencies: pathlib, os, sys
"""

import os, sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

class TestAuditor:
    """Audit all Python files for corresponding tests"""
    
    # Class-level attributes for pytest compatibility
    project_root = Path(".")
    test_results = {}
    missing_tests = []
    existing_tests = []
    
    def setup_audit(self):
        """Setup audit configuration"""
        self.project_root = Path(".")
        self.test_results = {}
        self.missing_tests = []
        self.existing_tests = []
    
    def find_all_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        
        # Exclude common directories
        exclude_dirs = {
            '__pycache__', 'venv', 'env', '.venv', '.env',
            'node_modules', '.git', '.vscode', '.idea',
            'site-packages', 'dist', 'build', '*.egg-info'
        }
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Skip files in excluded directories
                    if not any(exclude in str(file_path) for exclude in exclude_dirs):
                        python_files.append(file_path)
        
        return python_files
    
    def find_all_test_files(self) -> List[Path]:
        """Find all test files in the project"""
        test_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py') and ('test' in file.lower() or 'tests' in str(root)):
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
    
    def audit_tests(self) -> Dict:
        """Audit all Python files for tests"""
        print("🔍 Starting test audit...")
        
        # Find all Python files
        python_files = self.find_all_python_files()
        print(f"📁 Found {len(python_files)} Python files")
        
        # Find all test files
        test_files = self.find_all_test_files()
        print(f"🧪 Found {len(test_files)} test files")
        
        # Audit each Python file
        audit_results = {
            'total_files': len(python_files),
            'files_with_tests': 0,
            'files_without_tests': 0,
            'test_coverage': 0.0,
            'missing_tests': [],
            'existing_tests': [],
            'test_locations': {}
        }
        
        for py_file in python_files:
            component_name = self.get_component_name(py_file)
            corresponding_test = self.find_corresponding_test(component_name, test_files)
            
            if corresponding_test:
                audit_results['files_with_tests'] += 1
                audit_results['existing_tests'].append({
                    'component': str(py_file),
                    'test': str(corresponding_test),
                    'component_name': component_name
                })
                audit_results['test_locations'][str(py_file)] = str(corresponding_test)
            else:
                audit_results['files_without_tests'] += 1
                audit_results['missing_tests'].append({
                    'component': str(py_file),
                    'component_name': component_name,
                    'suggested_test_path': self.suggest_test_path(py_file)
                })
        
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
    
    def create_audit_report(self, audit_results: Dict) -> str:
        """Create a comprehensive audit report"""
        print("📋 Creating audit report...")
        
        report = f"""
# 🧪 TEST AUDIT REPORT

## 📊 Summary
- **Total Python Files**: {audit_results['total_files']}
- **Files with Tests**: {audit_results['files_with_tests']}
- **Files without Tests**: {audit_results['files_without_tests']}
- **Test Coverage**: {audit_results['test_coverage']:.1%}

## ✅ Files with Tests ({len(audit_results['existing_tests'])})
"""
        
        for item in audit_results['existing_tests']:
            report += f"- **{item['component_name']}**: `{item['component']}` → `{item['test']}`\n"
        
        report += f"""
## ❌ Missing Tests ({len(audit_results['missing_tests'])})
"""
        
        for item in audit_results['missing_tests']:
            report += f"- **{item['component_name']}**: `{item['component']}`\n"
            report += f"  - Suggested: `{item['suggested_test_path']}`\n"
        
        report += f"""
## 📈 Recommendations

### High Priority (Core Components)
"""
        
        # Prioritize core components
        core_components = ['brain_launcher', 'neural_components', 'developmental_timeline', 
                          'training_orchestrator', 'sleep_consolidation', 'multi_scale_integration']
        
        for component in core_components:
            missing = [item for item in audit_results['missing_tests'] if component in item['component_name'].lower()]
            if missing:
                report += f"- Create test for **{component}**\n"
        
        report += f"""
### Medium Priority (Supporting Components)
- Create tests for configuration files
- Create tests for utility scripts

### Low Priority
- `__init__.py` files (usually don't need tests)
- Simple script files

## 🚀 Next Steps
1. Run the comprehensive test runner: `python tests/comprehensive_test_runner.py`
2. Create missing tests for high-priority components
3. Ensure all tests include visual validation
4. Update test coverage regularly
"""
        
        return report
    
    def save_audit_report(self, report: str, filename: str = "test_audit_report.md"):
        """Save audit report to file"""
        output_path = Path("tests") / filename
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Audit report saved to {output_path}")
        return output_path
    
    def run_audit(self):
        """Run the complete audit"""
        print("🚀 Starting Test Audit...")
        
        # Perform audit
        audit_results = self.audit_tests()
        
        # Create report
        report = self.create_audit_report(audit_results)
        
        # Save report
        report_path = self.save_audit_report(report)
        
        # Print summary
        print(f"\n📊 Audit Summary:")
        print(f"  📁 Total Python files: {audit_results['total_files']}")
        print(f"  ✅ Files with tests: {audit_results['files_with_tests']}")
        print(f"  ❌ Files without tests: {audit_results['files_without_tests']}")
        print(f"  📈 Test coverage: {audit_results['test_coverage']:.1%}")
        print(f"  📋 Report saved to: {report_path}")
        
        if audit_results['missing_tests']:
            print(f"\n⚠️  Missing tests for {len(audit_results['missing_tests'])} files")
            print("💡 Run the comprehensive test runner to create visual tests")
        
        return audit_results

if __name__ == "__main__":
    auditor = TestAuditor()
    results = auditor.run_audit()
