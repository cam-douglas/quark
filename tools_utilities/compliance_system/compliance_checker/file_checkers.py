"""
File-Specific Compliance Checkers

Specialized checkers for different file types and patterns.

Author: Quark AI
Date: 2025-01-27
"""

import re
from pathlib import Path
from typing import List

# Handle both relative and absolute imports
try:
    from .violation_types import Violation
except ImportError:
    from violation_types import Violation


class FileSizeChecker:
    """Checks file size compliance"""
    
    def __init__(self, limits: dict):
        self.limits = limits
    
    def check_file_size(self, file_path: Path, lines: List[str]) -> List[Violation]:
        """Check file size against limits"""
        violations = []
        line_count = len(lines)
        
        # Determine file type and limit
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.py':
            limit = self.limits["python_files"]
            rule_type = "file_size_python"
        elif file_extension in ['.md', '.rst']:
            limit = self.limits["markdown_files"]
            rule_type = "file_size_markdown"
        elif file_extension in ['.yaml', '.yml']:
            limit = self.limits["yaml_files"]
            rule_type = "file_size_yaml"
        elif file_extension == '.json':
            limit = self.limits["json_files"]
            rule_type = "file_size_json"
        else:
            return violations  # No limit for this file type
        
        if line_count > limit:
            violations.append(Violation(
                rule_type=rule_type,
                file_path=str(file_path),
                line_number=None,
                message=f"File exceeds {limit} line limit: {line_count} lines",
                severity="error",
                fix_suggestion=f"Split file into smaller modules. Consider breaking at natural boundaries (functions, classes, sections)."
            ))
        elif line_count > limit * 0.8:  # Warning at 80% of limit
            violations.append(Violation(
                rule_type=rule_type,
                file_path=str(file_path),
                line_number=None,
                message=f"File approaching {limit} line limit: {line_count} lines",
                severity="warning",
                fix_suggestion="Consider refactoring to prevent future violations."
            ))
        
        return violations


class ContentPatternChecker:
    """Checks content for prohibited patterns"""
    
    def __init__(self, prohibited_patterns: dict):
        self.prohibited_patterns = prohibited_patterns
    
    def check_content_patterns(self, file_path: Path, content: str) -> List[Violation]:
        """Check content for prohibited patterns"""
        violations = []
        
        # Check prohibited patterns
        for pattern_type, patterns in self.prohibited_patterns.items():
            if pattern_type == "all_files" or self._is_brain_module(file_path):
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        violations.append(Violation(
                            rule_type="prohibited_pattern",
                            file_path=str(file_path),
                            line_number=line_num,
                            message=f"Prohibited pattern found: {pattern}",
                            severity="error",
                            fix_suggestion="Remove or refactor prohibited pattern."
                        ))
        
        return violations
    
    def _is_brain_module(self, file_path: Path) -> bool:
        """Check if file is in brain module directory"""
        return 'brain' in str(file_path) or 'modules' in str(file_path)


class PythonFileChecker:
    """Checks Python-specific requirements"""
    
    def check_python_requirements(self, file_path: Path, content: str) -> List[Violation]:
        """Check Python file requirements"""
        violations = []
        
        # Check for docstring
        if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
            violations.append(Violation(
                rule_type="python_docstring",
                file_path=str(file_path),
                line_number=1,
                message="Python file missing module docstring",
                severity="warning",
                fix_suggestion="Add module docstring at the top of the file."
            ))
        
        # Check for type hints in function definitions
        function_defs = re.findall(r'def\s+\w+\s*\([^)]*\):', content)
        if function_defs and 'from typing import' not in content:
            violations.append(Violation(
                rule_type="python_typing",
                file_path=str(file_path),
                line_number=None,
                message="Python file with functions missing typing imports",
                severity="warning",
                fix_suggestion="Add 'from typing import' statement for type hints."
            ))
        
        return violations


class TestFileChecker:
    """Checks test file requirements"""
    
    def check_test_requirements(self, file_path: Path, content: str) -> List[Violation]:
        """Check test file requirements"""
        violations = []
        
        if 'def test_' not in content:
            violations.append(Violation(
                rule_type="test_functions",
                file_path=str(file_path),
                line_number=None,
                message="Test file missing test functions",
                severity="warning",
                fix_suggestion="Add test functions starting with 'test_'."
            ))
        
        return violations
