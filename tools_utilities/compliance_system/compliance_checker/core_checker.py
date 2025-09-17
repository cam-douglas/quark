"""
Core Compliance Checker

Main compliance checking logic and orchestration.

Author: Quark AI
Date: 2025-01-27
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
# Handle both relative and absolute imports
try:
    from .violation_types import Violation, ComplianceReport
    from .rule_config import get_rule_config
    from .file_checkers import (
        FileSizeChecker, ContentPatternChecker,
        PythonFileChecker, TestFileChecker
    )
except ImportError:
    from violation_types import Violation, ComplianceReport
    from rule_config import get_rule_config
    from file_checkers import (
        FileSizeChecker, ContentPatternChecker,
        PythonFileChecker, TestFileChecker
    )


class RuleComplianceChecker:
    """
    Comprehensive rule compliance checker for Quark project.
    
    Prevents violations of:
    - 300-line file limit
    - Code quality standards
    - Documentation requirements
    - Security constraints
    - Biological compliance rules
    """
    
    def __init__(self, workspace_root: str = "/Users/camdouglas/quark"):
        """Initialize compliance checker"""
        self.workspace_root = Path(workspace_root)
        self.violations: List[Violation] = []
        self.rule_config = get_rule_config()
        
        # Initialize specialized checkers
        self.size_checker = FileSizeChecker(self.rule_config["file_size_limits"])
        self.pattern_checker = ContentPatternChecker(self.rule_config["prohibited_patterns"])
        self.python_checker = PythonFileChecker()
        self.test_checker = TestFileChecker()
    
    def check_file_compliance(self, file_path: Path) -> List[Violation]:
        """Check individual file for rule compliance"""
        violations = []
        
        if not file_path.exists():
            return violations
            
        # Skip excluded directories
        if any(exclusion in str(file_path) for exclusion in self.rule_config["directory_exclusions"]):
            return violations
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
        except (UnicodeDecodeError, PermissionError):
            return violations  # Skip binary or inaccessible files
        
        # Check file size limits
        violations.extend(self.size_checker.check_file_size(file_path, lines))
        
        # Check content patterns
        violations.extend(self.pattern_checker.check_content_patterns(file_path, content))
        
        # Check specific file type requirements
        violations.extend(self._check_file_type_requirements(file_path, content))
        
        return violations
    
    def _check_file_type_requirements(self, file_path: Path, content: str) -> List[Violation]:
        """Check file type specific requirements"""
        violations = []
        
        if file_path.suffix == '.py':
            violations.extend(self.python_checker.check_python_requirements(file_path, content))
        elif 'test' in file_path.name.lower():
            violations.extend(self.test_checker.check_test_requirements(file_path, content))
        
        return violations
    
    def check_workspace_compliance(self, paths: Optional[List[str]] = None) -> ComplianceReport:
        """Check entire workspace for compliance"""
        self.violations = []
        
        if paths is None:
            # Check all relevant files in workspace
            paths = self._get_files_to_check()
        
        total_files = 0
        for file_path in paths:
            path_obj = Path(file_path)
            if path_obj.is_file():
                total_files += 1
                self.violations.extend(self.check_file_compliance(path_obj))
        
        # Generate summary
        summary = {
            "errors": len([v for v in self.violations if v.severity == "error"]),
            "warnings": len([v for v in self.violations if v.severity == "warning"]),
            "info": len([v for v in self.violations if v.severity == "info"]),
        }
        
        return ComplianceReport(
            timestamp=datetime.now(),
            total_files_checked=total_files,
            violations=self.violations,
            compliant=summary["errors"] == 0,
            summary=summary
        )
    
    def _get_files_to_check(self) -> List[str]:
        """Get list of files to check"""
        files = []
        
        # Get all relevant files
        for root, dirs, filenames in os.walk(self.workspace_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(exclusion in d for exclusion in self.rule_config["directory_exclusions"])]
            
            for filename in filenames:
                if filename.endswith(('.py', '.md', '.yaml', '.yml', '.json')):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def generate_report(self, report: ComplianceReport, output_file: Optional[str] = None) -> str:
        """Generate compliance report"""
        report_lines = [
            f"# Rule Compliance Report",
            f"Generated: {report.timestamp}",
            f"Files Checked: {report.total_files_checked}",
            f"Compliant: {'✅ YES' if report.compliant else '❌ NO'}",
            "",
            f"## Summary",
            f"- Errors: {report.summary['errors']}",
            f"- Warnings: {report.summary['warnings']}",
            f"- Info: {report.summary['info']}",
            ""
        ]
        
        if report.violations:
            report_lines.extend([
                "## Violations",
                ""
            ])
            
            # Group by severity
            for severity in ["error", "warning", "info"]:
                severity_violations = [v for v in report.violations if v.severity == severity]
                if severity_violations:
                    report_lines.extend([
                        f"### {severity.title()}s",
                        ""
                    ])
                    
                    for violation in severity_violations:
                        report_lines.extend([
                            f"**{violation.file_path}**",
                            f"- Rule: {violation.rule_type}",
                            f"- Message: {violation.message}",
                        ])
                        
                        if violation.line_number:
                            report_lines.append(f"- Line: {violation.line_number}")
                        
                        if violation.fix_suggestion:
                            report_lines.append(f"- Fix: {violation.fix_suggestion}")
                        
                        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def fix_violations(self, violations: List[Violation]) -> Dict[str, Any]:
        """Attempt to automatically fix violations where possible"""
        fixes_applied = []
        fixes_failed = []
        
        for violation in violations:
            if violation.rule_type.startswith("file_size_"):
                # File size violations require manual intervention
                fixes_failed.append({
                    "violation": violation,
                    "reason": "File size violations require manual refactoring"
                })
            else:
                # Other violations might be auto-fixable
                fixes_failed.append({
                    "violation": violation,
                    "reason": "Auto-fix not implemented for this violation type"
                })
        
        return {
            "applied": fixes_applied,
            "failed": fixes_failed
        }
