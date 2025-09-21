#!/usr/bin/env python3
"""
Confidence Compliance Integration
Integrates anti-overconfidence validation into the Quark compliance system
"""

import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for confidence_validator import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from confidence_validator import ConfidenceValidator, ConfidenceLevel

# Setup logging
logger = logging.getLogger(__name__)


class ConfidenceComplianceChecker:
    """
    Compliance checker for confidence and validation requirements
    """
    
    def __init__(self):
        """Initialize the confidence compliance checker"""
        self.validator = ConfidenceValidator()
        self.violations = []
        self.warnings = []
        self.resources_used = []
        
    def check_file_confidence(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Check if a file contains proper confidence markers
        
        Args:
            file_path: Path to file to check
            
        Returns:
            Tuple of (passes_check, list_of_violations)
        """
        violations = []
        
        # Only check certain file types
        if file_path.suffix not in ['.py', '.md', '.txt']:
            return True, []
            
        try:
            content = file_path.read_text()
            
            # Check for overconfident patterns
            overconfident_patterns = [
                (r'(?:absolutely|definitely|certainly) (?:certain|sure|correct)', 
                 "Overconfident language detected"),
                (r'100%', "100% confidence claimed (forbidden)"),
                (r'guaranteed to work', "Absolute guarantee claimed"),
                (r'never fails', "Impossible claim made"),
                (r'always works', "Absolute claim without qualification")
            ]
            
            for pattern, violation_msg in overconfident_patterns:
                import re
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(f"{file_path}: {violation_msg}")
                    
            # Check for missing validation in code files
            if file_path.suffix == '.py':
                # Look for functions that make claims without validation
                function_pattern = r'def\s+\w+\s*\([^)]*\).*?(?=\n(?:def\s|\Z))'
                functions = re.findall(function_pattern, content, re.DOTALL)
                
                for func in functions:
                    # Check if function has assertions/validation
                    if 'return' in func and not any(word in func for word in 
                        ['assert', 'validate', 'check', 'verify', 'test']):
                        
                        # Extract function name for reporting
                        func_name = re.search(r'def\s+(\w+)', func)
                        if func_name:
                            violations.append(
                                f"{file_path}: Function '{func_name.group(1)}' "
                                f"returns without validation"
                            )
                            
        except Exception as e:
            logger.error(f"Error checking {file_path}: {e}")
            return False, [f"Error checking file: {e}"]
            
        return len(violations) == 0, violations
        
    def check_documentation_confidence(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check documentation for proper confidence expression
        
        Args:
            content: Documentation content to check
            
        Returns:
            Tuple of (passes_check, list_of_issues)
        """
        issues = []
        
        # Check for claims without sources
        import re
        
        claim_patterns = [
            r'(?:increases?|improves?|reduces?|enhances?) (?:by|performance|efficiency)',
            r'(?:faster|slower|better|worse) than',
            r'(?:scientifically|clinically) proven',
            r'best practice',
            r'industry standard'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Check if there's a citation nearby (within 50 chars)
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                # Look for citations
                if not re.search(r'\[[\d\w]+\]|\(\d{4}\)|https?://|source:|reference:', 
                                context, re.IGNORECASE):
                    issues.append(f"Unsourced claim: '{match.group()}'")
                    
        return len(issues) == 0, issues
    
    def perform_enhanced_validation(self, file_path: Path) -> Dict[str, Any]:
        """
        Perform enhanced validation using all available resources
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Enhanced validation results
        """
        try:
            content = file_path.read_text()
            
            # Detect validation needs
            categories = self.validator.detect_validation_needs(content)
            
            # Perform enhanced validation
            validation_plan = self.validator.perform_enhanced_validation(content)
            
            # Track resources used
            self.resources_used.extend(validation_plan['resources_selected'])
            
            # Check if sufficient resources were consulted
            if len(validation_plan['resources_selected']) == 0:
                self.warnings.append(
                    f"{file_path}: No validation resources available for detected categories"
                )
            elif len(validation_plan['resources_selected']) < 2:
                self.warnings.append(
                    f"{file_path}: Limited validation resources ({len(validation_plan['resources_selected'])}) - consider manual verification"
                )
            
            return {
                'file': str(file_path),
                'categories': validation_plan['categories'],
                'resources': validation_plan['resources_selected'],
                'instructions': validation_plan['validation_instructions'],
                'confidence_boost': validation_plan['confidence_adjustment']
            }
            
        except Exception as e:
            logger.error(f"Error performing enhanced validation on {file_path}: {e}")
            return {
                'file': str(file_path),
                'error': str(e)
            }
        
    def generate_confidence_report(self, files_checked: List[Path]) -> str:
        """
        Generate a confidence compliance report
        
        Args:
            files_checked: List of files that were checked
            
        Returns:
            Formatted report string
        """
        report = """
═══════════════════════════════════════════════════════════════════
           ENHANCED CONFIDENCE COMPLIANCE REPORT
═══════════════════════════════════════════════════════════════════

📊 SUMMARY:
-----------"""
        
        report += f"""
Files Checked: {len(files_checked)}
Violations Found: {len(self.violations)}
Warnings Issued: {len(self.warnings)}
Resources Available: {len(self.validator.resources)}
Resources Used: {len(set(r['name'] for r in self.resources_used))}
"""
        
        if self.violations:
            report += """
🚫 VIOLATIONS:
--------------
"""
            for violation in self.violations:
                report += f"  ❌ {violation}\n"
                
        if self.warnings:
            report += """
⚠️ WARNINGS:
------------
"""
            for warning in self.warnings:
                report += f"  ⚠️ {warning}\n"
                
        # Add resource usage details
        if self.resources_used:
            unique_resources = {}
            for resource in self.resources_used:
                name = resource['name']
                if name not in unique_resources:
                    unique_resources[name] = resource
            
            report += """
📡 VALIDATION RESOURCES USED:
-----------------------------
"""
            for name, resource in list(unique_resources.items())[:5]:
                report += f"  • {name} ({resource['type']})\n"
            if len(unique_resources) > 5:
                report += f"  ... and {len(unique_resources)-5} more\n"
        
        report += f"""
🔧 AVAILABLE RESOURCES:
-----------------------
MCP Servers: {sum(1 for r in self.validator.resources.values() if r.resource_type.value == 'mcp_server')}
APIs: {sum(1 for r in self.validator.resources.values() if r.resource_type.value == 'api')}
Credentials: {self.validator.credentials_path}

📋 COMPLIANCE REQUIREMENTS:
---------------------------
1. Never claim 100% confidence
2. Always include uncertainty markers
3. Cite sources for all claims
4. Validate before returning results
5. Express confidence levels explicitly
6. Use available MCP servers and APIs for validation

🔍 ENFORCEMENT ACTIONS:
-----------------------
- All violations must be fixed before commit
- Overconfident language triggers immediate review
- Missing validations block deployment
- Unsourced claims require citation addition
- Low confidence requires multi-resource validation

💡 VALIDATION INSTRUCTIONS:
---------------------------
- When uncertain: Use MCP servers (Context7, arXiv, PubMed)
- For biological data: Use AlphaFold, UniProt, BLAST APIs
- For chemical data: Use PubChem API
- For ML/datasets: Use OpenML API
- For materials: Use Materials Project API

═══════════════════════════════════════════════════════════════════
"""
        
        return report
        
    def integrate_with_compliance_system(self):
        """
        Hook into the main compliance system
        """
        try:
            # Import the main compliance system
            from compliance_system.core_system import QuarkComplianceSystem
            
            # Register confidence checks
            system = QuarkComplianceSystem()
            
            # Add confidence validation to pre-operation checks
            original_check = system.check_compliance_now
            
            def enhanced_check(target_files=None):
                """Enhanced compliance check with comprehensive validation"""
                # Run original checks
                result = original_check(target_files)
                
                # Add confidence and resource validation checks
                if target_files:
                    for file_path in target_files:
                        path = Path(file_path)
                        if path.exists():
                            # Basic confidence checks
                            passes, violations = self.check_file_confidence(path)
                            if not passes:
                                self.violations.extend(violations)
                                result = False
                            
                            # Enhanced validation for Python and Markdown files
                            if path.suffix in ['.py', '.md']:
                                validation_result = self.perform_enhanced_validation(path)
                                
                                # Log validation plan
                                if validation_result.get('instructions'):
                                    logger.info(f"Enhanced validation for {file_path}:")
                                    for instruction in validation_result['instructions']:
                                        logger.info(f"  • {instruction}")
                                
                                # Check for errors
                                if 'error' in validation_result:
                                    self.warnings.append(f"{file_path}: Validation error - {validation_result['error']}")
                                
                return result
                
            # Replace the method
            system.check_compliance_now = enhanced_check
            
            logger.info("✅ Confidence validation integrated with compliance system")
            
        except ImportError as e:
            logger.warning(f"Could not integrate with main compliance system: {e}")


def main():
    """
    CLI interface for confidence compliance checking
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Confidence Compliance Checker')
    parser.add_argument('--check', type=str, help='Check confidence compliance in file/directory')
    parser.add_argument('--report', action='store_true', help='Generate compliance report')
    parser.add_argument('--integrate', action='store_true', help='Integrate with main compliance system')
    
    args = parser.parse_args()
    
    checker = ConfidenceComplianceChecker()
    
    if args.integrate:
        checker.integrate_with_compliance_system()
        print("✅ Confidence validation integrated with compliance system")
        
    elif args.check:
        path = Path(args.check)
        
        if path.is_file():
            passes, violations = checker.check_file_confidence(path)
            if not passes:
                print("❌ Confidence compliance violations found:")
                for v in violations:
                    print(f"  - {v}")
                sys.exit(1)
            else:
                print("✅ File passes confidence compliance")
                
        elif path.is_dir():
            files_checked = []
            for file_path in path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.py', '.md', '.txt']:
                    files_checked.append(file_path)
                    passes, violations = checker.check_file_confidence(file_path)
                    if not passes:
                        checker.violations.extend(violations)
                        
            print(checker.generate_confidence_report(files_checked))
            
            if checker.violations:
                sys.exit(1)
                
    elif args.report:
        print(checker.generate_confidence_report([]))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
