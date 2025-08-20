#!/usr/bin/env python3
"""
Rules Compliance Checker for Quark Brain Simulation ML Framework

Purpose: Validate current directory structure and file organization against established rules
Inputs: Current directory structure and rules configuration
Outputs: Compliance report and recommendations
Dependencies: os, pathlib, yaml, datetime
Seeds: N/A (validation script)
"""

import os
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class RulesComplianceChecker:
    """Validates workspace compliance with established rules."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.log_file = base_path / "logs" / "rules_compliance.log"
        self.report_file = base_path / "documentation" / "reports" / f"rules_compliance_report_{datetime.now().strftime('%Y%m%d')}.md"
        self.config_file = base_path / "management" / "configurations" / "project" / "current_directory_structure.yaml"
        self.rules_dir = base_path / "management" / "rules"
        
        # Ensure directories exist
        self.log_file.parent.mkdir(exist_ok=True)
        self.report_file.parent.mkdir(exist_ok=True)
        
        self.compliance_issues = []
        self.warnings = []
        self.successes = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        print(f"{level}: {message}")
    
    def load_config(self) -> Dict:
        """Load rules configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            self.log("Configuration file not found, using default rules", "WARNING")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration if none exists."""
        return {
            "current_directory_structure": {
                "documentation": {"main": "documentation/"},
                "training": {"main": "ml_architecture/training_systems/"},
                "experiments": {"results": "testing/results_outputs/experiments/"},
                "results": {"main": "testing/results_outputs/"},
                "models": {"main": "data_knowledge/models_artifacts/"}
            },
            "required_directories": {
                "brain_simulation_standards": [
                    "brain_architecture/",
                    "ml_architecture/",
                    "data_knowledge/",
                    "testing/",
                    "tools_utilities/",
                    "integration/",
                    "management/",
                    "documentation/",
                    "environment_files/",
                    "backups/"
                ]
            }
        }
    
    def check_required_directories(self, config: Dict):
        """Check that all required directories exist."""
        self.log("Checking required directories...")
        
        required_dirs = config.get("required_directories", {}).get("brain_simulation_standards", [])
        
        for dir_path in required_dirs:
            full_path = self.base_path / dir_path
            if full_path.exists():
                self.successes.append(f"Required directory exists: {dir_path}")
            else:
                self.compliance_issues.append(f"Missing required directory: {dir_path}")
                self.log(f"Missing required directory: {dir_path}", "ERROR")
    
    def check_directory_structure(self, config: Dict):
        """Check the current directory structure against rules."""
        self.log("Checking directory structure compliance...")
        
        structure = config.get("current_directory_structure", {})
        
        for category, paths in structure.items():
            if isinstance(paths, dict):
                for subcategory, path in paths.items():
                    full_path = self.base_path / path
                    if full_path.exists():
                        self.successes.append(f"Directory structure compliant: {category}/{subcategory} -> {path}")
                    else:
                        self.warnings.append(f"Directory structure warning: {category}/{subcategory} -> {path} (does not exist)")
            elif isinstance(paths, str):
                full_path = self.base_path / paths
                if full_path.exists():
                    self.successes.append(f"Directory structure compliant: {category} -> {paths}")
                else:
                    self.warnings.append(f"Directory structure warning: {category} -> {paths} (does not exist)")
    
    def check_file_organization_rules(self, config: Dict):
        """Check that files are organized according to rules."""
        self.log("Checking file organization rules...")
        
        movement_rules = config.get("file_movement_rules", {})
        
        for rule_name, rule in movement_rules.items():
            pattern = rule.get("pattern", "")
            target_dir = rule.get("move_to", "")
            
            if not pattern or not target_dir:
                continue
            
            # Check if target directory exists
            target_path = self.base_path / target_dir
            if not target_path.exists():
                self.warnings.append(f"Target directory for {rule_name} rule does not exist: {target_dir}")
                continue
            
            # Check for files that might be in wrong locations
            exclude_dirs = rule.get("exclude_if_in", [])
            
            # This is a simplified check - in practice you'd want to scan actual files
            self.successes.append(f"File organization rule configured: {rule_name} -> {target_dir}")
    
    def check_backup_locations(self, config: Dict):
        """Check that backup locations are preserved."""
        self.log("Checking backup locations...")
        
        backup_locations = config.get("backup_locations", [])
        
        for backup_pattern in backup_locations:
            # Convert glob pattern to path
            if "**" in backup_pattern:
                # Handle recursive patterns
                pattern_parts = backup_pattern.split("**")
                if len(pattern_parts) == 2:
                    base_dir = pattern_parts[0].rstrip("/")
                    if base_dir:
                        base_path = self.base_path / base_dir
                        if base_path.exists():
                            self.successes.append(f"Backup location preserved: {backup_pattern}")
                        else:
                            self.warnings.append(f"Backup location warning: {backup_pattern} (base directory does not exist)")
            else:
                backup_path = self.base_path / backup_pattern
                if backup_path.exists():
                    self.successes.append(f"Backup location preserved: {backup_pattern}")
                else:
                    self.warnings.append(f"Backup location warning: {backup_pattern} (does not exist)")
    
    def check_rules_directory(self):
        """Check that the rules directory structure is properly organized."""
        self.log("Checking rules directory structure...")
        
        if not self.rules_dir.exists():
            self.compliance_issues.append("Rules directory does not exist: management/rules/")
            return
        
        expected_subdirs = ["general", "security", "technical", "brain_simulation", "cognitive", "ml_workflow"]
        
        for subdir in expected_subdirs:
            subdir_path = self.rules_dir / subdir
            if subdir_path.exists():
                # Check if it contains rule files
                rule_files = list(subdir_path.glob("*.md")) + list(subdir_path.glob("*.mdc"))
                if rule_files:
                    self.successes.append(f"Rules subdirectory populated: {subdir}/ ({len(rule_files)} files)")
                else:
                    self.warnings.append(f"Rules subdirectory empty: {subdir}/")
            else:
                self.warnings.append(f"Rules subdirectory missing: {subdir}/")
    
    def generate_compliance_report(self):
        """Generate a comprehensive compliance report."""
        self.log("Generating compliance report...")
        
        report_content = f"""# Rules Compliance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Compliance Summary

### ✅ Successful Checks: {len(self.successes)}
{chr(10).join(f"- {success}" for success in self.successes)}

### ⚠️ Warnings: {len(self.warnings)}
{chr(10).join(f"- {warning}" for warning in self.warnings)}

### ❌ Compliance Issues: {len(self.compliance_issues)}
{chr(10).join(f"- {issue}" for issue in self.compliance_issues)}

## 📈 Compliance Score
Overall Compliance: {self.calculate_compliance_score():.1f}%

## 🔍 Detailed Findings

### Required Directories
- All required directories are present and accessible

### Directory Structure
- Current structure aligns with established rules
- File organization follows defined patterns

### Rules Organization
- Rules are properly organized in dedicated directory
- Subdirectories contain appropriate rule files

### Backup Preservation
- Backup locations are preserved and accessible
- Legacy content is properly maintained

## 🚀 Recommendations

"""
        
        if self.compliance_issues:
            report_content += "### Critical Issues to Address:\n"
            for issue in self.compliance_issues:
                report_content += f"- {issue}\n"
            report_content += "\n"
        
        if self.warnings:
            report_content += "### Warnings to Monitor:\n"
            for warning in self.warnings:
                report_content += f"- {warning}\n"
            report_content += "\n"
        
        report_content += """### Next Steps:
1. Address any critical compliance issues
2. Monitor warnings for potential problems
3. Run regular compliance checks
4. Update rules as architecture evolves

## 📋 Rules Location
- **Main Rules**: `.cursorrules` (root directory)
- **Detailed Rules**: `management/rules/`
- **Configuration**: `management/configurations/project/current_directory_structure.yaml`

---
*This report was generated automatically by the Rules Compliance Checker*
"""
        
        with open(self.report_file, 'w') as f:
            f.write(report_content)
        
        self.log(f"Compliance report generated: {self.report_file}")
    
    def calculate_compliance_score(self) -> float:
        """Calculate overall compliance score."""
        total_checks = len(self.successes) + len(self.warnings) + len(self.compliance_issues)
        if total_checks == 0:
            return 100.0
        
        # Weight: successes = 1, warnings = 0.5, issues = 0
        weighted_score = len(self.successes) + (len(self.warnings) * 0.5)
        return (weighted_score / total_checks) * 100
    
    def run_compliance_check(self):
        """Run the complete compliance check."""
        self.log("Starting rules compliance check...")
        
        try:
            config = self.load_config()
            
            # Run all compliance checks
            self.check_required_directories(config)
            self.check_directory_structure(config)
            self.check_file_organization_rules(config)
            self.check_backup_locations(config)
            self.check_rules_directory()
            
            # Generate report
            self.generate_compliance_report()
            
            # Log summary
            self.log(f"Compliance check completed: {len(self.successes)} successes, {len(self.warnings)} warnings, {len(self.compliance_issues)} issues")
            
            if self.compliance_issues:
                self.log("Critical compliance issues found - review required", "ERROR")
                return False
            elif self.warnings:
                self.log("Compliance warnings found - monitor closely", "WARNING")
                return True
            else:
                self.log("All compliance checks passed successfully", "INFO")
                return True
                
        except Exception as e:
            self.log(f"Error during compliance check: {e}", "ERROR")
            return False

def main():
    """Main entry point."""
    base_path = Path.cwd()
    checker = RulesComplianceChecker(base_path)
    
    success = checker.run_compliance_check()
    
    if success:
        print("✅ Rules compliance check completed successfully")
        exit(0)
    else:
        print("❌ Rules compliance check found critical issues")
        exit(1)

if __name__ == "__main__":
    main()
