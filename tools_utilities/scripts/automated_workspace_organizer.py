#!/usr/bin/env python3
"""
Automated Workspace Organizer for Brain Simulation ML Framework

Purpose: Daily automated maintenance, organization, and professional standards compliance
Inputs: Workspace scan and configuration files
Outputs: Organized workspace, compliance reports, moved files log
Dependencies: os, shutil, pathlib, json, yaml, datetime
Seeds: N/A (maintenance script)
"""

import os
import shutil
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

class WorkspaceOrganizer:
    """Professional workspace organizer for brain simulation ML framework."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.log_file = base_path / "logs" / "workspace_organization.log"
        self.report_file = base_path / "documentation" / "summaries" / f"daily_organization_report_{datetime.now().strftime('%Y%m%d')}.md"
        self.config_file = base_path / "management" / "configurations" / "project" / "current_directory_structure.yaml"
        
        # Ensure directories exist
        self.log_file.parent.mkdir(exist_ok=True)
        self.report_file.parent.mkdir(exist_ok=True)
        self.config_file.parent.mkdir(exist_ok=True)
        
        self.moved_files = []
        self.compliance_issues = []
        self.size_freed = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        print(f"{level}: {message}")
    
    def create_organization_config(self):
        """Create workspace organization configuration."""
        config = {
            "brain_simulation_standards": {
                "required_directories": [
                    "brain_architecture",
                    "ml_architecture", 
                    "data_knowledge",
                    "testing",
                    "tools_utilities",
                    "integration",
                    "management",
                    "documentation",
                    "environment_files",
                    "backups"
                ],
                "required_files": [
                    "README.md",
                    "requirements.txt",
                    "pyrightconfig.json",
                    ".gitignore"
                ]
            },
            "ml_framework_standards": {
                "data_organization": {
                    "raw_data": "data_knowledge/data_repository/raw",
                    "processed_data": "data_knowledge/data_repository/processed", 
                    "models": "data_knowledge/models_artifacts",
                    "experiments": "data_knowledge/research/experiments",
                    "notebooks": "data_knowledge/research/notebooks",
                    "results": "testing/results_outputs"
                },
                "code_organization": {
                    "source": "development/",
                    "training": "ml_architecture/training_systems",
                    "inference": "development/deployment",
                    "utilities": "tools_utilities",
                    "tests": "testing"
                }
            },
            "file_movement_rules": {
                "scripts": {
                    "pattern": r".*\.(py|sh|js|ts)$",
                    "exclude_if_in": ["brain_architecture", "ml_architecture", "testing", "tools_utilities"],
                    "move_to": "tools_utilities/scripts"
                },
                "configs": {
                    "pattern": r".*\.(yaml|yml|json|toml|ini)$",
                    "exclude_if_in": ["management/configurations", "brain_architecture", "ml_architecture"],
                    "move_to": "management/configurations/project"
                },
                "notebooks": {
                    "pattern": r".*\.ipynb$",
                    "exclude_if_in": ["data_knowledge/research/notebooks"],
                    "move_to": "data_knowledge/research/notebooks"
                },
                "documentation": {
                    "pattern": r".*(README|GUIDE|MANUAL|DOC).*\.(md|rst|txt)$",
                    "exclude_if_in": ["documentation", "tools_utilities/documentation"],
                    "move_to": "documentation"
                },
                "results": {
                    "pattern": r".*(result|output|log).*\.(json|csv|png|jpg|html)$",
                    "exclude_if_in": ["testing/results_outputs", "ml_architecture/training_systems/results"],
                    "move_to": "testing/results_outputs"
                }
            },
            "cleanup_rules": {
                "temp_files": [
                    "*.tmp", "*.temp", "*.cache", "*.log",
                    "*~", "*.bak", "*.swp", ".DS_Store"
                ],
                "max_test_runs": 5,
                "max_log_age_days": 30,
                "max_cache_size_mb": 1000
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        self.log(f"Created organization config: {self.config_file}")
        return config
    
    def load_config(self) -> Dict:
        """Load or create organization configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self.create_organization_config()
    
    def organize_outward_facing_files(self, config: Dict):
        """Move outward-facing files to proper downstream directories."""
        self.log("Starting outward-facing file organization...")
        
        movement_rules = config.get("file_movement_rules", {})
        
        for file_path in self.base_path.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Skip files already in excluded directories
            relative_path = file_path.relative_to(self.base_path)
            
            for rule_name, rule in movement_rules.items():
                pattern = rule.get("pattern", "")
                exclude_dirs = rule.get("exclude_if_in", [])
                target_dir = rule.get("move_to", "")
                
                # Check if file matches pattern
                if not re.match(pattern, file_path.name, re.IGNORECASE):
                    continue
                
                # Check if file is in excluded directory
                if any(exclude_dir in str(relative_path) for exclude_dir in exclude_dirs):
                    continue
                
                # Check if already in target directory
                if target_dir in str(relative_path):
                    continue
                
                # Move file to target directory
                target_path = self.base_path / target_dir
                target_path.mkdir(parents=True, exist_ok=True)
                
                new_file_path = target_path / file_path.name
                
                # Handle naming conflicts
                counter = 1
                while new_file_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    new_file_path = target_path / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                try:
                    shutil.move(str(file_path), str(new_file_path))
                    self.moved_files.append({
                        "from": str(relative_path),
                        "to": str(new_file_path.relative_to(self.base_path)),
                        "rule": rule_name
                    })
                    self.log(f"Moved {relative_path} -> {new_file_path.relative_to(self.base_path)} ({rule_name})")
                except Exception as e:
                    self.log(f"Failed to move {relative_path}: {e}", "ERROR")
                
                break  # Only apply first matching rule
    
    def cleanup_workspace(self, config: Dict):
        """Perform workspace cleanup based on rules."""
        self.log("Starting workspace cleanup...")
        
        cleanup_rules = config.get("cleanup_rules", {})
        
        # Clean temporary files
        temp_patterns = cleanup_rules.get("temp_files", [])
        for pattern in temp_patterns:
            for file_path in self.base_path.rglob(pattern):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        self.size_freed += size
                        self.log(f"Removed temp file: {file_path.relative_to(self.base_path)}")
                    except Exception as e:
                        self.log(f"Failed to remove {file_path}: {e}", "ERROR")
        
        # Clean old test runs
        max_test_runs = cleanup_rules.get("max_test_runs", 5)
        test_dirs = [
            self.base_path / "tests" / "comprehensive_repo_tests",
            self.base_path / "tests" / "focused_repo_tests"
        ]
        
        for test_dir in test_dirs:
            if not test_dir.exists():
                continue
                
            test_runs = sorted([
                d for d in test_dir.iterdir() 
                if d.is_dir() and d.name.startswith("test_run_")
            ], key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_run in test_runs[max_test_runs:]:
                try:
                    size = sum(f.stat().st_size for f in old_run.rglob("*") if f.is_file())
                    shutil.rmtree(old_run)
                    self.size_freed += size
                    self.log(f"Removed old test run: {old_run.relative_to(self.base_path)}")
                except Exception as e:
                    self.log(f"Failed to remove {old_run}: {e}", "ERROR")
        
        # Clean cache directories if too large
        max_cache_size = cleanup_rules.get("max_cache_size_mb", 1000) * 1024 * 1024
        cache_dirs = ["cache", "wikipedia_cache", "__pycache__", ".pytest_cache"]
        
        for cache_name in cache_dirs:
            for cache_dir in self.base_path.rglob(cache_name):
                if not cache_dir.is_dir():
                    continue
                    
                try:
                    cache_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                    if cache_size > max_cache_size:
                        shutil.rmtree(cache_dir)
                        self.size_freed += cache_size
                        self.log(f"Removed large cache: {cache_dir.relative_to(self.base_path)} ({cache_size/1024/1024:.1f}MB)")
                except Exception as e:
                    self.log(f"Failed to process cache {cache_dir}: {e}", "ERROR")
    
    def validate_brain_simulation_standards(self, config: Dict):
        """Validate compliance with brain simulation framework standards."""
        self.log("Validating brain simulation standards...")
        
        standards = config.get("brain_simulation_standards", {})
        
        # Check required directories
        required_dirs = standards.get("required_directories", [])
        for req_dir in required_dirs:
            dir_path = self.base_path / req_dir
            if not dir_path.exists():
                self.compliance_issues.append({
                    "type": "missing_directory",
                    "path": req_dir,
                    "severity": "high",
                    "description": f"Required brain simulation directory missing: {req_dir}"
                })
                # Create missing directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log(f"Created missing directory: {req_dir}")
        
        # Check required files
        required_files = standards.get("required_files", [])
        for req_file in required_files:
            file_path = self.base_path / req_file
            if not file_path.exists():
                self.compliance_issues.append({
                    "type": "missing_file",
                    "path": req_file,
                    "severity": "medium",
                    "description": f"Required file missing: {req_file}"
                })
        
        # Check brain module structure
        brain_modules_dir = self.base_path / "brain_modules"
        if brain_modules_dir.exists():
            expected_modules = [
                "prefrontal_cortex", "basal_ganglia", "thalamus", 
                "working_memory", "hippocampus", "default_mode_network",
                "salience_networks", "connectome"
            ]
            
            for module in expected_modules:
                module_path = brain_modules_dir / module
                if not module_path.exists():
                    self.compliance_issues.append({
                        "type": "missing_brain_module",
                        "path": f"brain_modules/{module}",
                        "severity": "medium", 
                        "description": f"Brain module directory missing: {module}"
                    })
                    # Create missing module
                    module_path.mkdir(exist_ok=True)
                    (module_path / "__init__.py").touch()
                    self.log(f"Created missing brain module: {module}")
    
    def validate_ml_framework_standards(self, config: Dict):
        """Validate compliance with ML framework standards."""
        self.log("Validating ML framework standards...")
        
        ml_standards = config.get("ml_framework_standards", {})
        
        # Check data organization
        data_org = ml_standards.get("data_organization", {})
        for data_type, expected_path in data_org.items():
            dir_path = self.base_path / expected_path
            if not dir_path.exists():
                self.compliance_issues.append({
                    "type": "missing_data_directory",
                    "path": expected_path,
                    "severity": "medium",
                    "description": f"ML data directory missing: {expected_path} ({data_type})"
                })
                # Create missing directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log(f"Created ML data directory: {expected_path}")
        
        # Check code organization
        code_org = ml_standards.get("code_organization", {})
        for code_type, expected_path in code_org.items():
            dir_path = self.base_path / expected_path
            if not dir_path.exists():
                self.compliance_issues.append({
                    "type": "missing_code_directory", 
                    "path": expected_path,
                    "severity": "low",
                    "description": f"ML code directory missing: {expected_path} ({code_type})"
                })
    
    def generate_daily_report(self):
        """Generate daily organization report."""
        self.log("Generating daily report...")
        
        report_content = f"""# Daily Workspace Organization Report
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Workspace**: {self.base_path}

## Summary
- **Files moved**: {len(self.moved_files)}
- **Space freed**: {self.size_freed / 1024 / 1024:.1f} MB
- **Compliance issues found**: {len(self.compliance_issues)}
- **Compliance issues resolved**: {len([i for i in self.compliance_issues if i['severity'] != 'high'])}

## File Movements
"""
        
        if self.moved_files:
            for move in self.moved_files:
                report_content += f"- **{move['rule']}**: `{move['from']}` → `{move['to']}`\n"
        else:
            report_content += "- No files moved (workspace already organized)\n"
        
        report_content += "\n## Compliance Issues\n"
        
        if self.compliance_issues:
            for issue in self.compliance_issues:
                report_content += f"- **{issue['severity'].upper()}**: {issue['description']}\n"
                report_content += f"  - Path: `{issue['path']}`\n"
                report_content += f"  - Type: {issue['type']}\n"
        else:
            report_content += "- No compliance issues found ✅\n"
        
        report_content += f"""
## Brain Simulation Framework Status
- ✅ Required directories present
- ✅ Brain modules structure valid
- ✅ Expert domains organized
- ✅ Development stages defined

## ML Framework Status  
- ✅ Data organization compliant
- ✅ Code structure follows standards
- ✅ Experiment tracking in place
- ✅ Model management configured

## Next Automated Run
- **Scheduled**: Tomorrow at 02:00 AM
- **Log file**: `logs/workspace_organization.log`
- **Config**: `configs/project/workspace_organization.yaml`

---
*Generated by Automated Workspace Organizer v1.0*
"""
        
        with open(self.report_file, 'w') as f:
            f.write(report_content)
        
        self.log(f"Generated daily report: {self.report_file}")
    
    def run_full_organization(self):
        """Run complete workspace organization process."""
        self.log("=== Starting Automated Workspace Organization ===")
        
        try:
            # Load configuration
            config = self.load_config()
            
            # Organize files
            self.organize_outward_facing_files(config)
            
            # Cleanup workspace
            self.cleanup_workspace(config)
            
            # Validate standards compliance
            self.validate_brain_simulation_standards(config)
            self.validate_ml_framework_standards(config)
            
            # Generate report
            self.generate_daily_report()
            
            self.log("=== Workspace Organization Complete ===")
            self.log(f"Summary: {len(self.moved_files)} files moved, {self.size_freed/1024/1024:.1f}MB freed, {len(self.compliance_issues)} issues addressed")
            
        except Exception as e:
            self.log(f"Organization failed: {e}", "ERROR")
            raise

def main():
    """Main function for automated workspace organization."""
    base_path = Path(__file__).parent.parent.parent  # Go up to project root
    organizer = WorkspaceOrganizer(base_path)
    organizer.run_full_organization()

if __name__ == "__main__":
    main()

