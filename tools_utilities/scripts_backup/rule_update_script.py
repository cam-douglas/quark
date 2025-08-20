#!/usr/bin/env python3
"""
Cursor Rules Update Script
Automatically keeps all cursor rules synchronized with new knowledge and developments.

This script runs continuously and updates rules when new knowledge is available.
It ensures all rules maintain compliance with cognitive_brain_roadmap.md and
maintains the supreme authority of compliance_review.md.

Author: Cursor Rules System
Priority: SUPREME (Priority 0)
"""

import os, sys
import time
import json
import yaml
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any
import subprocess
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.cursor/rules/rule_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RuleUpdateSystem:
    """Supreme authority system for updating and synchronizing all cursor rules."""
    
    def __init__(self):
        self.project_root = Path("/Users/camdouglas/quark")
        self.rules_dir = self.project_root / ".cursor" / "rules"
        self.compliance_review = self.rules_dir / "compliance_review.md"
        
        # Track file hashes for change detection
        self.file_hashes = {}
        self.rule_files = self._discover_rule_files()
        
        # Knowledge base for tracking new information
        self.knowledge_base = {
            "new_components": set(),
            "updated_parameters": set(),
            "new_connections": set(),
            "architectural_changes": set(),
            "compliance_issues": set()
        }
        
        # Supreme authority configuration
        self.supreme_authority = {
            "priority": 0,
            "can_override": True,
            "always_active": True,
            "compliance_enforcement": True
        }
        
        logger.info("Rule Update System initialized with SUPREME AUTHORITY")

    def _discover_rule_files(self) -> Dict[str, Path]:
        """Discover all rule files in the system."""
        rule_files = {}
        
        # Executive level rules
        rule_files["cognitive_brain_roadmap"] = self.rules_dir / "cognitive_brain_roadmap.md"
        rule_files["roles"] = self.rules_dir / "roles.md"
        
        # Management level rules
        rule_files["master_config"] = self.rules_dir / "master-config.mdc"
        rule_files["cognitive_brain_rules"] = self.rules_dir / "cognitive-brain-rules.mdc"
        
        # Operational level rules
        rule_files["brain_simulation_rules"] = self.rules_dir / "brain-simulation-rules.mdc"
        rule_files["omnirules"] = self.rules_dir / "omnirules.mdc"
        rule_files["braincomputer"] = self.rules_dir / "braincomputer.mdc"
        
        # Support level rules
        rule_files["technicalrules"] = self.rules_dir / "technicalrules.md"
        rule_files["terminal_rules"] = self.rules_dir / "terminal_rules.zsh"
        rule_files["explain"] = self.rules_dir / "explain.mdc"
        rule_files["integrated_rules"] = self.rules_dir / "integrated-rules.mdc"
        
        # Documentation
        rule_files["cursor_hierarchy"] = self.rules_dir / "cursor_hierarchy.md"
        rule_files["activation_triggers"] = self.rules_dir / "activation_triggers.md"
        rule_files["compliance_review"] = self.rules_dir / "compliance_review.md"
        
        # Configuration files
        rule_files["connectome"] = self.project_root / "connectome.yaml"
        
        return rule_files

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def detect_changes(self) -> Dict[str, List[str]]:
        """Detect changes in rule files and project files."""
        changes = {
            "modified_rules": [],
            "new_files": [],
            "deleted_files": [],
            "knowledge_updates": []
        }
        
        # Check rule files for changes
        for name, file_path in self.rule_files.items():
            if file_path.exists():
                current_hash = self.calculate_file_hash(file_path)
                if name not in self.file_hashes:
                    changes["new_files"].append(str(file_path))
                    self.file_hashes[name] = current_hash
                elif self.file_hashes[name] != current_hash:
                    changes["modified_rules"].append(str(file_path))
                    self.file_hashes[name] = current_hash
            else:
                if name in self.file_hashes:
                    changes["deleted_files"].append(str(file_path))
                    del self.file_hashes[name]
        
        # Check for new knowledge in project files
        changes["knowledge_updates"] = self._detect_knowledge_updates()
        
        return changes

    def _detect_knowledge_updates(self) -> List[str]:
        """Detect new knowledge in project files."""
        knowledge_updates = []
        
        # Check for new Python files with brain-related content
        for py_file in self.project_root.rglob("*.py"):
            if py_file.name not in ["rule_update_script.py"]:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if any(keyword in content.lower() for keyword in [
                            "brain", "neural", "cortex", "pfc", "thalamus", 
                            "hippocampus", "dmn", "architecture agent"
                        ]):
                            knowledge_updates.append(f"New brain-related code: {py_file}")
                except Exception as e:
                    logger.warning(f"Error reading {py_file}: {e}")
        
        # Check for new configuration files
        for config_file in self.project_root.rglob("*.yaml"):
            if config_file.name not in ["connectome.yaml"]:
                knowledge_updates.append(f"New configuration: {config_file}")
        
        # Check for new documentation
        for doc_file in self.project_root.rglob("*.md"):
            if doc_file.parent != self.rules_dir:
                knowledge_updates.append(f"New documentation: {doc_file}")
        
        return knowledge_updates

    def update_rule_with_knowledge(self, rule_file: Path, new_knowledge: Dict[str, Any]) -> bool:
        """Update a rule file with new knowledge."""
        try:
            with open(rule_file, 'r') as f:
                content = f.read()
            
            updated = False
            
            # Update component knowledge
            if "new_components" in new_knowledge:
                for component in new_knowledge["new_components"]:
                    if component not in content:
                        # Add new component to appropriate section
                        if "CORE COMPONENTS" in content:
                            component_section = f"\n### {component}\n**Function**: [To be defined]\n**Connections**: [To be defined]\n**Implementation**: [To be defined]"
                            content = content.replace("## CORE COMPONENTS", f"## CORE COMPONENTS{component_section}")
                            updated = True
            
            # Update parameter knowledge
            if "updated_parameters" in new_knowledge:
                for param in new_knowledge["updated_parameters"]:
                    if param not in content:
                        # Add new parameter to appropriate section
                        if "PARAMETERS" in content or "CONFIGURATION" in content:
                            param_section = f"\n- **{param}**: [To be defined]"
                            content = content.replace("## CONFIGURATION", f"## CONFIGURATION{param_section}")
                            updated = True
            
            # Update connection knowledge
            if "new_connections" in new_knowledge:
                for connection in new_knowledge["new_connections"]:
                    if connection not in content:
                        # Add new connection to appropriate section
                        if "CONNECTIONS" in content or "WIRING" in content:
                            conn_section = f"\n- **{connection}**: [To be defined]"
                            content = content.replace("## CONNECTIONS", f"## CONNECTIONS{conn_section}")
                            updated = True
            
            if updated:
                # Add update timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                update_note = f"\n\n<!-- Updated by Rule Update System: {timestamp} -->"
                content += update_note
                
                with open(rule_file, 'w') as f:
                    f.write(content)
                
                logger.info(f"Updated {rule_file} with new knowledge")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating {rule_file}: {e}")
            return False

    def enforce_compliance(self) -> List[str]:
        """Enforce compliance across all rule sets."""
        compliance_issues = []
        
        # Check that all rules acknowledge supreme authority
        for name, file_path in self.rule_files.items():
            if file_path.exists() and name != "compliance_review":
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for supreme authority acknowledgment
                    if "compliance_review.md supreme authority" not in content:
                        compliance_issues.append(f"{name}: Missing supreme authority acknowledgment")
                    
                    # Check for proper hierarchy level
                    if "Hierarchy Level" not in content:
                        compliance_issues.append(f"{name}: Missing hierarchy level definition")
                    
                    # Check for activation triggers
                    if "Activation Triggers" not in content:
                        compliance_issues.append(f"{name}: Missing activation triggers")
                        
                except Exception as e:
                    compliance_issues.append(f"{name}: Error reading file - {e}")
        
        return compliance_issues

    def update_all_rules(self, new_knowledge: Dict[str, Any]) -> Dict[str, List[str]]:
        """Update all rule files with new knowledge."""
        update_results = {
            "updated": [],
            "failed": [],
            "skipped": []
        }
        
        for name, file_path in self.rule_files.items():
            if file_path.exists() and name != "compliance_review":
                try:
                    if self.update_rule_with_knowledge(file_path, new_knowledge):
                        update_results["updated"].append(name)
                    else:
                        update_results["skipped"].append(name)
                except Exception as e:
                    update_results["failed"].append(f"{name}: {e}")
                    logger.error(f"Failed to update {name}: {e}")
        
        return update_results

    def broadcast_knowledge_update(self, new_knowledge: Dict[str, Any]):
        """Broadcast knowledge updates to all components."""
        # Update knowledge base
        for key, value in new_knowledge.items():
            if key in self.knowledge_base:
                if isinstance(value, set):
                    self.knowledge_base[key].update(value)
                else:
                    self.knowledge_base[key].add(value)
        
        # Log the update
        logger.info(f"Knowledge base updated: {new_knowledge}")
        
        # Update all rules
        update_results = self.update_all_rules(new_knowledge)
        
        # Log results
        if update_results["updated"]:
            logger.info(f"Updated rules: {update_results['updated']}")
        if update_results["failed"]:
            logger.error(f"Failed updates: {update_results['failed']}")

    def run_compliance_audit(self):
        """Run a comprehensive compliance audit."""
        logger.info("Running compliance audit...")
        
        # Check compliance
        compliance_issues = self.enforce_compliance()
        
        if compliance_issues:
            logger.warning(f"Compliance issues found: {compliance_issues}")
            
            # Auto-fix compliance issues
            for issue in compliance_issues:
                self._auto_fix_compliance_issue(issue)
        else:
            logger.info("All rules are compliant")

    def _auto_fix_compliance_issue(self, issue: str):
        """Automatically fix compliance issues."""
        try:
            if "Missing supreme authority acknowledgment" in issue:
                rule_name = issue.split(":")[0]
                file_path = self.rule_files.get(rule_name)
                if file_path and file_path.exists():
                    self._add_supreme_authority_acknowledgment(file_path)
                    logger.info(f"Auto-fixed supreme authority acknowledgment in {rule_name}")
            
            elif "Missing hierarchy level definition" in issue:
                rule_name = issue.split(":")[0]
                file_path = self.rule_files.get(rule_name)
                if file_path and file_path.exists():
                    self._add_hierarchy_level(file_path)
                    logger.info(f"Auto-fixed hierarchy level in {rule_name}")
                    
        except Exception as e:
            logger.error(f"Error auto-fixing compliance issue {issue}: {e}")

    def _add_supreme_authority_acknowledgment(self, file_path: Path):
        """Add supreme authority acknowledgment to a rule file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find the authority line and update it
            if "**Authority**:" in content:
                content = content.replace(
                    "**Authority**:",
                    "**Authority**: (subject to compliance_review.md supreme authority)"
                )
                
                # Add supreme authority line
                if "**Supreme Authority**:" not in content:
                    content = content.replace(
                        "**Activation Triggers**:",
                        "**Activation Triggers**:\n- **Supreme Authority**: .cursor/rules/compliance_review.md (Priority 0) can override any decisions"
                    )
            
            with open(file_path, 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"Error adding supreme authority acknowledgment to {file_path}: {e}")

    def _add_hierarchy_level(self, file_path: Path):
        """Add hierarchy level to a rule file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Determine hierarchy level based on file name
            if "cognitive_brain_roadmap" in str(file_path):
                hierarchy_level = "EXECUTIVE (CEO)"
            elif "roles" in str(file_path):
                hierarchy_level = "EXECUTIVE (COO)"
            elif "master-config" in str(file_path):
                hierarchy_level = "MANAGEMENT (Senior Manager)"
            elif "cognitive-brain-rules" in str(file_path):
                hierarchy_level = "MANAGEMENT (Implementation Manager)"
            elif "brain-simulation" in str(file_path):
                hierarchy_level = "OPERATIONAL (Specialized Operations)"
            elif "omnirules" in str(file_path):
                hierarchy_level = "OPERATIONAL (General Development Team)"
            else:
                hierarchy_level = "SUPPORT (Technical Support)"
            
            # Add hierarchy level if missing
            if "**Hierarchy Level**:" not in content:
                content = content.replace(
                    "## ",
                    f"## 🏛️ RULE SET STATUS - {hierarchy_level.split('(')[0].strip()}\n- **Hierarchy Level**: {hierarchy_level}\n- **Activation**: 🟢 ALWAYS ACTIVE\n- **Reports to**: [To be defined]\n- **Direct Reports**: [To be defined]\n- **Authority**: [To be defined]\n- **Activation Triggers**: [To be defined]\n\n## "
                )
            
            with open(file_path, 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"Error adding hierarchy level to {file_path}: {e}")

    def start_monitoring(self):
        """Start continuous monitoring of the system."""
        logger.info("Starting continuous rule monitoring...")
        
        # Set up file system monitoring
        event_handler = RuleFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.project_root), recursive=True)
        observer.start()
        
        try:
            while True:
                # Check for changes every 30 seconds
                changes = self.detect_changes()
                
                if any(changes.values()):
                    logger.info(f"Changes detected: {changes}")
                    
                    # Update knowledge base
                    if changes["knowledge_updates"]:
                        new_knowledge = {
                            "new_components": set(),
                            "updated_parameters": set(),
                            "new_connections": set(),
                            "architectural_changes": set()
                        }
                        
                        for update in changes["knowledge_updates"]:
                            if "brain-related code" in update:
                                new_knowledge["new_components"].add("New brain component")
                            elif "configuration" in update:
                                new_knowledge["updated_parameters"].add("New configuration parameter")
                            elif "documentation" in update:
                                new_knowledge["architectural_changes"].add("New architectural information")
                        
                        self.broadcast_knowledge_update(new_knowledge)
                    
                    # Run compliance audit
                    self.run_compliance_audit()
                
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("Stopping rule monitoring...")
            observer.stop()
        
        observer.join()

class RuleFileHandler(FileSystemEventHandler):
    """Handle file system events for rule updates."""
    
    def __init__(self, rule_system: RuleUpdateSystem):
        self.rule_system = rule_system
    
    def on_modified(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix in ['.py', '.md', '.yaml', '.yml', '.json']:
                logger.info(f"File modified: {file_path}")
                # Trigger immediate update check
                self.rule_system.run_compliance_audit()
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix in ['.py', '.md', '.yaml', '.yml', '.json']:
                logger.info(f"New file created: {file_path}")
                # Update rule files with new knowledge
                new_knowledge = {
                    "new_components": {f"New component from {file_path.name}"},
                    "updated_parameters": set(),
                    "new_connections": set(),
                    "architectural_changes": set()
                }
                self.rule_system.broadcast_knowledge_update(new_knowledge)

def main():
    """Main entry point for the rule update system."""
    print("🚀 Starting Cursor Rules Update System with SUPREME AUTHORITY")
    print("Priority Level: 0 (Supreme - Above All Others)")
    print("Status: Always Active - Maximum Priority")
    print("Authority: Can override, veto, or modify any rule set or component")
    print("=" * 80)
    
    # Initialize the rule update system
    rule_system = RuleUpdateSystem()
    
    # Run initial compliance audit
    rule_system.run_compliance_audit()
    
    # Start continuous monitoring
    rule_system.start_monitoring()

if __name__ == "__main__":
    main()
