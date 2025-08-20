#!/usr/bin/env python3
"""
Cursor Settings Manager
=======================

Purpose: Programmatically manage Cursor IDE settings and configuration
Inputs: Settings dictionaries, configuration files, user preferences
Outputs: Updated Cursor configuration files, settings validation reports
Dependencies: json, os, pathlib, typing

This module provides automated management of Cursor IDE settings including:
- Keyboard shortcuts configuration
- Theme and appearance settings
- Model selection and API key management
- Rules and memory system configuration
- CLI and shell integration settings
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import subprocess
import sys
from datetime import datetime


class CursorSettingsManager:
    """Manages Cursor IDE settings programmatically."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the settings manager.
        
        Args:
            project_root: Root directory of the project (defaults to current directory)
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.cursor_dir = self.project_root / ".cursor"
        self.rules_dir = self.cursor_dir / "rules"
        self.settings_file = self.cursor_dir / "settings.json"
        self.keybindings_file = self.cursor_dir / "keybindings.json"
        self.workspace_file = self.project_root / ".vscode" / "settings.json"
        
        # Ensure directories exist
        self.cursor_dir.mkdir(exist_ok=True)
        self.rules_dir.mkdir(exist_ok=True)
        (self.project_root / ".vscode").mkdir(exist_ok=True)
        
        # Initialize settings structure
        self.default_settings = {
            "cursor.aiModel": "claude-3.5-sonnet",
            "cursor.rules": {
                "enabled": True,
                "autoLoad": True,
                "directory": str(self.rules_dir)
            },
            "cursor.memories": {
                "enabled": True,
                "autoSave": True
            },
            "cursor.agent": {
                "backgroundEnabled": True,
                "webMobileEnabled": True
            },
            "cursor.codebase": {
                "indexingEnabled": True,
                "ignorePatterns": [
                    "node_modules/**",
                    "venv/**",
                    "__pycache__/**",
                    "*.pyc",
                    ".git/**",
                    "cache/**",
                    "dist/**",
                    "build/**"
                ]
            },
            "cursor.integrations": {
                "github": True,
                "slack": False,
                "bugbot": True
            }
        }
        
        self.default_keybindings = [
            {
                "key": "cmd+k",
                "command": "cursor.askAI",
                "when": "editorTextFocus"
            },
            {
                "key": "cmd+shift+k",
                "command": "cursor.agent.start",
                "when": "editorTextFocus"
            },
            {
                "key": "cmd+i",
                "command": "cursor.inlineEdit",
                "when": "editorTextFocus"
            }
        ]

    def load_current_settings(self) -> Dict[str, Any]:
        """Load current Cursor settings.
        
        Returns:
            Current settings dictionary
        """
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {self.settings_file}")
                return {}
        return {}

    def load_current_keybindings(self) -> List[Dict[str, Any]]:
        """Load current keybindings.
        
        Returns:
            Current keybindings list
        """
        if self.keybindings_file.exists():
            try:
                with open(self.keybindings_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {self.keybindings_file}")
                return []
        return []

    def backup_settings(self) -> Path:
        """Create a backup of current settings.
        
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.cursor_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_file = backup_dir / f"settings_backup_{timestamp}.json"
        
        current_settings = self.load_current_settings()
        with open(backup_file, 'w') as f:
            json.dump(current_settings, f, indent=2)
        
        print(f"Settings backed up to: {backup_file}")
        return backup_file

    def update_settings(self, new_settings: Dict[str, Any], merge: bool = True) -> bool:
        """Update Cursor settings.
        
        Args:
            new_settings: New settings to apply
            merge: Whether to merge with existing settings or replace
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if merge:
                current_settings = self.load_current_settings()
                # Deep merge settings
                updated_settings = self._deep_merge(current_settings, new_settings)
            else:
                updated_settings = new_settings
            
            # Create backup before updating
            self.backup_settings()
            
            # Write updated settings
            with open(self.settings_file, 'w') as f:
                json.dump(updated_settings, f, indent=2)
            
            print(f"Settings updated successfully: {self.settings_file}")
            return True
            
        except Exception as e:
            print(f"Error updating settings: {e}")
            return False

    def update_keybindings(self, new_keybindings: List[Dict[str, Any]], merge: bool = True) -> bool:
        """Update keybindings.
        
        Args:
            new_keybindings: New keybindings to apply
            merge: Whether to merge with existing keybindings or replace
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if merge:
                current_keybindings = self.load_current_keybindings()
                # Merge keybindings (new ones override existing with same key)
                key_map = {kb.get('key'): kb for kb in current_keybindings}
                for kb in new_keybindings:
                    key_map[kb.get('key')] = kb
                updated_keybindings = list(key_map.values())
            else:
                updated_keybindings = new_keybindings
            
            with open(self.keybindings_file, 'w') as f:
                json.dump(updated_keybindings, f, indent=2)
            
            print(f"Keybindings updated successfully: {self.keybindings_file}")
            return True
            
        except Exception as e:
            print(f"Error updating keybindings: {e}")
            return False

    def set_ai_model(self, model_name: str) -> bool:
        """Set the AI model for Cursor.
        
        Args:
            model_name: Name of the AI model to use
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_settings({"cursor.aiModel": model_name})

    def configure_rules_system(self, enabled: bool = True, auto_load: bool = True) -> bool:
        """Configure the rules system.
        
        Args:
            enabled: Whether to enable the rules system
            auto_load: Whether to auto-load rules
            
        Returns:
            True if successful, False otherwise
        """
        rules_config = {
            "cursor.rules": {
                "enabled": enabled,
                "autoLoad": auto_load,
                "directory": str(self.rules_dir)
            }
        }
        return self.update_settings(rules_config)

    def apply_default_configuration(self) -> bool:
        """Apply the default Cursor configuration.
        
        Returns:
            True if successful, False otherwise
        """
        print("Applying default Cursor configuration...")
        
        success = True
        success &= self.update_settings(self.default_settings, merge=False)
        success &= self.update_keybindings(self.default_keybindings, merge=False)
        
        if success:
            print("Default configuration applied successfully")
        else:
            print("Failed to apply some configuration settings")
        
        return success

    def validate_settings(self) -> Dict[str, Any]:
        """Validate current settings.
        
        Returns:
            Validation report
        """
        current_settings = self.load_current_settings()
        current_keybindings = self.load_current_keybindings()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "settings_file_exists": self.settings_file.exists(),
            "keybindings_file_exists": self.keybindings_file.exists(),
            "rules_directory_exists": self.rules_dir.exists(),
            "settings_valid": bool(current_settings),
            "keybindings_valid": isinstance(current_keybindings, list),
            "ai_model_configured": "cursor.aiModel" in current_settings,
            "rules_enabled": current_settings.get("cursor.rules", {}).get("enabled", False),
            "current_ai_model": current_settings.get("cursor.aiModel", "unknown"),
            "rule_count": len(list(self.rules_dir.glob("*.md"))) if self.rules_dir.exists() else 0,
            "issues": []
        }
        
        # Check for issues
        if not report["settings_valid"]:
            report["issues"].append("Settings file is invalid or empty")
        
        if not report["keybindings_valid"]:
            report["issues"].append("Keybindings file is invalid")
        
        if not report["rules_enabled"]:
            report["issues"].append("Rules system is not enabled")
        
        return report

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            dict1: Base dictionary
            dict2: Dictionary to merge into base
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def export_settings_template(self, output_file: Optional[Path] = None) -> Path:
        """Export a settings template file.
        
        Args:
            output_file: Output file path (defaults to cursor_settings_template.json)
            
        Returns:
            Path to exported template
        """
        if output_file is None:
            output_file = self.cursor_dir / "cursor_settings_template.json"
        
        template = {
            "description": "Cursor Settings Template",
            "settings": self.default_settings,
            "keybindings": self.default_keybindings,
            "usage": {
                "apply_settings": "settings_manager.update_settings(template['settings'])",
                "apply_keybindings": "settings_manager.update_keybindings(template['keybindings'])",
                "set_model": "settings_manager.set_ai_model('claude-3.5-sonnet')"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Settings template exported to: {output_file}")
        return output_file


def main():
    """Main function for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python cursor_settings_manager.py [command] [args...]")
        print("Commands:")
        print("  init                    - Initialize default settings")
        print("  validate               - Validate current settings")
        print("  set-model <model>      - Set AI model")
        print("  backup                 - Backup current settings")
        print("  export-template        - Export settings template")
        return
    
    manager = CursorSettingsManager()
    command = sys.argv[1]
    
    if command == "init":
        manager.apply_default_configuration()
    elif command == "validate":
        report = manager.validate_settings()
        print(json.dumps(report, indent=2))
    elif command == "set-model" and len(sys.argv) > 2:
        model = sys.argv[2]
        manager.set_ai_model(model)
    elif command == "backup":
        manager.backup_settings()
    elif command == "export-template":
        manager.export_settings_template()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
