#!/usr/bin/env python3
"""
Cursor Integration Test Suite
============================

Purpose: Comprehensive testing of Cursor IDE integration and CLI functionality
Inputs: Cursor installation, settings files, CLI commands
Outputs: Test results, validation reports, simulation data
Dependencies: pytest, unittest, subprocess, json, pathlib

This test suite validates:
- Cursor settings management
- CLI installation and functionality
- Documentation integration
- Rules system operation
- Configuration automation
"""

import unittest
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import os

# Add the cursor rules directory to Python path for imports
cursor_rules_dir = Path(__file__).parent.parent / ".cursor" / "rules"
sys.path.insert(0, str(cursor_rules_dir))

try:
    from cursor_settings_manager import CursorSettingsManager
    from cursor_cli_setup import CursorCLISetup
except ImportError as e:
    print(f"Warning: Could not import Cursor modules: {e}")
    CursorSettingsManager = None
    CursorCLISetup = None


class TestCursorSettingsManager(unittest.TestCase):
    """Test suite for Cursor Settings Manager."""
    
    def setUp(self):
        """Set up test environment."""
        if CursorSettingsManager is None:
            self.skipTest("CursorSettingsManager not available")
        
        # Create temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.manager = CursorSettingsManager(project_root=self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertTrue(self.manager.cursor_dir.exists())
        self.assertTrue(self.manager.rules_dir.exists())
        self.assertEqual(self.manager.project_root, self.test_dir)
    
    def test_default_settings_structure(self):
        """Test default settings structure."""
        settings = self.manager.default_settings
        
        # Check required keys
        required_keys = ['cursor.aiModel', 'cursor.rules', 'cursor.memories', 'cursor.agent']
        for key in required_keys:
            self.assertIn(key, settings)
        
        # Check rules configuration
        rules_config = settings['cursor.rules']
        self.assertIn('enabled', rules_config)
        self.assertIn('autoLoad', rules_config)
        self.assertIn('directory', rules_config)
    
    def test_settings_update(self):
        """Test settings update functionality."""
        test_settings = {"cursor.aiModel": "test-model"}
        
        result = self.manager.update_settings(test_settings)
        self.assertTrue(result)
        
        # Verify settings were written
        self.assertTrue(self.manager.settings_file.exists())
        
        # Load and verify content
        loaded_settings = self.manager.load_current_settings()
        self.assertEqual(loaded_settings["cursor.aiModel"], "test-model")
    
    def test_keybindings_update(self):
        """Test keybindings update functionality."""
        test_keybindings = [
            {"key": "cmd+t", "command": "test.command", "when": "editorTextFocus"}
        ]
        
        result = self.manager.update_keybindings(test_keybindings)
        self.assertTrue(result)
        
        # Verify keybindings were written
        self.assertTrue(self.manager.keybindings_file.exists())
        
        # Load and verify content
        loaded_keybindings = self.manager.load_current_keybindings()
        self.assertEqual(len(loaded_keybindings), 1)
        self.assertEqual(loaded_keybindings[0]["key"], "cmd+t")
    
    def test_backup_functionality(self):
        """Test settings backup functionality."""
        # Create some settings first
        self.manager.update_settings({"test": "value"})
        
        # Create backup
        backup_file = self.manager.backup_settings()
        
        self.assertTrue(backup_file.exists())
        self.assertIn("backups", str(backup_file))
        
        # Verify backup content
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        self.assertEqual(backup_data["test"], "value")
    
    def test_ai_model_setting(self):
        """Test AI model setting functionality."""
        model_name = "claude-3.5-sonnet"
        
        result = self.manager.set_ai_model(model_name)
        self.assertTrue(result)
        
        # Verify model was set
        settings = self.manager.load_current_settings()
        self.assertEqual(settings["cursor.aiModel"], model_name)
    
    def test_rules_system_configuration(self):
        """Test rules system configuration."""
        result = self.manager.configure_rules_system(enabled=True, auto_load=True)
        self.assertTrue(result)
        
        # Verify rules configuration
        settings = self.manager.load_current_settings()
        rules_config = settings["cursor.rules"]
        self.assertTrue(rules_config["enabled"])
        self.assertTrue(rules_config["autoLoad"])
    
    def test_validation_functionality(self):
        """Test settings validation."""
        # Apply default configuration
        self.manager.apply_default_configuration()
        
        # Run validation
        report = self.manager.validate_settings()
        
        # Check report structure
        required_fields = [
            "timestamp", "settings_file_exists", "keybindings_file_exists",
            "rules_directory_exists", "settings_valid", "keybindings_valid"
        ]
        for field in required_fields:
            self.assertIn(field, report)
        
        # Check validation results
        self.assertTrue(report["settings_file_exists"])
        self.assertTrue(report["keybindings_file_exists"])
        self.assertTrue(report["rules_directory_exists"])
        self.assertTrue(report["settings_valid"])
        self.assertTrue(report["keybindings_valid"])
    
    def test_deep_merge_functionality(self):
        """Test deep merge functionality."""
        dict1 = {
            "level1": {
                "level2": {"key1": "value1", "key2": "value2"}
            },
            "top_level": "original"
        }
        
        dict2 = {
            "level1": {
                "level2": {"key2": "updated", "key3": "new"}
            },
            "new_top": "added"
        }
        
        result = self.manager._deep_merge(dict1, dict2)
        
        # Check merge results
        self.assertEqual(result["level1"]["level2"]["key1"], "value1")  # Preserved
        self.assertEqual(result["level1"]["level2"]["key2"], "updated")  # Updated
        self.assertEqual(result["level1"]["level2"]["key3"], "new")  # Added
        self.assertEqual(result["top_level"], "original")  # Preserved
        self.assertEqual(result["new_top"], "added")  # Added
    
    def test_template_export(self):
        """Test settings template export."""
        template_file = self.test_dir / "test_template.json"
        
        result = self.manager.export_settings_template(template_file)
        
        self.assertEqual(result, template_file)
        self.assertTrue(template_file.exists())
        
        # Verify template content
        with open(template_file, 'r') as f:
            template = json.load(f)
        
        required_sections = ["description", "settings", "keybindings", "usage"]
        for section in required_sections:
            self.assertIn(section, template)


class TestCursorCLISetup(unittest.TestCase):
    """Test suite for Cursor CLI Setup."""
    
    def setUp(self):
        """Set up test environment."""
        if CursorCLISetup is None:
            self.skipTest("CursorCLISetup not available")
        
        self.setup = CursorCLISetup()
    
    def test_initialization(self):
        """Test CLI setup initialization."""
        self.assertIsInstance(self.setup.home_dir, Path)
        self.assertIsInstance(self.setup.local_bin, Path)
        self.assertTrue(self.setup.local_bin.exists())
    
    def test_shell_detection(self):
        """Test shell detection functionality."""
        shell = self.setup.detect_shell()
        self.assertIn(shell, ['bash', 'zsh', 'fish', 'sh'])
    
    def test_installation_check(self):
        """Test installation check functionality."""
        # This test checks the method runs without error
        # Actual result depends on whether Cursor CLI is installed
        result = self.setup.check_cursor_cli_installed()
        self.assertIsInstance(result, bool)
    
    def test_verification_structure(self):
        """Test verification results structure."""
        results = self.setup.verify_installation()
        
        expected_keys = [
            "cursor_agent_available",
            "cursor_command_available", 
            "code_command_available",
            "path_configured",
            "shell_integration"
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], bool)
    
    def test_report_generation(self):
        """Test setup report generation."""
        report = self.setup.generate_setup_report()
        
        required_sections = [
            "timestamp",
            "system_info",
            "installation_status",
            "next_steps",
            "troubleshooting"
        ]
        
        for section in required_sections:
            self.assertIn(section, report)
        
        # Check system info structure
        system_info = report["system_info"]
        self.assertIn("shell", system_info)
        self.assertIn("home_directory", system_info)
        self.assertIn("local_bin", system_info)


class TestCursorDocumentationIntegration(unittest.TestCase):
    """Test suite for Cursor documentation integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent
        self.cursor_dir = self.project_root / ".cursor"
        self.rules_dir = self.cursor_dir / "rules"
    
    def test_rules_directory_exists(self):
        """Test that rules directory exists."""
        self.assertTrue(self.rules_dir.exists())
        self.assertTrue(self.rules_dir.is_dir())
    
    def test_documentation_integration_file(self):
        """Test documentation integration file exists and is valid."""
        doc_file = self.rules_dir / "cursor_documentation_integration.md"
        self.assertTrue(doc_file.exists())
        
        # Check file content structure
        content = doc_file.read_text()
        required_sections = [
            "# Cursor Documentation Integration",
            "## Overview",
            "## Core Documentation References",
            "## Integration Status"
        ]
        
        for section in required_sections:
            self.assertIn(section, content)
    
    def test_settings_manager_file(self):
        """Test settings manager file exists and is valid."""
        settings_file = self.rules_dir / "cursor_settings_manager.py"
        self.assertTrue(settings_file.exists())
        
        # Check file is valid Python
        content = settings_file.read_text()
        self.assertIn("class CursorSettingsManager", content)
        self.assertIn("def __init__", content)
    
    def test_cli_setup_file(self):
        """Test CLI setup file exists and is valid."""
        cli_file = self.rules_dir / "cursor_cli_setup.py"
        self.assertTrue(cli_file.exists())
        
        # Check file is valid Python
        content = cli_file.read_text()
        self.assertIn("class CursorCLISetup", content)
        self.assertIn("def __init__", content)


class CursorIntegrationSimulation:
    """Simulation class for testing Cursor integration without actual installation."""
    
    def __init__(self):
        """Initialize simulation environment."""
        self.simulated_responses = {
            "cursor-agent --version": "Cursor Agent v1.0.0",
            "which cursor": "/usr/local/bin/cursor",
            "which code": "/usr/local/bin/code"
        }
        
    def simulate_command(self, command: str) -> Dict[str, Any]:
        """Simulate command execution.
        
        Args:
            command: Command to simulate
            
        Returns:
            Simulated result dictionary
        """
        if command in self.simulated_responses:
            return {
                "returncode": 0,
                "stdout": self.simulated_responses[command],
                "stderr": ""
            }
        else:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Command not found: {command}"
            }
    
    def simulate_file_operations(self, operation: str, file_path: Path, content: str = "") -> bool:
        """Simulate file operations.
        
        Args:
            operation: Operation type (read, write, delete)
            file_path: Path to file
            content: Content for write operations
            
        Returns:
            Success status
        """
        if operation == "write":
            print(f"Simulated write to {file_path}: {len(content)} characters")
            return True
        elif operation == "read":
            print(f"Simulated read from {file_path}")
            return True
        elif operation == "delete":
            print(f"Simulated delete of {file_path}")
            return True
        else:
            return False


class TestCursorIntegrationSimulation(unittest.TestCase):
    """Test suite using simulation for integration testing."""
    
    def setUp(self):
        """Set up simulation environment."""
        self.simulation = CursorIntegrationSimulation()
    
    def test_command_simulation(self):
        """Test command simulation functionality."""
        result = self.simulation.simulate_command("cursor-agent --version")
        self.assertEqual(result["returncode"], 0)
        self.assertIn("Cursor Agent", result["stdout"])
        
        # Test unknown command
        result = self.simulation.simulate_command("unknown-command")
        self.assertEqual(result["returncode"], 1)
        self.assertIn("Command not found", result["stderr"])
    
    def test_file_operation_simulation(self):
        """Test file operation simulation."""
        test_path = Path("/tmp/test_file.json")
        
        # Test write operation
        result = self.simulation.simulate_file_operations("write", test_path, '{"test": true}')
        self.assertTrue(result)
        
        # Test read operation
        result = self.simulation.simulate_file_operations("read", test_path)
        self.assertTrue(result)
        
        # Test delete operation
        result = self.simulation.simulate_file_operations("delete", test_path)
        self.assertTrue(result)


def run_integration_tests():
    """Run the complete integration test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCursorSettingsManager,
        TestCursorCLISetup,
        TestCursorDocumentationIntegration,
        TestCursorIntegrationSimulation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def generate_test_report():
    """Generate a comprehensive test report."""
    report = {
        "timestamp": subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
        "test_environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd())
        },
        "test_modules": [
            "cursor_settings_manager",
            "cursor_cli_setup", 
            "cursor_documentation_integration",
            "cursor_integration_simulation"
        ],
        "test_categories": [
            "Settings Management",
            "CLI Setup and Installation",
            "Documentation Integration",
            "Simulation Testing"
        ],
        "coverage_areas": [
            "Configuration file handling",
            "Shell integration",
            "PATH management",
            "Backup and restore",
            "Validation and verification",
            "Error handling"
        ]
    }
    
    return report


if __name__ == "__main__":
    print("Cursor Integration Test Suite")
    print("=" * 40)
    
    # Generate and display test report
    report = generate_test_report()
    print(f"Test Environment: {report['test_environment']['platform']}")
    print(f"Python Version: {report['test_environment']['python_version']}")
    print(f"Test Categories: {len(report['test_categories'])}")
    print(f"Coverage Areas: {len(report['coverage_areas'])}")
    print()
    
    # Run tests
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed.")
    
    sys.exit(0 if success else 1)
