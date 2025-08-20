#!/usr/bin/env python3
"""
Cursor Integration Setup Script
==============================

Purpose: Main orchestrator for complete Cursor IDE integration setup
Inputs: User preferences, system configuration, installation options
Outputs: Fully configured Cursor environment with CLI, settings, and docs
Dependencies: cursor_settings_manager, cursor_cli_setup, pathlib, subprocess

This script orchestrates:
- Documentation integration into rules directory
- Programmatic settings management setup
- CLI installation and configuration
- Test suite execution and validation
- Summary reporting and next steps
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from cursor_settings_manager import CursorSettingsManager
    from cursor_cli_setup import CursorCLISetup
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure cursor_settings_manager.py and cursor_cli_setup.py are in the same directory.")
    sys.exit(1)


class CursorIntegrationOrchestrator:
    """Main orchestrator for Cursor integration setup."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the orchestrator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        while not (self.project_root / ".cursor").exists() and self.project_root != self.project_root.parent:
            self.project_root = self.project_root.parent
        
        self.cursor_dir = self.project_root / ".cursor"
        self.rules_dir = self.cursor_dir / "rules"
        self.tests_dir = self.project_root / "tests"
        
        # Initialize managers
        self.settings_manager = CursorSettingsManager(self.project_root)
        self.cli_setup = CursorCLISetup()
        
        # Setup state tracking
        self.setup_state = {
            "documentation_integrated": False,
            "settings_configured": False,
            "cli_installed": False,
            "tests_created": False,
            "validation_completed": False
        }
        
        print(f"Cursor Integration Orchestrator initialized")
        print(f"Project Root: {self.project_root}")
        print(f"Cursor Directory: {self.cursor_dir}")
    
    def integrate_documentation(self) -> bool:
        """Integrate Cursor documentation into rules directory.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n📚 Integrating Cursor Documentation...")
        
        try:
            # Check if documentation integration file exists
            doc_file = self.rules_dir / "cursor_documentation_integration.md"
            if doc_file.exists():
                print(f"✓ Documentation integration file exists: {doc_file}")
                self.setup_state["documentation_integrated"] = True
                return True
            else:
                print(f"✗ Documentation integration file missing: {doc_file}")
                return False
                
        except Exception as e:
            print(f"Error integrating documentation: {e}")
            return False
    
    def configure_settings(self) -> bool:
        """Configure Cursor settings with optimal defaults.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n⚙️ Configuring Cursor Settings...")
        
        try:
            # Apply default configuration
            if self.settings_manager.apply_default_configuration():
                print("✓ Default settings applied successfully")
                
                # Configure for brain simulation project
                brain_settings = {
                    "cursor.aiModel": "claude-3.5-sonnet",
                    "cursor.rules": {
                        "enabled": True,
                        "autoLoad": True,
                        "directory": str(self.rules_dir)
                    },
                    "cursor.codebase": {
                        "indexingEnabled": True,
                        "ignorePatterns": [
                            "node_modules/**",
                            "venv/**", 
                            "wikipedia_env/**",
                            "__pycache__/**",
                            "*.pyc",
                            ".git/**",
                            "cache/**",
                            "dist/**",
                            "build/**",
                            "logs/**",
                            "backups/**"
                        ]
                    },
                    "cursor.project": {
                        "type": "brain_simulation",
                        "architecture": "multi_agent_neural",
                        "domain": "computational_neuroscience"
                    }
                }
                
                if self.settings_manager.update_settings(brain_settings):
                    print("✓ Brain simulation project settings applied")
                    self.setup_state["settings_configured"] = True
                    return True
                else:
                    print("✗ Failed to apply brain simulation settings")
                    return False
            else:
                print("✗ Failed to apply default settings")
                return False
                
        except Exception as e:
            print(f"Error configuring settings: {e}")
            return False
    
    def install_cli(self) -> bool:
        """Install and configure Cursor CLI.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n🖥️ Installing Cursor CLI...")
        
        try:
            # Check if already installed
            if self.cli_setup.check_cursor_cli_installed():
                print("✓ Cursor CLI already installed")
                self.setup_state["cli_installed"] = True
                return True
            
            # Perform installation
            if self.cli_setup.setup_complete_cli():
                print("✓ Cursor CLI installation completed")
                self.setup_state["cli_installed"] = True
                return True
            else:
                print("✗ Cursor CLI installation failed")
                return False
                
        except Exception as e:
            print(f"Error installing CLI: {e}")
            return False
    
    def create_test_suite(self) -> bool:
        """Create and validate test suite.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n🧪 Creating Test Suite...")
        
        try:
            # Check if test file exists
            test_file = self.tests_dir / "cursor_integration_tests.py"
            if test_file.exists():
                print(f"✓ Test suite exists: {test_file}")
                
                # Try to run a basic import test
                try:
                    result = subprocess.run([
                        sys.executable, "-c", 
                        f"import sys; sys.path.insert(0, '{test_file.parent}'); "
                        "from cursor_integration_tests import generate_test_report; "
                        "print('Test suite import successful')"
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print("✓ Test suite validation passed")
                        self.setup_state["tests_created"] = True
                        return True
                    else:
                        print(f"✗ Test suite validation failed: {result.stderr}")
                        return False
                        
                except subprocess.TimeoutExpired:
                    print("✗ Test suite validation timed out")
                    return False
            else:
                print(f"✗ Test suite missing: {test_file}")
                return False
                
        except Exception as e:
            print(f"Error creating test suite: {e}")
            return False
    
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the setup.
        
        Returns:
            Validation results dictionary
        """
        print("\n✅ Running Validation...")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_success": False,
            "components": {}
        }
        
        try:
            # Validate settings
            print("Validating settings...")
            settings_report = self.settings_manager.validate_settings()
            validation_results["components"]["settings"] = settings_report
            print(f"Settings validation: {'✓' if settings_report['settings_valid'] else '✗'}")
            
            # Validate CLI
            print("Validating CLI...")
            cli_report = self.cli_setup.verify_installation()
            validation_results["components"]["cli"] = cli_report
            cli_success = cli_report.get("shell_integration", False)
            print(f"CLI validation: {'✓' if cli_success else '✗'}")
            
            # Validate documentation
            print("Validating documentation...")
            doc_file = self.rules_dir / "cursor_documentation_integration.md"
            doc_valid = doc_file.exists() and doc_file.stat().st_size > 0
            validation_results["components"]["documentation"] = {
                "file_exists": doc_file.exists(),
                "file_size": doc_file.stat().st_size if doc_file.exists() else 0,
                "valid": doc_valid
            }
            print(f"Documentation validation: {'✓' if doc_valid else '✗'}")
            
            # Validate test suite
            print("Validating test suite...")
            test_file = self.tests_dir / "cursor_integration_tests.py"
            test_valid = test_file.exists() and test_file.stat().st_size > 0
            validation_results["components"]["tests"] = {
                "file_exists": test_file.exists(),
                "file_size": test_file.stat().st_size if test_file.exists() else 0,
                "valid": test_valid
            }
            print(f"Test suite validation: {'✓' if test_valid else '✗'}")
            
            # Overall success check
            component_success = [
                settings_report.get("settings_valid", False),
                cli_success,
                doc_valid,
                test_valid
            ]
            
            validation_results["overall_success"] = all(component_success)
            validation_results["success_rate"] = sum(component_success) / len(component_success)
            
            self.setup_state["validation_completed"] = True
            
        except Exception as e:
            print(f"Error during validation: {e}")
            validation_results["error"] = str(e)
        
        return validation_results
    
    def generate_summary_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report.
        
        Args:
            validation_results: Results from validation step
            
        Returns:
            Summary report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_info": {
                "root": str(self.project_root),
                "cursor_directory": str(self.cursor_dir),
                "rules_directory": str(self.rules_dir),
                "tests_directory": str(self.tests_dir)
            },
            "setup_state": self.setup_state,
            "validation_results": validation_results,
            "components_status": {
                "documentation": "✅" if self.setup_state["documentation_integrated"] else "❌",
                "settings": "✅" if self.setup_state["settings_configured"] else "❌", 
                "cli": "✅" if self.setup_state["cli_installed"] else "❌",
                "tests": "✅" if self.setup_state["tests_created"] else "❌",
                "validation": "✅" if self.setup_state["validation_completed"] else "❌"
            },
            "next_steps": self._generate_next_steps(),
            "usage_examples": self._generate_usage_examples(),
            "troubleshooting": self._generate_troubleshooting_guide()
        }
        
        return report
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on setup state."""
        steps = []
        
        if not self.setup_state["cli_installed"]:
            steps.append("Complete Cursor CLI installation")
        else:
            steps.append("Restart terminal or run: source ~/.zshrc")
            steps.append("Test CLI with: cursor-agent --version")
        
        if self.setup_state["settings_configured"]:
            steps.append("Open Cursor and verify settings loaded correctly")
            steps.append("Test rules system with current project")
        
        if self.setup_state["tests_created"]:
            steps.append("Run test suite: python tests/cursor_integration_tests.py")
        
        steps.extend([
            "Explore Cursor documentation: https://docs.cursor.com",
            "Configure additional AI models if needed",
            "Set up team collaboration features",
            "Integrate with GitHub/Git workflows"
        ])
        
        return steps
    
    def _generate_usage_examples(self) -> Dict[str, str]:
        """Generate usage examples."""
        return {
            "open_project": "cursor .",
            "open_file": "cursor path/to/file.py",
            "agent_mode": "cursor-agent",
            "agent_prompt": "cursor-agent -p 'Refactor this code for better performance'",
            "settings_update": "python .cursor/rules/cursor_settings_manager.py set-model claude-3.5-sonnet",
            "run_tests": "python tests/cursor_integration_tests.py",
            "validate_setup": "python .cursor/rules/cursor_settings_manager.py validate"
        }
    
    def _generate_troubleshooting_guide(self) -> Dict[str, str]:
        """Generate troubleshooting guide."""
        return {
            "cursor_not_found": "Ensure ~/.local/bin is in your PATH. Run: echo $PATH",
            "settings_not_loading": "Check .cursor/settings.json exists and is valid JSON",
            "cli_permission_error": "Run: chmod +x ~/.local/bin/cursor-agent",
            "rules_not_working": "Verify cursor.rules.enabled is true in settings",
            "path_issues": "Add to shell profile: export PATH=\"$HOME/.local/bin:$PATH\"",
            "import_errors": "Ensure Python modules are in correct directory structure"
        }
    
    def run_complete_setup(self) -> bool:
        """Run the complete Cursor integration setup.
        
        Returns:
            True if setup successful, False otherwise
        """
        print("🚀 Starting Complete Cursor Integration Setup")
        print("=" * 50)
        
        # Setup steps
        steps = [
            ("Documentation Integration", self.integrate_documentation),
            ("Settings Configuration", self.configure_settings),
            ("CLI Installation", self.install_cli),
            ("Test Suite Validation", self.create_test_suite)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if step_func():
                success_count += 1
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
        
        # Run validation
        validation_results = self.run_validation()
        
        # Generate and save report
        report = self.generate_summary_report(validation_results)
        report_file = self.cursor_dir / "integration_setup_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Setup report saved: {report_file}")
        
        # Display summary
        print("\n" + "=" * 50)
        print("CURSOR INTEGRATION SETUP SUMMARY")
        print("=" * 50)
        
        for component, status in report["components_status"].items():
            print(f"{component.title()}: {status}")
        
        overall_success = validation_results.get("overall_success", False)
        success_rate = validation_results.get("success_rate", 0)
        
        print(f"\nSuccess Rate: {success_rate:.1%}")
        print(f"Overall Status: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
        
        if overall_success:
            print("\n🎉 Cursor integration setup completed successfully!")
            print("\nNext Steps:")
            for i, step in enumerate(report["next_steps"][:5], 1):
                print(f"{i}. {step}")
        else:
            print("\n⚠️ Setup completed with issues. Please check the report for details.")
        
        return overall_success


def main():
    """Main function for script execution."""
    print("Cursor Integration Setup")
    print("Integrating Cursor documentation, CLI, and settings management")
    print()
    
    try:
        orchestrator = CursorIntegrationOrchestrator()
        success = orchestrator.run_complete_setup()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
