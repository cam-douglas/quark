#!/usr/bin/env python3
"""
Cursor CLI Setup for Mac Terminal
=================================

Purpose: Automated setup of Cursor CLI for Mac terminal integration
Inputs: System configuration, shell preferences, installation options
Outputs: Configured Cursor CLI, updated PATH, shell integration
Dependencies: subprocess, os, pathlib, shutil

This module provides automated setup of Cursor CLI including:
- CLI installation and verification
- PATH configuration for bash/zsh
- Shell command integration
- Verification and testing
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import tempfile


class CursorCLISetup:
    """Manages Cursor CLI installation and setup for Mac terminal."""
    
    def __init__(self):
        """Initialize the CLI setup manager."""
        self.home_dir = Path.home()
        self.local_bin = self.home_dir / ".local" / "bin"
        self.bash_profile = self.home_dir / ".bashrc"
        self.zsh_profile = self.home_dir / ".zshrc"
        self.current_shell = os.environ.get('SHELL', '/bin/zsh')
        
        # Ensure local bin directory exists
        self.local_bin.mkdir(parents=True, exist_ok=True)
        
        # Installation URLs and paths
        self.install_url = "https://cursor.com/install"
        self.cursor_agent_path = self.local_bin / "cursor-agent"
        
    def detect_shell(self) -> str:
        """Detect the current shell.
        
        Returns:
            Shell name (bash, zsh, etc.)
        """
        shell = self.current_shell.split('/')[-1]
        print(f"Detected shell: {shell}")
        return shell
    
    def check_cursor_cli_installed(self) -> bool:
        """Check if Cursor CLI is already installed.
        
        Returns:
            True if installed, False otherwise
        """
        try:
            result = subprocess.run(['cursor-agent', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"Cursor CLI already installed: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print("Cursor CLI not found")
        return False
    
    def install_cursor_cli(self) -> bool:
        """Install Cursor CLI using the official installer.
        
        Returns:
            True if installation successful, False otherwise
        """
        print("Installing Cursor CLI...")
        
        try:
            # Download and run the installer
            cmd = f"curl -fsS {self.install_url} | bash"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("Cursor CLI installation completed")
                print(f"Output: {result.stdout}")
                return True
            else:
                print(f"Installation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Installation timed out")
            return False
        except Exception as e:
            print(f"Installation error: {e}")
            return False
    
    def update_path_configuration(self) -> bool:
        """Update PATH configuration in shell profile.
        
        Returns:
            True if successful, False otherwise
        """
        shell = self.detect_shell()
        
        if shell == 'zsh':
            profile_file = self.zsh_profile
        elif shell == 'bash':
            profile_file = self.bash_profile
        else:
            print(f"Unsupported shell: {shell}")
            return False
        
        path_export = f'export PATH="$HOME/.local/bin:$PATH"'
        
        # Check if PATH is already configured
        if profile_file.exists():
            content = profile_file.read_text()
            if path_export in content or "$HOME/.local/bin" in content:
                print(f"PATH already configured in {profile_file}")
                return True
        
        # Add PATH export to profile
        try:
            with open(profile_file, 'a') as f:
                f.write(f"\n# Cursor CLI PATH configuration\n")
                f.write(f"{path_export}\n")
            
            print(f"PATH configuration added to {profile_file}")
            return True
            
        except Exception as e:
            print(f"Error updating PATH configuration: {e}")
            return False
    
    def install_shell_commands(self) -> bool:
        """Install shell commands for Cursor integration.
        
        Returns:
            True if successful, False otherwise
        """
        print("Installing shell commands...")
        
        # Create cursor command wrapper
        cursor_command = self.local_bin / "cursor"
        
        cursor_script = """#!/bin/bash
# Cursor command wrapper
if [ $# -eq 0 ]; then
    # No arguments - open Cursor
    open -a Cursor
else
    # Arguments provided - open file/directory in Cursor
    open -a Cursor "$@"
fi
"""
        
        try:
            with open(cursor_command, 'w') as f:
                f.write(cursor_script)
            
            # Make executable
            cursor_command.chmod(0o755)
            print(f"Cursor command installed: {cursor_command}")
            
            # Create code command as alias to cursor
            code_command = self.local_bin / "code"
            if not code_command.exists():
                code_command.symlink_to(cursor_command)
                print(f"Code command linked: {code_command}")
            
            return True
            
        except Exception as e:
            print(f"Error installing shell commands: {e}")
            return False
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify the Cursor CLI installation.
        
        Returns:
            Verification results dictionary
        """
        print("Verifying Cursor CLI installation...")
        
        results = {
            "cursor_agent_available": False,
            "cursor_command_available": False,
            "code_command_available": False,
            "path_configured": False,
            "shell_integration": False
        }
        
        # Check cursor-agent
        try:
            result = subprocess.run(['cursor-agent', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            results["cursor_agent_available"] = result.returncode == 0
            if results["cursor_agent_available"]:
                print(f"✓ cursor-agent: {result.stdout.strip()}")
        except:
            print("✗ cursor-agent not available")
        
        # Check cursor command
        try:
            result = subprocess.run(['which', 'cursor'], 
                                  capture_output=True, text=True, timeout=5)
            results["cursor_command_available"] = result.returncode == 0
            if results["cursor_command_available"]:
                print(f"✓ cursor command: {result.stdout.strip()}")
        except:
            print("✗ cursor command not available")
        
        # Check code command
        try:
            result = subprocess.run(['which', 'code'], 
                                  capture_output=True, text=True, timeout=5)
            results["code_command_available"] = result.returncode == 0
            if results["code_command_available"]:
                print(f"✓ code command: {result.stdout.strip()}")
        except:
            print("✗ code command not available")
        
        # Check PATH configuration
        shell = self.detect_shell()
        if shell == 'zsh':
            profile_file = self.zsh_profile
        else:
            profile_file = self.bash_profile
        
        if profile_file.exists():
            content = profile_file.read_text()
            results["path_configured"] = "$HOME/.local/bin" in content
            if results["path_configured"]:
                print(f"✓ PATH configured in {profile_file}")
            else:
                print(f"✗ PATH not configured in {profile_file}")
        
        # Overall shell integration
        results["shell_integration"] = (
            results["cursor_agent_available"] and 
            results["cursor_command_available"] and 
            results["path_configured"]
        )
        
        return results
    
    def create_test_script(self) -> Path:
        """Create a test script for Cursor CLI functionality.
        
        Returns:
            Path to test script
        """
        test_script = Path.cwd() / "test_cursor_cli.py"
        
        test_content = """#!/usr/bin/env python3
\"\"\"
Test script for Cursor CLI functionality.
\"\"\"

import subprocess
import tempfile
import os
from pathlib import Path

def test_cursor_agent():
    \"\"\"Test cursor-agent command.\"\"\"
    try:
        result = subprocess.run(['cursor-agent', '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"cursor-agent version: {result.stdout.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"cursor-agent test failed: {e}")
        return False

def test_cursor_command():
    \"\"\"Test cursor command.\"\"\"
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test file for Cursor CLI\\nprint('Hello, Cursor!')")
            temp_file = f.name
        
        # Test opening file with cursor command
        result = subprocess.run(['cursor', temp_file], 
                              capture_output=True, text=True, timeout=5)
        
        # Clean up
        os.unlink(temp_file)
        
        print("cursor command test completed")
        return True
        
    except Exception as e:
        print(f"cursor command test failed: {e}")
        return False

def main():
    \"\"\"Run all tests.\"\"\"
    print("Testing Cursor CLI functionality...")
    
    tests = [
        ("cursor-agent", test_cursor_agent),
        ("cursor command", test_cursor_command)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\\nTesting {name}...")
        results[name] = test_func()
        status = "PASS" if results[name] else "FAIL"
        print(f"{name}: {status}")
    
    print("\\nTest Summary:")
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    print(f"\\nOverall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

if __name__ == "__main__":
    main()
"""
        
        with open(test_script, 'w') as f:
            f.write(test_content)
        
        test_script.chmod(0o755)
        print(f"Test script created: {test_script}")
        return test_script
    
    def setup_complete_cli(self) -> bool:
        """Perform complete Cursor CLI setup.
        
        Returns:
            True if setup successful, False otherwise
        """
        print("Starting Cursor CLI setup for Mac terminal...")
        
        steps = [
            ("Checking existing installation", self.check_cursor_cli_installed),
            ("Installing Cursor CLI", lambda: self.install_cursor_cli() if not self.check_cursor_cli_installed() else True),
            ("Updating PATH configuration", self.update_path_configuration),
            ("Installing shell commands", self.install_shell_commands)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"Failed: {step_name}")
                return False
            print(f"Completed: {step_name}")
        
        print("\nVerifying installation...")
        verification = self.verify_installation()
        
        if verification["shell_integration"]:
            print("\n✅ Cursor CLI setup completed successfully!")
            print("\nNext steps:")
            print("1. Restart your terminal or run: source ~/.zshrc (or ~/.bashrc)")
            print("2. Test with: cursor-agent --version")
            print("3. Test with: cursor . (to open current directory)")
            return True
        else:
            print("\n❌ Setup completed with issues. Please check the verification results.")
            return False
    
    def generate_setup_report(self) -> Dict:
        """Generate a setup report.
        
        Returns:
            Setup report dictionary
        """
        verification = self.verify_installation()
        
        report = {
            "timestamp": subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            "system_info": {
                "shell": self.detect_shell(),
                "home_directory": str(self.home_dir),
                "local_bin": str(self.local_bin)
            },
            "installation_status": verification,
            "next_steps": [
                "Restart terminal or source shell profile",
                "Test cursor-agent --version",
                "Test cursor . to open current directory",
                "Run test script to verify functionality"
            ],
            "troubleshooting": {
                "path_issues": "Ensure ~/.local/bin is in your PATH",
                "permission_issues": "Check file permissions on cursor commands",
                "installation_issues": "Try manual installation from cursor.com"
            }
        }
        
        return report


def main():
    """Main function for CLI usage."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run test mode
        setup = CursorCLISetup()
        test_script = setup.create_test_script()
        print(f"Test script created: {test_script}")
        print("Run with: python test_cursor_cli.py")
        return
    
    # Run full setup
    setup = CursorCLISetup()
    success = setup.setup_complete_cli()
    
    # Generate report
    report = setup.generate_setup_report()
    report_file = Path.cwd() / ".cursor" / "cli_setup_report.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSetup report saved: {report_file}")
    
    if success:
        # Create test script
        test_script = setup.create_test_script()
        print(f"Test script available: {test_script}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
