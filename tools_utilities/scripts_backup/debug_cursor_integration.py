#!/usr/bin/env python3
"""
Cursor Integration Debug Script
==============================

Purpose: Comprehensive debugging and troubleshooting for Cursor integration
Inputs: System state, configuration files, installation status
Outputs: Debug reports, diagnostic information, fix suggestions
Dependencies: subprocess, json, pathlib, sys

This script provides:
- System diagnostics
- Configuration validation
- Installation verification
- Error analysis
- Automated fixes
"""

import sys
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import tempfile


class CursorIntegrationDebugger:
    """Debug and troubleshoot Cursor integration issues."""
    
    def __init__(self):
        """Initialize the debugger."""
        self.project_root = Path.cwd()
        while not (self.project_root / ".cursor").exists() and self.project_root != self.project_root.parent:
            self.project_root = self.project_root.parent
        
        self.cursor_dir = self.project_root / ".cursor"
        self.rules_dir = self.cursor_dir / "rules"
        self.home_dir = Path.home()
        
        # Debug results storage
        self.debug_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "file_checks": {},
            "command_checks": {},
            "configuration_checks": {},
            "issues_found": [],
            "fixes_applied": [],
            "recommendations": []
        }
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information.
        
        Returns:
            System information dictionary
        """
        print("🔍 Collecting system information...")
        
        system_info = {}
        
        try:
            # Python information
            system_info["python"] = {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path[:3]  # First 3 entries
            }
            
            # Shell information
            shell_result = subprocess.run(['echo', '$SHELL'], capture_output=True, text=True)
            system_info["shell"] = {
                "current": os.environ.get('SHELL', 'unknown'),
                "user": os.environ.get('USER', 'unknown')
            }
            
            # PATH information
            system_info["path"] = {
                "PATH": os.environ.get('PATH', '').split(':'),
                "local_bin_in_path": str(self.home_dir / ".local" / "bin") in os.environ.get('PATH', '')
            }
            
            # macOS version
            try:
                sw_vers = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True)
                system_info["macos_version"] = sw_vers.stdout.strip() if sw_vers.returncode == 0 else "unknown"
            except:
                system_info["macos_version"] = "unknown"
            
            # Available disk space
            try:
                df_result = subprocess.run(['df', '-h', str(self.project_root)], capture_output=True, text=True)
                lines = df_result.stdout.split('\n')
                if len(lines) > 1:
                    fields = lines[1].split()
                    system_info["disk_space"] = {
                        "available": fields[3] if len(fields) > 3 else "unknown",
                        "used": fields[2] if len(fields) > 2 else "unknown"
                    }
            except:
                system_info["disk_space"] = {"available": "unknown", "used": "unknown"}
            
        except Exception as e:
            system_info["error"] = str(e)
        
        self.debug_results["system_info"] = system_info
        return system_info
    
    def check_file_structure(self) -> Dict[str, Any]:
        """Check file structure and permissions.
        
        Returns:
            File check results
        """
        print("📁 Checking file structure...")
        
        file_checks = {}
        
        # Required files and directories
        required_paths = {
            "project_root": self.project_root,
            "cursor_directory": self.cursor_dir,
            "rules_directory": self.rules_dir,
            "settings_file": self.cursor_dir / "settings.json",
            "keybindings_file": self.cursor_dir / "keybindings.json",
            "documentation_file": self.rules_dir / "cursor_documentation_integration.md",
            "settings_manager": self.rules_dir / "cursor_settings_manager.py",
            "cli_setup": self.rules_dir / "cursor_cli_setup.py",
            "integration_script": self.rules_dir / "setup_cursor_integration.py",
            "test_file": self.project_root / "tests" / "cursor_integration_tests.py",
            "local_bin": self.home_dir / ".local" / "bin",
            "cursor_agent": self.home_dir / ".local" / "bin" / "cursor-agent",
            "cursor_command": self.home_dir / ".local" / "bin" / "cursor"
        }
        
        for name, path in required_paths.items():
            try:
                file_checks[name] = {
                    "path": str(path),
                    "exists": path.exists(),
                    "is_file": path.is_file() if path.exists() else False,
                    "is_dir": path.is_dir() if path.exists() else False,
                    "readable": os.access(path, os.R_OK) if path.exists() else False,
                    "writable": os.access(path, os.W_OK) if path.exists() else False,
                    "executable": os.access(path, os.X_OK) if path.exists() else False,
                    "size": path.stat().st_size if path.exists() else 0
                }
            except Exception as e:
                file_checks[name] = {
                    "path": str(path),
                    "error": str(e)
                }
        
        # Check shell profile files
        shell_profiles = [
            self.home_dir / ".bashrc",
            self.home_dir / ".zshrc",
            self.home_dir / ".bash_profile",
            self.home_dir / ".profile"
        ]
        
        file_checks["shell_profiles"] = {}
        for profile in shell_profiles:
            file_checks["shell_profiles"][profile.name] = {
                "exists": profile.exists(),
                "size": profile.stat().st_size if profile.exists() else 0,
                "contains_local_bin": False
            }
            
            if profile.exists():
                try:
                    content = profile.read_text()
                    file_checks["shell_profiles"][profile.name]["contains_local_bin"] = ".local/bin" in content
                except:
                    pass
        
        self.debug_results["file_checks"] = file_checks
        return file_checks
    
    def check_commands(self) -> Dict[str, Any]:
        """Check command availability and functionality.
        
        Returns:
            Command check results
        """
        print("⚡ Checking commands...")
        
        command_checks = {}
        
        # Commands to test
        commands_to_test = [
            ("cursor-agent", ["cursor-agent", "--version"]),
            ("cursor", ["cursor", "--help"]),
            ("code", ["code", "--help"]),
            ("which_cursor", ["which", "cursor"]),
            ("which_cursor_agent", ["which", "cursor-agent"]),
            ("python", ["python", "--version"]),
            ("python3", ["python3", "--version"]),
            ("curl", ["curl", "--version"]),
            ("git", ["git", "--version"])
        ]
        
        for name, command in commands_to_test:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                command_checks[name] = {
                    "available": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip()[:200],  # First 200 chars
                    "stderr": result.stderr.strip()[:200] if result.stderr else "",
                    "command": " ".join(command)
                }
            except subprocess.TimeoutExpired:
                command_checks[name] = {
                    "available": False,
                    "error": "timeout",
                    "command": " ".join(command)
                }
            except FileNotFoundError:
                command_checks[name] = {
                    "available": False,
                    "error": "not_found",
                    "command": " ".join(command)
                }
            except Exception as e:
                command_checks[name] = {
                    "available": False,
                    "error": str(e),
                    "command": " ".join(command)
                }
        
        self.debug_results["command_checks"] = command_checks
        return command_checks
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files and settings.
        
        Returns:
            Configuration check results
        """
        print("⚙️ Checking configuration...")
        
        config_checks = {}
        
        # Check settings.json
        settings_file = self.cursor_dir / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                config_checks["settings"] = {
                    "valid_json": True,
                    "ai_model": settings.get("cursor.aiModel", "not_set"),
                    "rules_enabled": settings.get("cursor.rules", {}).get("enabled", False),
                    "settings_count": len(settings),
                    "has_project_config": "cursor.project" in settings
                }
            except json.JSONDecodeError as e:
                config_checks["settings"] = {
                    "valid_json": False,
                    "error": str(e)
                }
        else:
            config_checks["settings"] = {
                "exists": False
            }
        
        # Check keybindings.json
        keybindings_file = self.cursor_dir / "keybindings.json"
        if keybindings_file.exists():
            try:
                with open(keybindings_file, 'r') as f:
                    keybindings = json.load(f)
                config_checks["keybindings"] = {
                    "valid_json": True,
                    "count": len(keybindings) if isinstance(keybindings, list) else 0,
                    "has_ask_ai": any(kb.get("command") == "cursor.askAI" for kb in keybindings) if isinstance(keybindings, list) else False
                }
            except json.JSONDecodeError as e:
                config_checks["keybindings"] = {
                    "valid_json": False,
                    "error": str(e)
                }
        else:
            config_checks["keybindings"] = {
                "exists": False
            }
        
        # Check PATH configuration
        path_env = os.environ.get('PATH', '')
        local_bin = str(self.home_dir / ".local" / "bin")
        config_checks["path"] = {
            "local_bin_in_path": local_bin in path_env,
            "path_entries": len(path_env.split(':')),
            "local_bin_path": local_bin
        }
        
        self.debug_results["configuration_checks"] = config_checks
        return config_checks
    
    def analyze_issues(self) -> List[Dict[str, Any]]:
        """Analyze collected data to identify issues.
        
        Returns:
            List of identified issues
        """
        print("🔍 Analyzing issues...")
        
        issues = []
        
        # Check file structure issues
        file_checks = self.debug_results.get("file_checks", {})
        
        for name, check in file_checks.items():
            if isinstance(check, dict) and "exists" in check:
                if not check["exists"] and name in ["cursor_directory", "rules_directory"]:
                    issues.append({
                        "type": "missing_directory",
                        "severity": "high",
                        "description": f"Required directory missing: {name}",
                        "file": check.get("path", "unknown"),
                        "fix": f"Create directory: mkdir -p {check.get('path', '')}"
                    })
                elif not check["exists"] and name in ["settings_manager", "cli_setup", "integration_script"]:
                    issues.append({
                        "type": "missing_file",
                        "severity": "high", 
                        "description": f"Required file missing: {name}",
                        "file": check.get("path", "unknown"),
                        "fix": "Re-run the integration setup script"
                    })
                elif check["exists"] and not check.get("executable", True) and name in ["cursor_agent", "cursor_command"]:
                    issues.append({
                        "type": "permission_issue",
                        "severity": "medium",
                        "description": f"File not executable: {name}",
                        "file": check.get("path", "unknown"),
                        "fix": f"chmod +x {check.get('path', '')}"
                    })
        
        # Check command issues
        command_checks = self.debug_results.get("command_checks", {})
        
        if not command_checks.get("cursor-agent", {}).get("available", False):
            issues.append({
                "type": "missing_command",
                "severity": "high",
                "description": "cursor-agent command not available",
                "fix": "Install Cursor CLI or check PATH configuration"
            })
        
        if not command_checks.get("cursor", {}).get("available", False):
            issues.append({
                "type": "missing_command", 
                "severity": "medium",
                "description": "cursor command not available",
                "fix": "Run CLI setup script to create cursor command"
            })
        
        # Check configuration issues
        config_checks = self.debug_results.get("configuration_checks", {})
        
        if not config_checks.get("path", {}).get("local_bin_in_path", False):
            issues.append({
                "type": "path_issue",
                "severity": "high",
                "description": "~/.local/bin not in PATH",
                "fix": "Add to shell profile: export PATH=\"$HOME/.local/bin:$PATH\""
            })
        
        if not config_checks.get("settings", {}).get("valid_json", True):
            issues.append({
                "type": "config_issue",
                "severity": "medium",
                "description": "Invalid settings.json file",
                "fix": "Re-initialize settings with cursor_settings_manager.py"
            })
        
        self.debug_results["issues_found"] = issues
        return issues
    
    def suggest_fixes(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Suggest fixes for identified issues.
        
        Args:
            issues: List of identified issues
            
        Returns:
            List of fix suggestions
        """
        print("🔧 Generating fix suggestions...")
        
        fixes = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # Generate fixes by priority
        if "missing_directory" in issue_types:
            fixes.append("Create missing directories:")
            for issue in issue_types["missing_directory"]:
                fixes.append(f"  mkdir -p {issue.get('file', '')}")
        
        if "path_issue" in issue_types:
            shell = os.environ.get('SHELL', '/bin/zsh').split('/')[-1]
            profile_file = f"~/.{shell}rc" if shell in ['bash', 'zsh'] else "~/.profile"
            fixes.extend([
                f"Fix PATH configuration:",
                f"  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> {profile_file}",
                f"  source {profile_file}"
            ])
        
        if "missing_command" in issue_types:
            fixes.extend([
                "Install missing commands:",
                "  python .cursor/rules/cursor_cli_setup.py"
            ])
        
        if "permission_issue" in issue_types:
            fixes.append("Fix permissions:")
            for issue in issue_types["permission_issue"]:
                fixes.append(f"  chmod +x {issue.get('file', '')}")
        
        if "config_issue" in issue_types:
            fixes.extend([
                "Fix configuration issues:",
                "  python .cursor/rules/cursor_settings_manager.py init"
            ])
        
        # General fixes
        fixes.extend([
            "",
            "General troubleshooting:",
            "  1. Restart terminal after PATH changes",
            "  2. Re-run complete setup: python .cursor/rules/setup_cursor_integration.py",
            "  3. Check Cursor app is installed: open -a Cursor",
            "  4. Verify system requirements: macOS 10.15+, Python 3.7+"
        ])
        
        self.debug_results["recommendations"] = fixes
        return fixes
    
    def run_automated_fixes(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Run automated fixes for issues that can be safely fixed.
        
        Args:
            issues: List of issues to fix
            
        Returns:
            List of fixes applied
        """
        print("🤖 Running automated fixes...")
        
        fixes_applied = []
        
        for issue in issues:
            issue_type = issue.get("type")
            severity = issue.get("severity")
            
            # Only apply low-risk automated fixes
            if issue_type == "missing_directory" and severity != "critical":
                try:
                    path = Path(issue.get("file", ""))
                    path.mkdir(parents=True, exist_ok=True)
                    fixes_applied.append(f"Created directory: {path}")
                except Exception as e:
                    fixes_applied.append(f"Failed to create directory {path}: {e}")
            
            elif issue_type == "permission_issue":
                try:
                    path = Path(issue.get("file", ""))
                    if path.exists():
                        path.chmod(0o755)
                        fixes_applied.append(f"Fixed permissions: {path}")
                except Exception as e:
                    fixes_applied.append(f"Failed to fix permissions {path}: {e}")
        
        self.debug_results["fixes_applied"] = fixes_applied
        return fixes_applied
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report.
        
        Returns:
            Complete debug report
        """
        print("📊 Generating debug report...")
        
        # Add summary information
        issues = self.debug_results.get("issues_found", [])
        self.debug_results["summary"] = {
            "total_issues": len(issues),
            "high_severity": len([i for i in issues if i.get("severity") == "high"]),
            "medium_severity": len([i for i in issues if i.get("severity") == "medium"]),
            "low_severity": len([i for i in issues if i.get("severity") == "low"]),
            "fixes_available": len([i for i in issues if "fix" in i])
        }
        
        return self.debug_results
    
    def save_debug_report(self, report: Dict[str, Any]) -> Path:
        """Save debug report to file.
        
        Args:
            report: Debug report to save
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.cursor_dir / f"debug_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Debug report saved: {report_file}")
        return report_file
    
    def run_complete_debug(self, apply_fixes: bool = False) -> Dict[str, Any]:
        """Run complete debug analysis.
        
        Args:
            apply_fixes: Whether to apply automated fixes
            
        Returns:
            Complete debug report
        """
        print("🚀 Starting Cursor Integration Debug Analysis")
        print("=" * 50)
        
        # Collect system information
        self.collect_system_info()
        
        # Check file structure
        self.check_file_structure()
        
        # Check commands
        self.check_commands()
        
        # Check configuration
        self.check_configuration()
        
        # Analyze issues
        issues = self.analyze_issues()
        
        # Generate fix suggestions
        self.suggest_fixes(issues)
        
        # Apply fixes if requested
        if apply_fixes:
            self.run_automated_fixes(issues)
        
        # Generate final report
        report = self.generate_debug_report()
        
        # Save report
        report_file = self.save_debug_report(report)
        
        # Display summary
        self.display_summary(report)
        
        return report
    
    def display_summary(self, report: Dict[str, Any]):
        """Display debug summary.
        
        Args:
            report: Debug report to summarize
        """
        print("\n" + "=" * 50)
        print("DEBUG ANALYSIS SUMMARY")
        print("=" * 50)
        
        summary = report.get("summary", {})
        issues = report.get("issues_found", [])
        
        print(f"Total Issues Found: {summary.get('total_issues', 0)}")
        print(f"  High Severity: {summary.get('high_severity', 0)}")
        print(f"  Medium Severity: {summary.get('medium_severity', 0)}")
        print(f"  Low Severity: {summary.get('low_severity', 0)}")
        
        if issues:
            print("\nTop Issues:")
            for i, issue in enumerate(issues[:5], 1):
                print(f"{i}. {issue.get('description', 'Unknown issue')} ({issue.get('severity', 'unknown')})")
        
        fixes = report.get("recommendations", [])
        if fixes:
            print("\nRecommended Fixes:")
            for fix in fixes[:10]:  # Show first 10 fixes
                if fix.strip():
                    print(f"  {fix}")
        
        print(f"\nDetailed report saved to debug report file")
        
        # Overall status
        high_issues = summary.get('high_severity', 0)
        if high_issues == 0:
            print("\n✅ No critical issues found - Cursor integration should work properly")
        else:
            print(f"\n⚠️ {high_issues} critical issues found - Please address these for proper functionality")


def main():
    """Main function for debug script."""
    print("Cursor Integration Debugger")
    print("Comprehensive analysis and troubleshooting")
    print()
    
    apply_fixes = "--fix" in sys.argv or "-f" in sys.argv
    
    try:
        debugger = CursorIntegrationDebugger()
        report = debugger.run_complete_debug(apply_fixes=apply_fixes)
        
        # Exit with appropriate code
        high_issues = report.get("summary", {}).get("high_severity", 0)
        sys.exit(0 if high_issues == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nDebug analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nDebug analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
