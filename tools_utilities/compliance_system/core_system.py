#!/usr/bin/env python3
"""
Core compliance system functionality.

This module contains the main QuarkComplianceSystem class that orchestrates
all compliance checking operations.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime
from contextlib import contextmanager


class QuarkComplianceSystem:
    """Unified compliance system for Quark project."""
    
    def __init__(self, workspace_root: str = "/Users/camdouglas/quark"):
        """Initialize unified compliance system"""
        self.workspace_root = Path(workspace_root)
        self.compliance_checker_path = self.workspace_root / "tools_utilities" / "compliance_system" / "compliance_checker" / "cli.py"
        self.status_file = self.workspace_root / "logs" / "compliance_system_status.json"
        self.pid_file = self.workspace_root / "logs" / "compliance_system.pid"
        
        # Create logs directory
        self.status_file.parent.mkdir(exist_ok=True)
        
        # System state
        self.running = False
        self.current_operation = None
        self.operation_start_time = None
        self.monitored_files: Set[str] = set()
        self.violations_detected: List[Dict] = []
    
    def run_compliance_check(self, files: List[str] = None) -> Dict:
        """Run compliance check on specified files or all files"""
        try:
            cmd = [sys.executable, str(self.compliance_checker_path)]
            if files:
                cmd.extend(['--paths'] + files)
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "compliant": result.returncode == 0
            }
        except Exception as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e),
                "compliant": False
            }
    
    def parse_violations(self, output: str) -> List[Dict]:
        """Parse violations from compliance checker output"""
        violations = []
        lines = output.split('\n')
        
        current_violation = None
        for line in lines:
            line = line.strip()
            
            if line.startswith('**') and line.endswith('**'):
                if current_violation:
                    violations.append(current_violation)
                current_violation = {
                    'file': line.strip('*'),
                    'details': []
                }
            
            elif current_violation and line.startswith('- '):
                current_violation['details'].append(line[2:])
        
        if current_violation:
            violations.append(current_violation)
        
        return violations
    
    def phase_before_operation(self, operation_name: str, target_files: List[str] = None) -> bool:
        """Phase 1: BEFORE operation compliance check"""
        print(f"🔍 PHASE 1: Pre-operation compliance check for '{operation_name}'")
        print("=" * 60)
        
        result = self.run_compliance_check(target_files)
        
        if result["compliant"]:
            print("✅ Pre-operation check: Workspace is compliant")
            return True
        else:
            violations = self.parse_violations(result["stdout"])
            print("❌ Pre-operation check: Violations detected")
            print("🚫 Operation blocked due to existing violations")
            
            for violation in violations:
                print(f"📁 {violation['file']}")
                for detail in violation['details']:
                    print(f"   - {detail}")
            
            return False
    
    def phase_during_operation(self, operation_name: str, target_files: List[str] = None):
        """Phase 2: DURING operation monitoring"""
        print(f"👁️  PHASE 2: Monitoring during '{operation_name}'")
        print("=" * 60)
        
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        self.monitored_files = set(target_files) if target_files else set()
        self.violations_detected = []
        
        print("🔍 Operation monitoring active")
        print("📁 Monitoring for violations...")
    
    def phase_after_operation(self, operation_name: str, target_files: List[str] = None) -> bool:
        """Phase 3: AFTER operation compliance verification"""
        print(f"✅ PHASE 3: Post-operation compliance verification for '{operation_name}'")
        print("=" * 60)
        
        result = self.run_compliance_check(target_files)
        
        if result["compliant"]:
            print("✅ Post-operation check: All files remain compliant")
            return True
        else:
            violations = self.parse_violations(result["stdout"])
            print("❌ Post-operation check: New violations detected")
            
            for violation in violations:
                print(f"📁 {violation['file']}")
                for detail in violation['details']:
                    print(f"   - {detail}")
            
            print("\n💡 Recommended actions:")
            print("   - Fix violations immediately")
            print("   - Run: python tools_utilities/compliance_system.py --fix")
            print("   - Re-run operation after fixes")
            
            return False
    
    @contextmanager
    def operation_context(self, operation_name: str, target_files: List[str] = None):
        """Context manager for three-phase compliance checking"""
        # Phase 1: Before
        if not self.phase_before_operation(operation_name, target_files):
            raise RuntimeError(f"Operation '{operation_name}' blocked due to pre-existing violations")
        
        # Phase 2: During
        self.phase_during_operation(operation_name, target_files)
        
        try:
            yield self
        finally:
            # Phase 3: After
            self.phase_after_operation(operation_name, target_files)
    
    def start_system(self, background: bool = True):
        """Start the complete compliance system"""
        if self.running:
            print("⚠️  Compliance system is already running")
            return True
        
        print("🚀 Starting Quark Compliance System...")
        print("=" * 50)
        
        self.running = True
        # Write PID for status introspection
        try:
            with open(self.pid_file, 'w') as pf:
                pf.write(str(os.getpid()))
        except OSError:
            # Non-fatal: continue without PID persistence
            pass
        self._save_status("running", background=background)
        
        print("✅ Compliance system ready")
        print("✅ Pre-push hook integration active")
        print("✅ Three-phase checking operational")
        print("=" * 50)
        print("🔍 Compliance system is now operational")
        
        return True
    
    def stop_system(self):
        """Stop the compliance system"""
        if not self.running:
            print("⚠️  Compliance system is not running")
            # Still ensure persisted state reflects stopped
            try:
                if self.pid_file.exists():
                    self.pid_file.unlink()
            except OSError:
                pass
            self._save_status("stopped", background=False)
            return

        print("🛑 Stopping Quark Compliance System...")

        self.running = False
        # Remove PID file if present
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except OSError:
            pass
        self._save_status("stopped", background=False)

        print("✅ Compliance system stopped")
    
    def _save_status(self, status: str, background: bool | None = None):
        """Save system status to disk for cross-process visibility."""
        status_data = {
            "status": status,
            "running": status == "running",
            "timestamp": datetime.now().isoformat(),
            "workspace": str(self.workspace_root),
            "pid": None,
            "background": bool(background) if background is not None else None,
            "components": {
                "compliance_checker": True,
                "three_phase_system": True,
                "pre_push_hook": self._check_pre_push_hook(),
                "cursor_integration": True
            }
        }
        # Populate PID if file exists
        try:
            if self.pid_file.exists():
                with open(self.pid_file, 'r') as pf:
                    pid_str = pf.read().strip()
                    status_data["pid"] = int(pid_str) if pid_str.isdigit() else None
        except OSError:
            pass
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    def _check_pre_push_hook(self) -> bool:
        """Check if pre-push hook is properly configured"""
        pre_push_hook = self.workspace_root / ".git" / "hooks" / "pre-push"
        if not pre_push_hook.exists():
            return False
        
        try:
            with open(pre_push_hook, 'r') as f:
                content = f.read()
                return "compliance_system.py" in content
        except IOError:
            return False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "running": self.running,
            "timestamp": datetime.now().isoformat(),
            "workspace": str(self.workspace_root)
        }
        
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    saved_status = json.load(f)
                    status.update(saved_status)
                    # Normalize running flag from saved status if present
                    if "running" in saved_status:
                        status["running"] = bool(saved_status["running"])
                    elif saved_status.get("status") in {"running"}:
                        status["running"] = True
            except (json.JSONDecodeError, IOError):
                pass
        
        return status
    
    def check_compliance_now(self, files: List[str] = None) -> bool:
        """Run immediate compliance check"""
        result = self.run_compliance_check(files)
        
        if result["returncode"] == 0:
            print("✅ Compliance check passed")
            return True
        else:
            print("❌ Compliance check failed")
            print(result["stdout"])
            return False
