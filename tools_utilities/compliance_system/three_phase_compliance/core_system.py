"""
Core Three-Phase Compliance System

Main system class for three-phase compliance checking.

Author: Quark AI
Date: 2025-01-27
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set, Callable, Any
from datetime import datetime
from .file_monitor import ComplianceFileMonitor


class ThreePhaseComplianceSystem:
    """
    Three-phase compliance checking system that validates
    before, monitors during, and verifies after operations.
    """
    
    def __init__(self, workspace_root: str = "/Users/camdouglas/quark"):
        """Initialize three-phase compliance system"""
        self.workspace_root = Path(workspace_root)
        self.compliance_checker_path = self.workspace_root / "tools_utilities" / "rule_compliance_checker.py"
        self.compliance_log = self.workspace_root / "logs" / "three_phase_compliance.json"
        
        # Create logs directory
        self.compliance_log.parent.mkdir(exist_ok=True)
        
        # Phase tracking
        self.current_operation = None
        self.operation_start_time = None
        self.monitored_files: Set[str] = set()
        self.violations_detected: List[Dict] = []
        
        # File monitoring
        self.file_monitor = ComplianceFileMonitor(self)
        self.monitoring_active = False
        
        # Callbacks
        self.before_callbacks: List[Callable] = []
        self.during_callbacks: List[Callable] = []
        self.after_callbacks: List[Callable] = []
    
    def log_phase(self, phase: str, operation: str, details: Dict):
        """Log compliance phase information"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "operation": operation,
            "details": details
        }
        
        # Load existing log
        if self.compliance_log.exists():
            try:
                with open(self.compliance_log, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                log_data = {"entries": []}
        else:
            log_data = {"entries": []}
        
        # Add new entry
        log_data["entries"].append(log_entry)
        
        # Keep only last 200 entries
        if len(log_data["entries"]) > 200:
            log_data["entries"] = log_data["entries"][-200:]
        
        # Save log
        with open(self.compliance_log, 'w') as f:
            json.dump(log_data, f, indent=2)
    
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
        """
        Phase 1: BEFORE operation compliance check
        
        Validates that the workspace is in a compliant state
        before starting any operation.
        """
        print(f"🔍 PHASE 1: Pre-operation compliance check for '{operation_name}'")
        print("=" * 60)
        
        # Run compliance check
        result = self.run_compliance_check(target_files)
        
        if result["compliant"]:
            print("✅ Pre-operation check: Workspace is compliant")
            self.log_phase("BEFORE", operation_name, {
                "status": "compliant",
                "files_checked": target_files or "all",
                "violations": 0
            })
            
            # Run before callbacks
            for callback in self.before_callbacks:
                try:
                    callback("compliant", result)
                except Exception as e:
                    print(f"⚠️  Before callback error: {e}")
            
            return True
        else:
            violations = self.parse_violations(result["stdout"])
            print("❌ Pre-operation check: Violations detected")
            print("🚫 Operation blocked due to existing violations")
            
            for violation in violations:
                print(f"📁 {violation['file']}")
                for detail in violation['details']:
                    print(f"   - {detail}")
            
            self.log_phase("BEFORE", operation_name, {
                "status": "blocked",
                "files_checked": target_files or "all",
                "violations": len(violations),
                "violation_details": violations
            })
            
            # Run before callbacks
            for callback in self.before_callbacks:
                try:
                    callback("blocked", result)
                except Exception as e:
                    print(f"⚠️  Before callback error: {e}")
            
            return False
    
    def phase_during_operation(self, operation_name: str, target_files: List[str] = None):
        """
        Phase 2: DURING operation real-time monitoring
        
        Monitors file changes in real-time during the operation
        and provides immediate feedback on violations.
        """
        print(f"👁️  PHASE 2: Real-time monitoring during '{operation_name}'")
        print("=" * 60)
        
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        self.monitored_files = set(target_files) if target_files else set()
        self.violations_detected = []
        
        # Start file monitoring
        self.start_file_monitoring()
        
        # Log monitoring start
        self.log_phase("DURING", operation_name, {
            "status": "monitoring_started",
            "monitored_files": list(self.monitored_files) if self.monitored_files else "all"
        })
        
        print("🔍 Real-time compliance monitoring active")
        print("📁 Monitoring file changes for violations...")
    
    def phase_after_operation(self, operation_name: str, target_files: List[str] = None) -> bool:
        """
        Phase 3: AFTER operation compliance verification
        
        Verifies that the operation completed without introducing
        new violations and provides a final compliance report.
        """
        print(f"✅ PHASE 3: Post-operation compliance verification for '{operation_name}'")
        print("=" * 60)
        
        # Stop file monitoring
        self.stop_file_monitoring()
        
        # Run final compliance check
        result = self.run_compliance_check(target_files)
        
        if result["compliant"]:
            print("✅ Post-operation check: All files remain compliant")
            self.log_phase("AFTER", operation_name, {
                "status": "compliant",
                "files_checked": target_files or "all",
                "violations": 0,
                "operation_duration": time.time() - self.operation_start_time if self.operation_start_time else 0
            })
            
            # Run after callbacks
            for callback in self.after_callbacks:
                try:
                    callback("compliant", result)
                except Exception as e:
                    print(f"⚠️  After callback error: {e}")
            
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
            print("   - Run: python tools_utilities/rule_compliance_checker.py --fix")
            print("   - Re-run operation after fixes")
            
            self.log_phase("AFTER", operation_name, {
                "status": "violations_detected",
                "files_checked": target_files or "all",
                "violations": len(violations),
                "violation_details": violations,
                "operation_duration": time.time() - self.operation_start_time if self.operation_start_time else 0
            })
            
            # Run after callbacks
            for callback in self.after_callbacks:
                try:
                    callback("violations_detected", result)
                except Exception as e:
                    print(f"⚠️  After callback error: {e}")
            
            return False
    
    def start_file_monitoring(self):
        """Start real-time file monitoring"""
        if self.monitoring_active:
            return
        
        self.file_monitor.start_monitoring()
        self.monitoring_active = True
    
    def stop_file_monitoring(self):
        """Stop real-time file monitoring"""
        if self.monitoring_active:
            self.file_monitor.stop_monitoring()
            self.monitoring_active = False
    
    def handle_file_change(self, file_path: Path):
        """Handle file change during monitoring"""
        # Check if we should monitor this file
        if (self.monitored_files and 
            str(file_path) not in self.monitored_files and
            not any(str(file_path).startswith(f) for f in self.monitored_files)):
            return
        
        # Skip excluded files
        excluded_dirs = {'__pycache__', '.git', 'node_modules', 'venv', 'env', 'logs'}
        if any(part in excluded_dirs for part in file_path.parts):
            return
        
        # Check file extension
        if file_path.suffix.lower() not in {'.py', '.md', '.yaml', '.yml', '.json'}:
            return
        
        # Run compliance check on changed file
        result = self.run_compliance_check([str(file_path)])
        
        if not result["compliant"]:
            violations = self.parse_violations(result["stdout"])
            self.violations_detected.extend(violations)
            
            print(f"\n🚨 REAL-TIME VIOLATION DETECTED")
            print(f"📁 File: {file_path}")
            print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
            
            for violation in violations:
                for detail in violation['details']:
                    if 'exceeds' in detail.lower() or 'error' in detail.lower():
                        print(f"❌ {detail}")
                    elif 'warning' in detail.lower():
                        print(f"⚠️  {detail}")
                    else:
                        print(f"ℹ️  {detail}")
            
            # Run during callbacks
            for callback in self.during_callbacks:
                try:
                    callback("violation_detected", {"file": str(file_path), "violations": violations})
                except Exception as e:
                    print(f"⚠️  During callback error: {e}")
    
    def add_callback(self, phase: str, callback: Callable):
        """Add callback for specific phase"""
        if phase == "before":
            self.before_callbacks.append(callback)
        elif phase == "during":
            self.during_callbacks.append(callback)
        elif phase == "after":
            self.after_callbacks.append(callback)
    
    def get_operation_summary(self) -> Dict:
        """Get summary of current operation"""
        return {
            "current_operation": self.current_operation,
            "operation_duration": time.time() - self.operation_start_time if self.operation_start_time else 0,
            "monitored_files": list(self.monitored_files),
            "violations_detected": len(self.violations_detected),
            "monitoring_active": self.monitoring_active
        }
