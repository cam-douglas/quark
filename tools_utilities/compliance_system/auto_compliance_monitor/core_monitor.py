"""
Core Auto Compliance Monitor

Main monitoring system for automatic compliance checking.

Author: Quark AI
Date: 2025-01-27
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class AutoComplianceMonitor:
    """
    Automatic compliance monitoring system that runs checks
    after any file operation and provides immediate feedback.
    """
    
    def __init__(self, workspace_root: str = "/Users/camdouglas/quark"):
        """Initialize auto compliance monitor"""
        self.workspace_root = Path(workspace_root)
        self.compliance_checker_path = self.workspace_root / "tools_utilities" / "rule_compliance_checker.py"
        self.violation_log = self.workspace_root / "logs" / "compliance_violations.json"
        self.last_check_times: Dict[str, float] = {}
        self.check_cooldown = 1.0  # Minimum seconds between checks for same file
        
        # Create logs directory
        self.violation_log.parent.mkdir(exist_ok=True)
        
        # File extensions to monitor
        self.monitored_extensions = {'.py', '.md', '.yaml', '.yml', '.json'}
        
        # Directories to exclude
        self.excluded_dirs = {
            '__pycache__', '.git', 'node_modules', 'venv', 'env',
            'archive', 'backup', 'deprecated', 'superseded', 'logs'
        }
        
        # Track recent violations
        self.recent_violations: Dict[str, List[Dict]] = {}
    
    def should_monitor_file(self, file_path: Path) -> bool:
        """Check if file should be monitored"""
        # Check extension
        if file_path.suffix.lower() not in self.monitored_extensions:
            return False
        
        # Check if in excluded directory
        for part in file_path.parts:
            if part in self.excluded_dirs:
                return False
        
        return True
    
    def check_file_compliance(self, file_path: Path) -> Dict:
        """Check compliance for a specific file"""
        try:
            # Run compliance checker on single file
            result = subprocess.run([
                sys.executable, str(self.compliance_checker_path),
                '--paths', str(file_path)
            ], capture_output=True, text=True, cwd=self.workspace_root)
            
            if result.returncode != 0:
                # Violations found
                violations = self.parse_violations(result.stdout)
                return {
                    "file": str(file_path),
                    "compliant": False,
                    "violations": violations,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # No violations
                return {
                    "file": str(file_path),
                    "compliant": True,
                    "violations": [],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "file": str(file_path),
                "compliant": False,
                "violations": [{"error": f"Check failed: {str(e)}"}],
                "timestamp": datetime.now().isoformat()
            }
    
    def parse_violations(self, output: str) -> List[Dict]:
        """Parse violations from compliance checker output"""
        violations = []
        lines = output.split('\n')
        
        current_violation = None
        for line in lines:
            line = line.strip()
            
            if line.startswith('**') and line.endswith('**'):
                # File path
                if current_violation:
                    violations.append(current_violation)
                current_violation = {
                    'file': line.strip('*'),
                    'details': []
                }
            
            elif current_violation and line.startswith('- '):
                # Violation detail
                current_violation['details'].append(line[2:])
        
        if current_violation:
            violations.append(current_violation)
        
        return violations
    
    def log_violations(self, file_path: str, violations: List[Dict]):
        """Log violations to file for tracking"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "file": file_path,
            "violations": violations
        }
        
        # Load existing log
        if self.violation_log.exists():
            try:
                with open(self.violation_log, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                log_data = {"entries": []}
        else:
            log_data = {"entries": []}
        
        # Add new entry
        log_data["entries"].append(log_entry)
        
        # Keep only last 100 entries
        if len(log_data["entries"]) > 100:
            log_data["entries"] = log_data["entries"][-100:]
        
        # Save log
        with open(self.violation_log, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def display_violations(self, file_path: Path, violations: List[Dict]):
        """Display violations in a user-friendly format"""
        print(f"\n🚨 COMPLIANCE VIOLATIONS DETECTED")
        print(f"📁 File: {file_path}")
        print("=" * 60)
        
        for violation in violations:
            for detail in violation.get('details', []):
                if 'exceeds' in detail.lower() or 'error' in detail.lower():
                    print(f"❌ {detail}")
                elif 'warning' in detail.lower():
                    print(f"⚠️  {detail}")
                else:
                    print(f"ℹ️  {detail}")
        
        print("=" * 60)
        print("💡 Quick fixes:")
        print(f"   - Check: python tools_utilities/rule_compliance_checker.py --paths {file_path}")
        print("   - Fix: python tools_utilities/rule_compliance_checker.py --fix")
        print()
    
    def auto_check_file(self, file_path: Path):
        """Automatically check a file for compliance"""
        if not self.should_monitor_file(file_path):
            return
        
        # Check cooldown
        current_time = time.time()
        file_str = str(file_path)
        if (file_str in self.last_check_times and 
            current_time - self.last_check_times[file_str] < self.check_cooldown):
            return
        
        self.last_check_times[file_str] = current_time
        
        # Run compliance check
        result = self.check_file_compliance(file_path)
        
        if not result["compliant"] and result["violations"]:
            # Display violations immediately
            self.display_violations(file_path, result["violations"])
            
            # Log violations
            self.log_violations(file_str, result["violations"])
            
            # Store in recent violations
            self.recent_violations[file_str] = result["violations"]
        else:
            # Clear any previous violations for this file
            if file_str in self.recent_violations:
                del self.recent_violations[file_str]
                print(f"✅ {file_path} - Compliance restored")
    
    def get_violation_summary(self) -> Dict:
        """Get summary of current violations"""
        return {
            "files_with_violations": len(self.recent_violations),
            "total_violations": sum(len(v) for v in self.recent_violations.values()),
            "violations_by_file": self.recent_violations
        }
    
    def start_monitoring(self):
        """Start automatic compliance monitoring"""
        print("🔍 Starting automatic compliance monitoring...")
        print(f"📁 Monitoring: {self.workspace_root}")
        print("📋 File types: .py, .md, .yaml, .yml, .json")
        print("⏹️  Press Ctrl+C to stop")
        print()
        
        # Create file system event handler
        class ComplianceEventHandler(FileSystemEventHandler):
            def __init__(self, monitor):
                self.monitor = monitor
            
            def on_modified(self, event):
                if not event.is_directory:
                    self.monitor.auto_check_file(Path(event.src_path))
            
            def on_created(self, event):
                if not event.is_directory:
                    # Small delay to ensure file is fully written
                    time.sleep(0.5)
                    self.monitor.auto_check_file(Path(event.src_path))
        
        # Start file system monitoring
        observer = Observer()
        observer.schedule(ComplianceEventHandler(self), str(self.workspace_root), recursive=True)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping compliance monitor...")
            observer.stop()
        
        observer.join()
        print("✅ Compliance monitor stopped")
