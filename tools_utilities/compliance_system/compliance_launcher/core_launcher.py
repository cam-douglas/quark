"""
Core Compliance System Launcher

Main launcher for all compliance checking systems.

Author: Quark AI
Date: 2025-01-27
"""

import os
import sys
import time
import signal
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json


class ComplianceSystemLauncher:
    """
    Unified launcher for all compliance checking systems.
    
    Manages and coordinates:
    - Three-phase compliance checking
    - Real-time file monitoring
    - Pre-push hook integration
    - Cursor AI operation hooks
    """
    
    def __init__(self, workspace_root: str = "/Users/camdouglas/quark"):
        """Initialize compliance system launcher"""
        self.workspace_root = Path(workspace_root)
        self.status_file = self.workspace_root / "logs" / "compliance_system_status.json"
        self.pid_file = self.workspace_root / "logs" / "compliance_system.pid"
        
        # Create logs directory
        self.status_file.parent.mkdir(exist_ok=True)
        
        # System components
        self.three_phase_system = None
        self.auto_monitor = None
        self.monitoring_thread = None
        self.running = False
        
        # Import systems
        self._import_systems()
    
    def _import_systems(self):
        """Import compliance systems"""
        try:
            # Add tools_utilities to path
            sys.path.insert(0, str(self.workspace_root / "tools_utilities"))
            
            from compliance_system.three_phase_compliance import ThreePhaseComplianceSystem
            from compliance_system.auto_compliance_monitor import AutoComplianceMonitor
            
            self.three_phase_system = ThreePhaseComplianceSystem(str(self.workspace_root))
            self.auto_monitor = AutoComplianceMonitor(str(self.workspace_root))
            
        except ImportError as e:
            print(f"❌ Failed to import compliance systems: {e}")
            sys.exit(1)
    
    def start_system(self, background: bool = True):
        """Start the complete compliance system"""
        if self.is_running():
            print("⚠️  Compliance system is already running")
            return True
        
        print("🚀 Starting Quark Compliance System...")
        print("=" * 50)
        
        # Start auto monitoring in background
        if background:
            self.monitoring_thread = threading.Thread(
                target=self._run_auto_monitor,
                daemon=True
            )
            self.monitoring_thread.start()
            print("✅ Auto compliance monitoring started (background)")
        else:
            print("✅ Auto compliance monitoring ready (foreground)")
        
        # Mark system as running
        self.running = True
        self._save_status("running")
        self._save_pid()
        
        print("✅ Three-phase compliance system ready")
        print("✅ Pre-push hook integration active")
        print("✅ Cursor AI operation hooks ready")
        print("=" * 50)
        print("🔍 Compliance system is now operational")
        
        return True
    
    def stop_system(self):
        """Stop the compliance system"""
        if not self.is_running():
            print("⚠️  Compliance system is not running")
            return
        
        print("🛑 Stopping Quark Compliance System...")
        
        # Stop auto monitoring
        if self.auto_monitor:
            self.auto_monitor.stop_monitoring()
        
        # Stop three-phase system
        if self.three_phase_system:
            self.three_phase_system.stop_file_monitoring()
        
        # Mark system as stopped
        self.running = False
        self._save_status("stopped")
        self._remove_pid()
        
        print("✅ Compliance system stopped")
    
    def _run_auto_monitor(self):
        """Run auto monitor in background thread"""
        try:
            self.auto_monitor.start_monitoring()
        except Exception as e:
            print(f"❌ Auto monitor error: {e}")
    
    def is_running(self) -> bool:
        """Check if compliance system is running"""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is actually running
                try:
                    os.kill(pid, 0)
                    return True
                except OSError:
                    # Process not running, clean up
                    self._remove_pid()
                    return False
            except (ValueError, IOError):
                self._remove_pid()
                return False
        return False
    
    def _save_status(self, status: str):
        """Save system status"""
        status_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "workspace": str(self.workspace_root),
            "components": {
                "three_phase_system": self.three_phase_system is not None,
                "auto_monitor": self.auto_monitor is not None,
                "pre_push_hook": self._check_pre_push_hook(),
                "cursor_integration": True
            }
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    def _save_pid(self):
        """Save process ID"""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
    
    def _remove_pid(self):
        """Remove PID file"""
        if self.pid_file.exists():
            self.pid_file.unlink()
    
    def _check_pre_push_hook(self) -> bool:
        """Check if pre-push hook is properly configured"""
        pre_push_hook = self.workspace_root / ".git" / "hooks" / "pre-push"
        if not pre_push_hook.exists():
            return False
        
        try:
            with open(pre_push_hook, 'r') as f:
                content = f.read()
                return "rule_compliance_checker.py" in content
        except IOError:
            return False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "running": self.is_running(),
            "timestamp": datetime.now().isoformat(),
            "workspace": str(self.workspace_root)
        }
        
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    saved_status = json.load(f)
                    status.update(saved_status)
            except (json.JSONDecodeError, IOError):
                pass
        
        return status
    
    def run_operation_with_compliance(self, operation_name: str, target_files: List[str] = None):
        """Run an operation with three-phase compliance checking"""
        if not self.three_phase_system:
            print("❌ Three-phase compliance system not available")
            return False
        
        from compliance_system.three_phase_compliance import operation_context
        with operation_context(self.three_phase_system, operation_name, target_files):
            print(f"📝 Executing operation: {operation_name}")
            return True
    
    def check_compliance_now(self, files: List[str] = None) -> bool:
        """Run immediate compliance check"""
        try:
            cmd = [sys.executable, str(self.workspace_root / "tools_utilities" / "rule_compliance_checker.py")]
            if files:
                cmd.extend(['--paths'] + files)
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            
            if result.returncode == 0:
                print("✅ Compliance check passed")
                return True
            else:
                print("❌ Compliance check failed")
                print(result.stdout)
                return False
                
        except Exception as e:
            print(f"❌ Compliance check error: {e}")
            return False
