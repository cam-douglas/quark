"""
File Monitor for Three-Phase Compliance

Handles real-time file monitoring during operations.

Author: Quark AI
Date: 2025-01-27
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

if TYPE_CHECKING:
    from .core_system import ThreePhaseComplianceSystem


class ComplianceFileMonitor:
    """File system monitor for compliance checking"""
    
    def __init__(self, compliance_system: 'ThreePhaseComplianceSystem'):
        """Initialize file monitor"""
        self.compliance_system = compliance_system
        self.observer = None
        self.monitoring_active = False
    
    def start_monitoring(self):
        """Start file system monitoring"""
        if self.monitoring_active:
            return
        
        class ComplianceEventHandler(FileSystemEventHandler):
            def __init__(self, monitor):
                self.monitor = monitor
            
            def on_modified(self, event):
                if not event.is_directory:
                    self.monitor.handle_file_change(Path(event.src_path))
            
            def on_created(self, event):
                if not event.is_directory:
                    time.sleep(0.5)  # Ensure file is fully written
                    self.monitor.handle_file_change(Path(event.src_path))
        
        self.observer = Observer()
        self.observer.schedule(
            ComplianceEventHandler(self), 
            str(self.compliance_system.workspace_root), 
            recursive=True
        )
        self.observer.start()
        self.monitoring_active = True
    
    def stop_monitoring(self):
        """Stop file system monitoring"""
        if self.observer and self.monitoring_active:
            self.observer.stop()
            self.observer.join()
            self.monitoring_active = False
            self.observer = None
    
    def handle_file_change(self, file_path: Path):
        """Handle file change event"""
        # Delegate to compliance system
        self.compliance_system.handle_file_change(file_path)
