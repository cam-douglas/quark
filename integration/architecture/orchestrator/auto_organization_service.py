#!/usr/bin/env python3
"""
Auto Organization Service - Background File Organization
Purpose: Provides automatic file organization as a background service
Inputs: File system events, periodic triggers
Outputs: Organized file structure, real-time organization
Seeds: N/A (service daemon)
Dependencies: watchdog, threading, organization_agent
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not available. Install with: pip install watchdog")

from organization_agent import OrganizationAgent

class AutoOrganizationEventHandler(FileSystemEventHandler):
    """File system event handler for automatic organization"""
    
    def __init__(self, organization_agent: OrganizationAgent):
        super().__init__()
        self.agent = organization_agent
        self.logger = logging.getLogger('auto_organization')
        self.debounce_delay = 2.0  # Seconds to wait before processing
        self.pending_files = {}  # File path -> timestamp
        self.processing = False
        
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self._schedule_file_processing(event.src_path, "created")
            
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self._schedule_file_processing(event.src_path, "modified")
            
    def on_moved(self, event):
        """Handle file move events"""
        if not event.is_directory:
            # Only process if moved to root (indicating user placement)
            dest_parent = Path(event.dest_path).parent
            if str(dest_parent) == str(self.agent.root_path):
                self._schedule_file_processing(event.dest_path, "moved_to_root")
    
    def _schedule_file_processing(self, file_path: str, event_type: str):
        """Schedule file for processing with debouncing"""
        file_path_obj = Path(file_path)
        
        # Only process files in root directory
        if file_path_obj.parent != self.agent.root_path:
            return
            
        # Skip hidden files and directories
        if file_path_obj.name.startswith('.'):
            return
            
        # Skip files that should stay in root
        if file_path_obj.name in self.agent.root_files:
            return
            
        # Skip temporary files
        for pattern in self.agent.temp_patterns:
            import re
            if re.match(pattern, file_path_obj.name, re.IGNORECASE):
                return
        
        # Schedule processing with debounce
        current_time = time.time()
        self.pending_files[file_path] = {
            "timestamp": current_time,
            "event_type": event_type
        }
        
        self.logger.info(f"Scheduled for organization: {file_path_obj.name} ({event_type})")
        
        # Start processing thread if not already running
        if not self.processing:
            threading.Thread(target=self._process_pending_files, daemon=True).start()
    
    def _process_pending_files(self):
        """Process files that have been pending for debounce delay"""
        self.processing = True
        
        try:
            while self.pending_files:
                current_time = time.time()
                files_to_process = []
                
                # Find files that have been pending long enough
                for file_path, info in list(self.pending_files.items()):
                    if current_time - info["timestamp"] >= self.debounce_delay:
                        files_to_process.append((file_path, info))
                        del self.pending_files[file_path]
                
                # Process the files
                for file_path, info in files_to_process:
                    self._process_file(file_path, info["event_type"])
                
                # Sleep before checking again
                time.sleep(1.0)
                
        finally:
            self.processing = False
    
    def _process_file(self, file_path: str, event_type: str):
        """Process a single file for organization"""
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return
                
            self.logger.info(f"Processing file: {file_path_obj.name} ({event_type})")
            
            # Organize the file
            moved = self.agent.organize_file(file_path_obj, dry_run=False)
            
            if moved:
                self.logger.info(f"Auto-organized: {file_path_obj.name}")
            else:
                self.logger.debug(f"No organization needed: {file_path_obj.name}")
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")


class AutoOrganizationService:
    """Background service for automatic file organization"""
    
    def __init__(self, root_path: str = "/Users/camdouglas/quark"):
        self.root_path = Path(root_path)
        self.agent = OrganizationAgent(str(root_path))
        self.running = False
        self.observer = None
        self.periodic_thread = None
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger('auto_organization')
        
        # Service configuration
        self.config = {
            "enabled": True,
            "watch_filesystem": WATCHDOG_AVAILABLE,
            "periodic_interval": 300,  # 5 minutes
            "semantic_organization": True,
            "auto_cleanup": True,
            "max_file_age_hours": 1  # Files older than 1 hour trigger organization
        }
        
        # Load configuration if it exists
        self.load_config()
        
    def setup_logging(self):
        """Setup logging for the service"""
        log_dir = self.root_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('auto_organization')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "auto_organization_service.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def load_config(self):
        """Load service configuration"""
        config_file = self.root_path / "configs" / "auto_organization.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                    self.logger.info("Loaded custom configuration")
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}")
    
    def save_config(self):
        """Save current configuration"""
        config_dir = self.root_path / "configs"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "auto_organization.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Could not save config: {e}")
    
    def start(self):
        """Start the auto organization service"""
        if self.running:
            self.logger.warning("Service already running")
            return
            
        if not self.config["enabled"]:
            self.logger.info("Service disabled in configuration")
            return
        
        self.running = True
        self.logger.info("Starting Auto Organization Service")
        
        # Start file system watcher if available
        if self.config["watch_filesystem"] and WATCHDOG_AVAILABLE:
            self.start_filesystem_watcher()
        else:
            self.logger.warning("File system watching disabled or unavailable")
        
        # Start periodic organization
        self.start_periodic_organization()
        
        self.logger.info("Auto Organization Service started successfully")
    
    def stop(self):
        """Stop the auto organization service"""
        if not self.running:
            return
            
        self.running = False
        self.logger.info("Stopping Auto Organization Service")
        
        # Stop file system watcher
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        # Stop periodic thread
        if self.periodic_thread:
            self.periodic_thread.join(timeout=5)
            self.periodic_thread = None
        
        self.logger.info("Auto Organization Service stopped")
    
    def start_filesystem_watcher(self):
        """Start watching the file system for changes"""
        try:
            event_handler = AutoOrganizationEventHandler(self.agent)
            self.observer = Observer()
            self.observer.schedule(
                event_handler, 
                str(self.root_path), 
                recursive=False  # Only watch root directory
            )
            self.observer.start()
            self.logger.info("File system watcher started")
        except Exception as e:
            self.logger.error(f"Could not start file system watcher: {e}")
    
    def start_periodic_organization(self):
        """Start periodic organization checks"""
        if self.config["periodic_interval"] <= 0:
            return
            
        def periodic_worker():
            while self.running:
                try:
                    time.sleep(self.config["periodic_interval"])
                    if self.running:
                        self.run_periodic_organization()
                except Exception as e:
                    self.logger.error(f"Error in periodic organization: {e}")
        
        self.periodic_thread = threading.Thread(target=periodic_worker, daemon=True)
        self.periodic_thread.start()
        self.logger.info(f"Periodic organization started (interval: {self.config['periodic_interval']}s)")
    
    def run_periodic_organization(self):
        """Run periodic organization check"""
        self.logger.info("Running periodic organization check")
        
        try:
            # Run validation check
            report = self.agent.validate_structure()
            
            if not report["valid"]:
                self.logger.info(f"Directory structure needs organization: {len(report['issues'])} issues")
                
                # Run organization based on configuration
                if self.config["semantic_organization"]:
                    summary = self.agent.organize_by_semantic_clusters(dry_run=False)
                    self.logger.info(f"Semantic organization: {summary['moved']} files moved")
                else:
                    summary = self.agent.scan_and_organize(dry_run=False)
                    self.logger.info(f"Pattern organization: {summary['moved']} files moved")
            else:
                self.logger.debug("Directory structure is valid")
                
            # Auto-cleanup if enabled
            if self.config["auto_cleanup"]:
                self.run_cleanup()
                
        except Exception as e:
            self.logger.error(f"Error in periodic organization: {e}")
    
    def run_cleanup(self):
        """Run automatic cleanup of temporary files"""
        try:
            # Find and remove temporary files
            temp_files = []
            for pattern in self.agent.temp_patterns:
                import re
                for file_path in self.root_path.iterdir():
                    if file_path.is_file() and re.match(pattern, file_path.name, re.IGNORECASE):
                        temp_files.append(file_path)
            
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    self.logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                except Exception as e:
                    self.logger.warning(f"Could not remove {temp_file.name}: {e}")
                    
            if temp_files:
                self.logger.info(f"Cleaned up {len(temp_files)} temporary files")
                
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "running": self.running,
            "config": self.config,
            "filesystem_watcher": self.observer is not None and self.observer.is_alive() if self.observer else False,
            "periodic_thread": self.periodic_thread is not None and self.periodic_thread.is_alive() if self.periodic_thread else False,
            "watchdog_available": WATCHDOG_AVAILABLE,
            "last_check": datetime.now().isoformat()
        }
    
    def enable_semantic_organization(self, enabled: bool = True):
        """Enable or disable semantic organization"""
        self.config["semantic_organization"] = enabled
        self.save_config()
        self.logger.info(f"Semantic organization {'enabled' if enabled else 'disabled'}")
    
    def set_periodic_interval(self, interval_seconds: int):
        """Set periodic organization interval"""
        self.config["periodic_interval"] = interval_seconds
        self.save_config()
        self.logger.info(f"Periodic interval set to {interval_seconds} seconds")
        
        # Restart periodic organization with new interval
        if self.running:
            if self.periodic_thread:
                # Note: In a real implementation, you'd want to properly signal the thread to restart
                self.logger.info("Restart service to apply new periodic interval")


# Global service instance
_service_instance: Optional[AutoOrganizationService] = None

def get_service() -> AutoOrganizationService:
    """Get the global service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = AutoOrganizationService()
    return _service_instance

def start_auto_organization():
    """Start the auto organization service"""
    service = get_service()
    service.start()
    return service

def stop_auto_organization():
    """Stop the auto organization service"""
    service = get_service()
    service.stop()

def get_auto_organization_status():
    """Get service status"""
    service = get_service()
    return service.get_status()


def main():
    """CLI interface for the auto organization service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Organization Service")
    parser.add_argument("--start", action="store_true", help="Start the service")
    parser.add_argument("--stop", action="store_true", help="Stop the service")
    parser.add_argument("--status", action="store_true", help="Show service status")
    parser.add_argument("--enable-semantic", action="store_true", help="Enable semantic organization")
    parser.add_argument("--disable-semantic", action="store_true", help="Disable semantic organization")
    parser.add_argument("--interval", type=int, help="Set periodic interval in seconds")
    
    args = parser.parse_args()
    
    service = get_service()
    
    if args.start:
        service.start()
        print("Auto organization service started")
        try:
            # Keep running until interrupted
            while service.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping service...")
            service.stop()
            
    elif args.stop:
        service.stop()
        print("Auto organization service stopped")
        
    elif args.status:
        status = service.get_status()
        print("\n=== Auto Organization Service Status ===")
        print(f"Running: {status['running']}")
        print(f"Filesystem Watcher: {status['filesystem_watcher']}")
        print(f"Periodic Thread: {status['periodic_thread']}")
        print(f"Watchdog Available: {status['watchdog_available']}")
        print(f"Semantic Organization: {status['config']['semantic_organization']}")
        print(f"Periodic Interval: {status['config']['periodic_interval']}s")
        print(f"Last Check: {status['last_check']}")
        
    elif args.enable_semantic:
        service.enable_semantic_organization(True)
        print("Semantic organization enabled")
        
    elif args.disable_semantic:
        service.enable_semantic_organization(False)
        print("Semantic organization disabled")
        
    elif args.interval:
        service.set_periodic_interval(args.interval)
        print(f"Periodic interval set to {args.interval} seconds")
        
    else:
        print("Available commands:")
        print("  --start            : Start the service")
        print("  --stop             : Stop the service")
        print("  --status           : Show service status")
        print("  --enable-semantic  : Enable semantic organization")
        print("  --disable-semantic : Disable semantic organization")
        print("  --interval N       : Set periodic interval to N seconds")
        print("\nUse --help for detailed usage information.")


if __name__ == "__main__":
    main()
