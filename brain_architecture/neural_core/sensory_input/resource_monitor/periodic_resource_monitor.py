#!/usr/bin/env python3
"""
🛡️ Periodic Resource Monitor Service
=====================================

This service runs the Ultimate Resource Authority periodically to maintain
system health and generate resource usage reports. Designed for continuous
monitoring of Mac M2 Max systems with automated optimization.

Features:
- Hourly resource monitoring and optimization
- Automatic report generation
- System health checks
- Resource trend analysis
- Integration with macOS launch services

Author: Quark Resource Management Team
Created: 2025-01-21
"""

import os, sys
import time
import json
import logging
import signal
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import psutil
import threading
import subprocess

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(QUARK_ROOT))

try:
    from 🧠_BRAIN_ARCHITECTURE.01_NEURAL_CORE.resource_monitor import (
        create_integrated_resource_manager,
        IntegratedResourceConfig
    )
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError as e:
    # Try alternative import paths
    try:
        sys.path.append(str(QUARK_ROOT / "brain_modules" / "resource_monitor"))
        from ultimate_resource_authority import create_ultimate_authority
        from cloud_offload_authority import create_cloud_offload_authority
        RESOURCE_MONITORING_AVAILABLE = True
    except ImportError:
        print(f"Warning: Resource monitoring not available: {e}")
        RESOURCE_MONITORING_AVAILABLE = False

class PeriodicResourceMonitor:
    """
    Periodic resource monitoring service for Mac M2 Max.
    
    This service runs continuously and performs resource monitoring
    and optimization at regular intervals.
    """
    
    def __init__(self, 
                 monitor_interval_hours: float = 1.0,
                 enable_cloud_offload: bool = True,
                 enable_reports: bool = True,
                 log_level: str = "INFO"):
        
        self.monitor_interval_hours = monitor_interval_hours
        self.enable_cloud_offload = enable_cloud_offload
        self.enable_reports = enable_reports
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Service state
        self.is_running = False
        self.monitor_thread = None
        self.last_check_time = None
        self.check_count = 0
        
        # Resource manager
        self.resource_manager = None
        
        # Statistics
        self.service_stats = {
            'start_time': None,
            'total_checks': 0,
            'optimizations_applied': 0,
            'cloud_jobs_executed': 0,
            'reports_generated': 0,
            'average_memory_percent': 0.0,
            'average_cpu_percent': 0.0,
            'peak_memory_percent': 0.0,
            'peak_cpu_percent': 0.0
        }
        
        # Create directories
        self.log_dir = QUARK_ROOT / "logs" / "periodic_monitoring"
        self.report_dir = QUARK_ROOT / "reports" / "resource_monitoring"
        self.config_dir = QUARK_ROOT / "configs" / "monitoring"
        
        for directory in [self.log_dir, self.report_dir, self.config_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🛡️ Periodic Resource Monitor initialized")
        self.logger.info(f"Monitor interval: {monitor_interval_hours} hours")
        self.logger.info(f"Cloud offload: {'enabled' if enable_cloud_offload else 'disabled'}")
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging for the periodic service."""
        logger = logging.getLogger('PeriodicResourceMonitor')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            log_dir = QUARK_ROOT / "logs" / "periodic_monitoring"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Daily log file rotation
            log_file = log_dir / f"resource_monitor_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
    
    def initialize_resource_manager(self) -> bool:
        """Initialize the resource manager."""
        if not RESOURCE_MONITORING_AVAILABLE:
            self.logger.error("Resource monitoring not available")
            return False
        
        try:
            # Create resource manager with fallback to basic monitoring
            if 'create_integrated_resource_manager' in globals():
                # Full featured resource manager
                self.resource_manager = create_integrated_resource_manager()
            elif 'create_ultimate_authority' in globals():
                # Basic resource authority
                self.resource_manager = create_ultimate_authority()
            else:
                # Fallback - will use basic psutil monitoring
                self.resource_manager = None
            self.logger.info("Resource manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize resource manager: {e}")
            return False
    
    def perform_resource_check(self) -> Dict[str, Any]:
        """Perform a comprehensive resource check."""
        self.logger.info("🔍 Performing periodic resource check")
        
        check_start_time = time.time()
        check_results = {
            'timestamp': datetime.now().isoformat(),
            'check_number': self.check_count + 1,
            'success': False,
            'resource_metrics': {},
            'optimizations_applied': 0,
            'cloud_jobs_submitted': 0,
            'recommendations': [],
            'alerts': [],
            'execution_time_seconds': 0.0
        }
        
        try:
            # Get current resource metrics
            if self.resource_manager:
                current_metrics = self.resource_manager.resource_authority.get_current_metrics()
                
                check_results['resource_metrics'] = {
                    'memory_used_gb': current_metrics.memory_used_gb,
                    'memory_percent': current_metrics.memory_percent,
                    'cpu_percent': current_metrics.cpu_percent,
                    'temperature_celsius': current_metrics.temperature_celsius,
                    'disk_io_read_mbps': current_metrics.disk_io_read_mbps,
                    'disk_io_write_mbps': current_metrics.disk_io_write_mbps,
                    'active_processes': current_metrics.active_processes,
                    'swap_used_gb': current_metrics.swap_used_gb
                }
                
                # Update statistics
                self._update_service_stats(current_metrics)
                
                # Assess resource status
                severity, issues = self.resource_manager.resource_authority.assess_resource_status(current_metrics)
                
                # Apply optimizations if needed
                if severity in ['warning', 'critical']:
                    self.logger.warning(f"Resource issues detected: {severity}")
                    
                    optimizations = self.resource_manager.resource_authority.optimize_system_resources(
                        current_metrics, severity
                    )
                    
                    check_results['optimizations_applied'] = len(optimizations)
                    check_results['alerts'] = issues
                    
                    for optimization in optimizations:
                        self.logger.info(f"Applied optimization: {optimization.description}")
                
                # Get recommendations
                recommendations = self.resource_manager.get_optimization_recommendations()
                check_results['recommendations'] = recommendations.get('recommendations', [])
                
                # Check cloud status
                cloud_status = self.resource_manager.cloud_authority.get_system_status()
                check_results['cloud_jobs_submitted'] = cloud_status.get('completed_jobs', 0)
                
                check_results['success'] = True
                
            else:
                # Fallback to basic psutil monitoring
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1.0)
                
                check_results['resource_metrics'] = {
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'cpu_percent': cpu_percent,
                    'temperature_celsius': 0.0,  # Not available without resource manager
                    'disk_io_read_mbps': 0.0,
                    'disk_io_write_mbps': 0.0,
                    'active_processes': len(psutil.pids()),
                    'swap_used_gb': psutil.swap_memory().used / (1024**3)
                }
                
                # Basic alerts
                if memory.percent > 80:
                    check_results['alerts'].append(f"High memory usage: {memory.percent:.1f}%")
                if cpu_percent > 85:
                    check_results['alerts'].append(f"High CPU usage: {cpu_percent:.1f}%")
                
                check_results['success'] = True
            
            execution_time = time.time() - check_start_time
            check_results['execution_time_seconds'] = execution_time
            
            self.check_count += 1
            self.last_check_time = datetime.now()
            
            # Log summary
            metrics = check_results['resource_metrics']
            self.logger.info(
                f"✅ Check completed: Memory {metrics['memory_percent']:.1f}%, "
                f"CPU {metrics['cpu_percent']:.1f}%, "
                f"Optimizations {check_results['optimizations_applied']}, "
                f"Time {execution_time:.1f}s"
            )
            
            return check_results
            
        except Exception as e:
            self.logger.error(f"Error during resource check: {e}")
            check_results['error'] = str(e)
            check_results['execution_time_seconds'] = time.time() - check_start_time
            return check_results
    
    def _update_service_stats(self, metrics):
        """Update service statistics with current metrics."""
        self.service_stats['total_checks'] += 1
        
        # Update averages
        total_checks = self.service_stats['total_checks']
        current_avg_memory = self.service_stats['average_memory_percent']
        current_avg_cpu = self.service_stats['average_cpu_percent']
        
        self.service_stats['average_memory_percent'] = (
            (current_avg_memory * (total_checks - 1) + metrics.memory_percent) / total_checks
        )
        self.service_stats['average_cpu_percent'] = (
            (current_avg_cpu * (total_checks - 1) + metrics.cpu_percent) / total_checks
        )
        
        # Update peaks
        self.service_stats['peak_memory_percent'] = max(
            self.service_stats['peak_memory_percent'], 
            metrics.memory_percent
        )
        self.service_stats['peak_cpu_percent'] = max(
            self.service_stats['peak_cpu_percent'], 
            metrics.cpu_percent
        )
    
    def generate_periodic_report(self, check_results: Dict[str, Any]) -> Optional[Path]:
        """Generate a periodic monitoring report."""
        if not self.enable_reports:
            return None
        
        try:
            report_data = {
                'metadata': {
                    'report_type': 'periodic_monitoring',
                    'generated_at': datetime.now().isoformat(),
                    'service_uptime_hours': self._get_service_uptime_hours(),
                    'monitor_interval_hours': self.monitor_interval_hours
                },
                'service_statistics': self.service_stats.copy(),
                'latest_check': check_results,
                'system_info': {
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                    'cpu_count': psutil.cpu_count(),
                    'system': 'Mac M2 Max'
                }
            }
            
            # Add resource manager stats if available
            if self.resource_manager:
                comprehensive_status = self.resource_manager.get_comprehensive_status()
                report_data['comprehensive_status'] = comprehensive_status
            
            # Generate report file
            report_filename = f"periodic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file = self.report_dir / report_filename
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.service_stats['reports_generated'] += 1
            self.logger.info(f"📊 Report generated: {report_file}")
            
            # Keep only last 24 reports (daily cleanup)
            self._cleanup_old_reports()
            
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return None
    
    def _cleanup_old_reports(self):
        """Clean up old report files to save disk space."""
        try:
            report_files = sorted(self.report_dir.glob("periodic_report_*.json"))
            
            # Keep only the last 24 reports
            if len(report_files) > 24:
                for old_file in report_files[:-24]:
                    old_file.unlink()
                    self.logger.debug(f"Cleaned up old report: {old_file}")
                    
        except Exception as e:
            self.logger.warning(f"Error cleaning up old reports: {e}")
    
    def _get_service_uptime_hours(self) -> float:
        """Get service uptime in hours."""
        if self.service_stats['start_time']:
            uptime_seconds = (datetime.now() - self.service_stats['start_time']).total_seconds()
            return uptime_seconds / 3600.0
        return 0.0
    
    def start_service(self):
        """Start the periodic monitoring service."""
        if self.is_running:
            self.logger.warning("Service already running")
            return
        
        self.logger.info("🚀 Starting periodic resource monitoring service")
        
        # Initialize resource manager
        if not self.initialize_resource_manager():
            self.logger.error("Failed to initialize resource manager, continuing with basic monitoring")
        
        # Set service state
        self.is_running = True
        self.service_stats['start_time'] = datetime.now()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"✅ Service started - monitoring every {self.monitor_interval_hours} hours")
    
    def stop_service(self):
        """Stop the periodic monitoring service."""
        if not self.is_running:
            return
        
        self.logger.info("🛑 Stopping periodic resource monitoring service")
        
        self.is_running = False
        
        # Stop resource manager
        if self.resource_manager:
            self.resource_manager.stop_integrated_management()
        
        # Wait for monitoring thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
        
        # Generate final report
        if self.enable_reports:
            final_report_data = {
                'metadata': {
                    'report_type': 'service_shutdown',
                    'generated_at': datetime.now().isoformat(),
                    'total_uptime_hours': self._get_service_uptime_hours()
                },
                'final_statistics': self.service_stats.copy()
            }
            
            final_report_file = self.report_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(final_report_file, 'w') as f:
                json.dump(final_report_data, f, indent=2, default=str)
            
            self.logger.info(f"📊 Final report generated: {final_report_file}")
        
        self.logger.info("✅ Service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                # Perform resource check
                check_results = self.perform_resource_check()
                
                # Generate report
                if self.enable_reports:
                    self.generate_periodic_report(check_results)
                
                # Wait for next interval
                interval_seconds = self.monitor_interval_hours * 3600
                
                # Sleep in small increments to allow clean shutdown
                elapsed = 0
                while elapsed < interval_seconds and self.is_running:
                    time.sleep(min(60, interval_seconds - elapsed))  # Sleep max 1 minute at a time
                    elapsed += 60
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
        
        self.logger.info("Monitoring loop ended")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            'is_running': self.is_running,
            'uptime_hours': self._get_service_uptime_hours(),
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'check_count': self.check_count,
            'monitor_interval_hours': self.monitor_interval_hours,
            'statistics': self.service_stats.copy(),
            'resource_manager_available': self.resource_manager is not None
        }


def setup_signal_handlers(service: PeriodicResourceMonitor):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        service.stop_service()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def create_launchd_plist(interval_hours: float = 1.0, 
                        enable_cloud_offload: bool = True,
                        user_home: Path = Path.home()) -> Path:
    """Create a macOS LaunchAgent plist file for automatic startup."""
    
    # LaunchAgent directory
    launch_agents_dir = user_home / "Library" / "LaunchAgents"
    launch_agents_dir.mkdir(exist_ok=True)
    
    # Plist content
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quark.resource.monitor</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/python3</string>
        <string>{QUARK_ROOT}/brain_modules/resource_monitor/periodic_resource_monitor.py</string>
        <string>--daemon</string>
        <string>--interval</string>
        <string>{interval_hours}</string>
        {'<string>--enable-cloud-offload</string>' if enable_cloud_offload else ''}
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>{QUARK_ROOT}/logs/periodic_monitoring/launchd_stdout.log</string>
    
    <key>StandardErrorPath</key>
    <string>{QUARK_ROOT}/logs/periodic_monitoring/launchd_stderr.log</string>
    
    <key>WorkingDirectory</key>
    <string>{QUARK_ROOT}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{QUARK_ROOT}</string>
    </dict>
    
    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>"""
    
    # Write plist file
    plist_file = launch_agents_dir / "com.quark.resource.monitor.plist"
    with open(plist_file, 'w') as f:
        f.write(plist_content)
    
    return plist_file


def install_service(interval_hours: float = 1.0, enable_cloud_offload: bool = True) -> bool:
    """Install the periodic resource monitor as a macOS service."""
    try:
        print("🔧 Installing periodic resource monitor service...")
        
        # Create LaunchAgent plist
        plist_file = create_launchd_plist(interval_hours, enable_cloud_offload)
        print(f"✅ Created LaunchAgent plist: {plist_file}")
        
        # Load the service
        result = subprocess.run([
            'launchctl', 'load', str(plist_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Service installed and loaded successfully")
            print(f"🛡️ Resource monitoring will run every {interval_hours} hours")
            print(f"☁️ Cloud offload: {'enabled' if enable_cloud_offload else 'disabled'}")
            print()
            print("Service management commands:")
            print(f"  Start:  launchctl load {plist_file}")
            print(f"  Stop:   launchctl unload {plist_file}")
            print(f"  Status: launchctl list | grep com.quark.resource.monitor")
            print()
            print(f"Logs location: {QUARK_ROOT}/logs/periodic_monitoring/")
            print(f"Reports location: {QUARK_ROOT}/reports/resource_monitoring/")
            return True
        else:
            print(f"❌ Failed to load service: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing service: {e}")
        return False


def uninstall_service() -> bool:
    """Uninstall the periodic resource monitor service."""
    try:
        print("🗑️ Uninstalling periodic resource monitor service...")
        
        plist_file = Path.home() / "Library" / "LaunchAgents" / "com.quark.resource.monitor.plist"
        
        if plist_file.exists():
            # Unload the service
            result = subprocess.run([
                'launchctl', 'unload', str(plist_file)
            ], capture_output=True, text=True)
            
            # Remove plist file
            plist_file.unlink()
            
            print("✅ Service uninstalled successfully")
            return True
        else:
            print("⚠️ Service not found")
            return False
            
    except Exception as e:
        print(f"❌ Error uninstalling service: {e}")
        return False


def main():
    """Main entry point for the periodic resource monitor."""
    parser = argparse.ArgumentParser(description="Periodic Resource Monitor for Mac M2 Max")
    
    parser.add_argument("--daemon", action="store_true", 
                       help="Run as daemon service")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Monitoring interval in hours (default: 1.0)")
    parser.add_argument("--enable-cloud-offload", action="store_true", default=True,
                       help="Enable cloud offloading (default: True)")
    parser.add_argument("--disable-cloud-offload", action="store_true",
                       help="Disable cloud offloading")
    parser.add_argument("--enable-reports", action="store_true", default=True,
                       help="Enable report generation (default: True)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    # Service management
    parser.add_argument("--install", action="store_true",
                       help="Install as macOS LaunchAgent service")
    parser.add_argument("--uninstall", action="store_true",
                       help="Uninstall macOS LaunchAgent service")
    parser.add_argument("--status", action="store_true",
                       help="Show service status")
    
    args = parser.parse_args()
    
    # Handle cloud offload setting
    enable_cloud_offload = args.enable_cloud_offload and not args.disable_cloud_offload
    
    # Service management commands
    if args.install:
        success = install_service(args.interval, enable_cloud_offload)
        sys.exit(0 if success else 1)
    
    if args.uninstall:
        success = uninstall_service()
        sys.exit(0 if success else 1)
    
    if args.status:
        # Check if service is running
        result = subprocess.run([
            'launchctl', 'list', 'com.quark.resource.monitor'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Service is running")
            print(result.stdout)
        else:
            print("❌ Service is not running")
        sys.exit(0)
    
    # Create and run service
    service = PeriodicResourceMonitor(
        monitor_interval_hours=args.interval,
        enable_cloud_offload=enable_cloud_offload,
        enable_reports=args.enable_reports,
        log_level=args.log_level
    )
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(service)
    
    if args.daemon:
        # Run as daemon
        print(f"🛡️ Starting resource monitor daemon (interval: {args.interval} hours)")
        service.start_service()
        
        try:
            # Keep main thread alive
            while service.is_running:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            service.stop_service()
    else:
        # Run single check
        print("🔍 Running single resource check...")
        
        if service.initialize_resource_manager():
            check_results = service.perform_resource_check()
            
            if service.enable_reports:
                report_file = service.generate_periodic_report(check_results)
                if report_file:
                    print(f"📊 Report generated: {report_file}")
            
            # Display results
            metrics = check_results['resource_metrics']
            print(f"\nResource Status:")
            print(f"  Memory: {metrics['memory_percent']:.1f}% ({metrics['memory_used_gb']:.1f}GB)")
            print(f"  CPU: {metrics['cpu_percent']:.1f}%")
            print(f"  Temperature: {metrics['temperature_celsius']:.1f}°C")
            print(f"  Active Processes: {metrics['active_processes']}")
            
            if check_results['alerts']:
                print(f"\nAlerts:")
                for alert in check_results['alerts']:
                    print(f"  ⚠️ {alert}")
            
            if check_results['optimizations_applied'] > 0:
                print(f"\nOptimizations Applied: {check_results['optimizations_applied']}")
            
            print(f"\n✅ Check completed in {check_results['execution_time_seconds']:.1f}s")
        else:
            print("❌ Failed to initialize resource manager")
            sys.exit(1)


if __name__ == "__main__":
    main()
