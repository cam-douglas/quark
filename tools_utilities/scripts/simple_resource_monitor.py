#!/usr/bin/env python3
"""
🛡️ Simple Resource Monitor
=========================

A lightweight resource monitor that runs every hour to track and optimize
Mac M2 Max resources. This version focuses on core functionality and
reliability.

Author: Quark Resource Management Team
Created: 2025-01-21
"""

import os, sys
import time
import json
import psutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Configuration
MONITOR_INTERVAL_HOURS = 1.0
MAX_MEMORY_PERCENT = 80.0
MAX_CPU_PERCENT = 85.0
LOG_DIR = Path.home() / "quark_logs" / "resource_monitoring"
REPORT_DIR = Path.home() / "quark_reports" / "resource_monitoring"

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Setup logging for the resource monitor."""
    log_file = LOG_DIR / f"resource_monitor_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def get_system_info():
    """Get comprehensive system information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_percent = psutil.cpu_percent(interval=1.0)
    
    # Get CPU per-core usage
    cpu_per_core = psutil.cpu_percent(interval=1.0, percpu=True)
    
    # Get disk usage
    disk = psutil.disk_usage('/')
    
    # Get top memory processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Sort by memory usage
    top_memory_processes = sorted(processes, 
                                 key=lambda x: x['memory_percent'] or 0, 
                                 reverse=True)[:10]
    
    return {
        'timestamp': datetime.now().isoformat(),
        'memory': {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent
        },
        'swap': {
            'total_gb': swap.total / (1024**3),
            'used_gb': swap.used / (1024**3),
            'percent': swap.percent
        },
        'cpu': {
            'percent': cpu_percent,
            'per_core': cpu_per_core,
            'count': psutil.cpu_count()
        },
        'disk': {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': (disk.used / disk.total) * 100
        },
        'processes': {
            'total': len(processes),
            'top_memory': top_memory_processes
        }
    }

def analyze_system_health(info):
    """Analyze system health and provide recommendations."""
    issues = []
    recommendations = []
    severity = 'normal'
    
    # Memory analysis
    if info['memory']['percent'] >= 90:
        issues.append(f"Critical memory usage: {info['memory']['percent']:.1f}%")
        recommendations.append("Restart high-memory applications")
        recommendations.append("Close unnecessary browser tabs")
        severity = 'critical'
    elif info['memory']['percent'] >= MAX_MEMORY_PERCENT:
        issues.append(f"High memory usage: {info['memory']['percent']:.1f}%")
        recommendations.append("Consider closing some applications")
        if severity != 'critical':
            severity = 'warning'
    
    # CPU analysis
    if info['cpu']['percent'] >= 95:
        issues.append(f"Critical CPU usage: {info['cpu']['percent']:.1f}%")
        recommendations.append("Check for runaway processes")
        severity = 'critical'
    elif info['cpu']['percent'] >= MAX_CPU_PERCENT:
        issues.append(f"High CPU usage: {info['cpu']['percent']:.1f}%")
        recommendations.append("Monitor CPU-intensive applications")
        if severity not in ['critical']:
            severity = 'warning'
    
    # Swap analysis
    if info['swap']['used_gb'] > 1.0:
        issues.append(f"Swap usage detected: {info['swap']['used_gb']:.1f}GB")
        recommendations.append("Consider increasing available memory")
        if severity == 'normal':
            severity = 'warning'
    
    # Disk analysis
    if info['disk']['percent'] >= 95:
        issues.append(f"Critical disk usage: {info['disk']['percent']:.1f}%")
        recommendations.append("Free up disk space immediately")
        severity = 'critical'
    elif info['disk']['percent'] >= 85:
        issues.append(f"High disk usage: {info['disk']['percent']:.1f}%")
        recommendations.append("Consider cleaning up files")
        if severity not in ['critical']:
            severity = 'warning'
    
    # Process analysis
    top_memory_process = info['processes']['top_memory'][0] if info['processes']['top_memory'] else None
    if top_memory_process and top_memory_process['memory_percent'] > 20:
        issues.append(f"High memory process: {top_memory_process['name']} ({top_memory_process['memory_percent']:.1f}%)")
        recommendations.append(f"Monitor {top_memory_process['name']} memory usage")
    
    return {
        'severity': severity,
        'issues': issues,
        'recommendations': recommendations
    }

def apply_optimizations(info, analysis):
    """Apply basic system optimizations."""
    optimizations_applied = []
    
    try:
        # Force garbage collection (Python-specific)
        import gc
        collected = gc.collect()
        if collected > 0:
            optimizations_applied.append(f"Freed {collected} Python objects")
        
        # Memory pressure relief
        if info['memory']['percent'] > MAX_MEMORY_PERCENT:
            # Try to reduce memory pressure
            try:
                # Clear system caches (requires sudo, so we'll skip for now)
                # subprocess.run(['sudo', 'purge'], check=True)
                # optimizations_applied.append("Cleared system caches")
                pass
            except:
                pass
        
        # CPU optimization suggestions
        if info['cpu']['percent'] > MAX_CPU_PERCENT:
            optimizations_applied.append("Recommended CPU optimization (see logs)")
        
    except Exception as e:
        logging.warning(f"Error applying optimizations: {e}")
    
    return optimizations_applied

def generate_report(info, analysis, optimizations):
    """Generate a comprehensive report."""
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'monitor_version': '1.0',
            'system': 'Mac M2 Max'
        },
        'system_info': info,
        'health_analysis': analysis,
        'optimizations_applied': optimizations,
        'summary': {
            'memory_status': 'critical' if info['memory']['percent'] >= 90 
                           else 'warning' if info['memory']['percent'] >= MAX_MEMORY_PERCENT 
                           else 'good',
            'cpu_status': 'critical' if info['cpu']['percent'] >= 95 
                         else 'warning' if info['cpu']['percent'] >= MAX_CPU_PERCENT 
                         else 'good',
            'overall_health': analysis['severity']
        }
    }
    
    # Save report
    report_file = REPORT_DIR / f"resource_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_file

def send_notification(title, message):
    """Send macOS notification."""
    try:
        subprocess.run([
            'osascript', '-e',
            f'display notification "{message}" with title "{title}"'
        ], check=True)
    except:
        pass  # Fail silently if notifications don't work

def perform_resource_check():
    """Perform a complete resource check."""
    logger = setup_logging()
    logger.info("🔍 Starting resource check")
    
    try:
        # Get system information
        info = get_system_info()
        logger.info(f"Memory: {info['memory']['percent']:.1f}%, CPU: {info['cpu']['percent']:.1f}%")
        
        # Analyze system health
        analysis = analyze_system_health(info)
        
        # Log issues
        if analysis['issues']:
            for issue in analysis['issues']:
                if analysis['severity'] == 'critical':
                    logger.critical(issue)
                elif analysis['severity'] == 'warning':
                    logger.warning(issue)
                else:
                    logger.info(issue)
        
        # Apply optimizations
        optimizations = apply_optimizations(info, analysis)
        if optimizations:
            logger.info(f"Applied optimizations: {', '.join(optimizations)}")
        
        # Generate report
        report_file = generate_report(info, analysis, optimizations)
        logger.info(f"Report generated: {report_file}")
        
        # Send notifications for critical issues
        if analysis['severity'] == 'critical':
            send_notification("Resource Alert", f"Critical system resource usage detected")
        elif analysis['severity'] == 'warning' and analysis['issues']:
            send_notification("Resource Warning", f"{len(analysis['issues'])} resource warnings")
        
        # Print summary
        print(f"✅ Resource check completed")
        print(f"   Memory: {info['memory']['percent']:.1f}% ({info['memory']['used_gb']:.1f}GB)")
        print(f"   CPU: {info['cpu']['percent']:.1f}%")
        print(f"   Status: {analysis['severity']}")
        print(f"   Issues: {len(analysis['issues'])}")
        print(f"   Optimizations: {len(optimizations)}")
        print(f"   Report: {report_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during resource check: {e}")
        return False

def create_launchd_plist():
    """Create macOS LaunchAgent plist for automatic execution."""
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quark.simple.resource.monitor</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/python3</string>
        <string>{Path(__file__).absolute()}</string>
        <string>--run-check</string>
    </array>
    
    <key>StartInterval</key>
    <integer>{int(MONITOR_INTERVAL_HOURS * 3600)}</integer>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>{LOG_DIR}/launchd_stdout.log</string>
    
    <key>StandardErrorPath</key>
    <string>{LOG_DIR}/launchd_stderr.log</string>
    
    <key>WorkingDirectory</key>
    <string>{Path.home()}</string>
</dict>
</plist>"""
    
    plist_file = Path.home() / "Library" / "LaunchAgents" / "com.quark.simple.resource.monitor.plist"
    plist_file.parent.mkdir(exist_ok=True)
    
    with open(plist_file, 'w') as f:
        f.write(plist_content)
    
    return plist_file

def install_service():
    """Install the periodic monitoring service."""
    try:
        print("🔧 Installing simple resource monitor...")
        
        # Create plist
        plist_file = create_launchd_plist()
        print(f"✅ Created plist: {plist_file}")
        
        # Load service
        result = subprocess.run(['launchctl', 'load', str(plist_file)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Service installed successfully!")
            print(f"🛡️ Resource monitoring will run every {MONITOR_INTERVAL_HOURS} hours")
            print(f"📊 Reports saved to: {REPORT_DIR}")
            print(f"📝 Logs saved to: {LOG_DIR}")
            print()
            print("Management commands:")
            print(f"  Status: launchctl list | grep com.quark.simple.resource.monitor")
            print(f"  Stop:   launchctl unload {plist_file}")
            print(f"  Manual: python3 {__file__} --run-check")
            return True
        else:
            print(f"❌ Failed to load service: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def uninstall_service():
    """Uninstall the monitoring service."""
    try:
        plist_file = Path.home() / "Library" / "LaunchAgents" / "com.quark.simple.resource.monitor.plist"
        
        if plist_file.exists():
            subprocess.run(['launchctl', 'unload', str(plist_file)], 
                         capture_output=True, text=True)
            plist_file.unlink()
            print("✅ Service uninstalled successfully")
            return True
        else:
            print("⚠️ Service not found")
            return False
            
    except Exception as e:
        print(f"❌ Uninstall failed: {e}")
        return False

def show_status():
    """Show service status."""
    try:
        result = subprocess.run(['launchctl', 'list', 'com.quark.simple.resource.monitor'],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Service is running")
            print(result.stdout)
        else:
            print("❌ Service is not running")
        
        # Show recent reports
        reports = sorted(REPORT_DIR.glob("resource_report_*.json"))
        if reports:
            latest_report = reports[-1]
            with open(latest_report) as f:
                data = json.load(f)
            
            print(f"\n📊 Latest Report ({latest_report.name}):")
            print(f"   Memory: {data['system_info']['memory']['percent']:.1f}%")
            print(f"   CPU: {data['system_info']['cpu']['percent']:.1f}%")
            print(f"   Status: {data['summary']['overall_health']}")
            print(f"   Issues: {len(data['health_analysis']['issues'])}")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Resource Monitor for Mac M2 Max")
    parser.add_argument('--run-check', action='store_true', help='Run single resource check')
    parser.add_argument('--install', action='store_true', help='Install monitoring service')
    parser.add_argument('--uninstall', action='store_true', help='Uninstall monitoring service') 
    parser.add_argument('--status', action='store_true', help='Show service status')
    
    args = parser.parse_args()
    
    if args.install:
        install_service()
    elif args.uninstall:
        uninstall_service()
    elif args.status:
        show_status()
    elif args.run_check:
        perform_resource_check()
    else:
        # Interactive mode
        print("🛡️ Simple Resource Monitor for Mac M2 Max")
        print("=" * 50)
        print("1. Run single resource check")
        print("2. Install hourly monitoring service")
        print("3. Show service status")
        print("4. Uninstall service")
        print()
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            perform_resource_check()
        elif choice == '2':
            install_service()
        elif choice == '3':
            show_status()
        elif choice == '4':
            uninstall_service()
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()

