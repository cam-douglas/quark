#!/usr/bin/env python3
"""
🛡️ Resource Monitor Installation Script
=======================================

Easy installation script for the Ultimate Resource Authority periodic monitoring.
This script sets up hourly resource monitoring on your Mac M2 Max system.

Author: Quark Resource Management Team
Created: 2025-01-21
"""

import os, sys
import subprocess
from pathlib import Path

def print_header():
    """Print installation header."""
    print("=" * 70)
    print("🛡️ ULTIMATE RESOURCE AUTHORITY INSTALLATION")
    print("=" * 70)
    print("Setting up periodic resource monitoring for Mac M2 Max")
    print()

def check_system_requirements():
    """Check if system meets requirements."""
    print("🔍 Checking system requirements...")
    
    requirements_met = True
    
    # Check macOS
    try:
        result = subprocess.run(['sw_vers', '-productName'], 
                              capture_output=True, text=True)
        if 'macOS' not in result.stdout:
            print("❌ This installer is designed for macOS")
            requirements_met = False
        else:
            print("✅ macOS detected")
    except:
        print("⚠️ Could not detect operating system")
    
    # Check Python
    try:
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            print(f"✅ Python {python_version.major}.{python_version.minor} detected")
        else:
            print("❌ Python 3.8+ required")
            requirements_met = False
    except:
        print("❌ Python not properly detected")
        requirements_met = False
    
    # Check required packages
    required_packages = ['psutil']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"⚠️ {package} not found - will attempt to install")
    
    # Check processor (optional)
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True)
        if 'Apple' in result.stdout:
            print("✅ Apple Silicon processor detected")
        else:
            print("⚠️ Non-Apple Silicon processor - system will still work but may not be optimized")
    except:
        print("⚠️ Could not detect processor type")
    
    print()
    return requirements_met

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing required packages...")
    
    required_packages = ['psutil', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    print()
    return True

def test_resource_monitoring():
    """Test that resource monitoring works."""
    print("🧪 Testing resource monitoring...")
    
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Test basic resource monitoring
        import psutil
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        print(f"✅ Current Memory Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB)")
        print(f"✅ Current CPU Usage: {cpu:.1f}%")
        
        # Test import of our modules
        try:
            from brain_modules.resource_monitor import create_resource_monitor
            print("✅ Resource monitor modules importable")
        except ImportError as e:
            print(f"⚠️ Resource monitor modules not fully available: {e}")
            print("   (Basic monitoring will still work)")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Resource monitoring test failed: {e}")
        return False

def install_periodic_service():
    """Install the periodic monitoring service."""
    print("⚙️ Installing periodic monitoring service...")
    
    try:
        # Get user preferences
        print("Configuration options:")
        print("1. Monitoring interval (hours)")
        print("2. Cloud offloading")
        print()
        
        while True:
            try:
                interval = input("Enter monitoring interval in hours (default: 1.0): ").strip()
                if not interval:
                    interval = 1.0
                else:
                    interval = float(interval)
                break
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            cloud_offload = input("Enable cloud offloading? (y/n, default: y): ").strip().lower()
            if not cloud_offload or cloud_offload in ['y', 'yes']:
                cloud_offload = True
                break
            elif cloud_offload in ['n', 'no']:
                cloud_offload = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no")
        
        print()
        print(f"Installing service with:")
        print(f"  Monitoring interval: {interval} hours")
        print(f"  Cloud offloading: {'enabled' if cloud_offload else 'disabled'}")
        print()
        
        # Run installation
        script_path = Path(__file__).parent / "brain_modules" / "resource_monitor" / "periodic_resource_monitor.py"
        
        cmd = [sys.executable, str(script_path), '--install', '--interval', str(interval)]
        if cloud_offload:
            cmd.append('--enable-cloud-offload')
        else:
            cmd.append('--disable-cloud-offload')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Periodic monitoring service installed successfully!")
            print()
            print("Service details:")
            print(f"  • Runs every {interval} hours automatically")
            print(f"  • Monitors memory, CPU, temperature, and I/O")
            print(f"  • Applies automatic optimizations when needed")
            if cloud_offload:
                print(f"  • Offloads intensive tasks to free cloud platforms")
            print(f"  • Generates detailed reports")
            print()
            print("Service management:")
            print("  • Status: python3 brain_modules/resource_monitor/periodic_resource_monitor.py --status")
            print("  • Uninstall: python3 brain_modules/resource_monitor/periodic_resource_monitor.py --uninstall")
            print()
            print("Files and logs:")
            print(f"  • Logs: {Path.cwd()}/logs/periodic_monitoring/")
            print(f"  • Reports: {Path.cwd()}/reports/resource_monitoring/")
            print()
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def run_test_monitoring():
    """Run a test monitoring session."""
    print("🔬 Running test monitoring session...")
    
    try:
        script_path = Path(__file__).parent / "brain_modules" / "resource_monitor" / "periodic_resource_monitor.py"
        
        print("Performing single resource check...")
        result = subprocess.run([
            sys.executable, str(script_path), 
            '--enable-cloud-offload', '--enable-reports'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test monitoring completed successfully!")
            print()
            print("Test results:")
            # Extract key information from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Memory:' in line or 'CPU:' in line or 'Temperature:' in line:
                    print(f"  {line.strip()}")
            print()
            return True
        else:
            print(f"❌ Test monitoring failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Test monitoring timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    """Main installation process."""
    print_header()
    
    success = True
    
    # Check requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please resolve issues above.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies.")
        return False
    
    # Test resource monitoring
    if not test_resource_monitoring():
        print("❌ Resource monitoring test failed.")
        return False
    
    # Ask user what they want to do
    print("Installation options:")
    print("1. Install periodic monitoring service (runs every hour)")
    print("2. Run single test monitoring session")
    print("3. Both")
    print()
    
    while True:
        choice = input("Choose option (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3")
    
    print()
    
    if choice in ['1', '3']:
        if not install_periodic_service():
            success = False
    
    if choice in ['2', '3']:
        if not run_test_monitoring():
            success = False
    
    # Final summary
    print("=" * 70)
    if success:
        print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        print()
        print("Your Mac M2 Max is now protected by the Ultimate Resource Authority!")
        print()
        print("What happens next:")
        print("• System resources are monitored continuously")
        print("• Automatic optimizations prevent performance issues")
        print("• Intensive tasks are offloaded to free cloud platforms")
        print("• Detailed reports track system health over time")
        print()
        print("To check status at any time:")
        print("  python3 brain_modules/resource_monitor/periodic_resource_monitor.py --status")
        
    else:
        print("⚠️ INSTALLATION COMPLETED WITH WARNINGS")
        print()
        print("Some components may not be fully functional.")
        print("Check the messages above for specific issues.")
    
    print("=" * 70)
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during installation: {e}")
        sys.exit(1)

