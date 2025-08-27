#!/usr/bin/env python3
"""
System Performance Optimizer for Quark Development Environment
Optimizes macOS settings, Python environment, and development tools
"""

import os
import subprocess
import sys
import psutil
import shutil
from pathlib import Path
import json
from typing import Dict, List, Any
import logging

class SystemOptimizer:
    """Optimizes system performance for development"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quark_root = Path.cwd()
        self.optimizations_applied = []
        
    def check_system_resources(self) -> Dict[str, Any]:
        """Check current system resource usage"""
        
        # Get CPU info
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Get memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Get disk info
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        # Check processes
        cursor_processes = []
        python_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                if 'cursor' in proc.info['name'].lower():
                    cursor_processes.append(proc.info)
                elif 'python' in proc.info['name'].lower():
                    python_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {
            "cpu": {
                "cores": cpu_count,
                "usage_percent": cpu_usage
            },
            "memory": {
                "total_gb": round(memory_gb, 1),
                "available_gb": round(memory_available_gb, 1),
                "usage_percent": round((memory_gb - memory_available_gb) / memory_gb * 100, 1)
            },
            "disk": {
                "total_gb": round(disk_total_gb, 1),
                "free_gb": round(disk_free_gb, 1),
                "usage_percent": round((disk_total_gb - disk_free_gb) / disk_total_gb * 100, 1)
            },
            "processes": {
                "cursor_count": len(cursor_processes),
                "python_count": len(python_processes),
                "cursor_processes": cursor_processes[:3],  # Top 3
                "python_processes": python_processes[:3]   # Top 3
            }
        }
    
    def optimize_spotlight_indexing(self) -> bool:
        """Disable Spotlight indexing for large directories"""
        
        large_dirs = [
            self.quark_root / "datasets",
            self.quark_root / "venv", 
            self.quark_root / "fresh_venv",
            self.quark_root / "external"
        ]
        
        optimized_count = 0
        
        for directory in large_dirs:
            if directory.exists():
                try:
                    # Disable Spotlight indexing
                    result = subprocess.run([
                        'sudo', 'mdutil', '-i', 'off', str(directory)
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        self.logger.info(f"Disabled Spotlight indexing for {directory}")
                        optimized_count += 1
                    else:
                        self.logger.warning(f"Failed to disable Spotlight for {directory}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Timeout disabling Spotlight for {directory}")
                except Exception as e:
                    self.logger.error(f"Error optimizing Spotlight for {directory}: {e}")
        
        if optimized_count > 0:
            self.optimizations_applied.append(f"Disabled Spotlight indexing for {optimized_count} directories")
            return True
        
        return False
    
    def optimize_cursor_cache(self) -> bool:
        """Clean and optimize Cursor cache"""
        
        cursor_dirs = [
            Path.home() / "Library" / "Application Support" / "Cursor" / "CachedData",
            Path.home() / "Library" / "Application Support" / "Cursor" / "logs",
            Path.home() / "Library" / "Caches" / "com.todesktop.230313mzl4w4u92" / "CachedData"
        ]
        
        cleaned_size = 0
        
        for cache_dir in cursor_dirs:
            if cache_dir.exists():
                try:
                    # Get size before cleaning
                    size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                    
                    # Remove cache contents
                    for item in cache_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    
                    cleaned_size += size_before
                    self.logger.info(f"Cleaned Cursor cache: {cache_dir}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not clean {cache_dir}: {e}")
        
        if cleaned_size > 0:
            cleaned_mb = cleaned_size / (1024**2)
            self.optimizations_applied.append(f"Cleaned {cleaned_mb:.1f}MB of Cursor cache")
            return True
        
        return False
    
    def optimize_python_environment(self) -> bool:
        """Optimize Python environment for performance"""
        
        optimizations = []
        
        # Check if we're in a virtual environment
        venv_path = self.quark_root / "venv"
        if venv_path.exists():
            
            # Create or update pip.conf for faster downloads
            pip_conf_dir = venv_path / "pip.conf"
            pip_conf_content = """[global]
timeout = 60
retries = 5
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org

[install]
upgrade-strategy = only-if-needed
"""
            
            try:
                with open(pip_conf_dir, 'w') as f:
                    f.write(pip_conf_content)
                optimizations.append("Optimized pip configuration")
            except Exception as e:
                self.logger.warning(f"Could not create pip.conf: {e}")
            
            # Set Python optimization environment variables
            env_vars = {
                'PYTHONDONTWRITEBYTECODE': '1',  # Don't create .pyc files
                'PYTHONUNBUFFERED': '1',        # Unbuffered output
                'PYTHONUTF8': '1',              # UTF-8 mode
            }
            
            # Update shell configuration
            shell_config = Path.home() / ".zshrc"
            if shell_config.exists():
                try:
                    with open(shell_config, 'r') as f:
                        content = f.read()
                    
                    new_exports = []
                    for var, value in env_vars.items():
                        if f"export {var}" not in content:
                            new_exports.append(f"export {var}={value}")
                    
                    if new_exports:
                        with open(shell_config, 'a') as f:
                            f.write(f"\n# Quark Python optimizations\n")
                            f.write("\n".join(new_exports) + "\n")
                        
                        optimizations.append("Added Python environment optimizations to .zshrc")
                        
                except Exception as e:
                    self.logger.warning(f"Could not update .zshrc: {e}")
        
        if optimizations:
            self.optimizations_applied.extend(optimizations)
            return True
        
        return False
    
    def optimize_file_limits(self) -> bool:
        """Increase file descriptor limits for large projects"""
        
        try:
            # Check current limits
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            
            if soft < 32768:
                # Try to increase soft limit
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (min(32768, hard), hard))
                    self.optimizations_applied.append(f"Increased file descriptor limit from {soft} to {min(32768, hard)}")
                    return True
                except Exception as e:
                    self.logger.warning(f"Could not increase file descriptor limit: {e}")
            
        except Exception as e:
            self.logger.warning(f"Could not check file limits: {e}")
        
        return False
    
    def create_performance_aliases(self) -> bool:
        """Create useful performance monitoring aliases"""
        
        aliases = """
# Quark Performance Monitoring Aliases
alias quark-status='python optimization/system_optimizer.py status'
alias quark-clean='python optimization/system_optimizer.py clean'
alias quark-monitor='top -pid $(pgrep -f "cursor\\|python")'
alias quark-memory='ps aux | grep -E "(cursor|python)" | head -10'
alias quark-disk='du -sh * | sort -hr | head -10'
alias quark-network='netstat -an | grep ESTABLISHED | wc -l'
"""
        
        shell_config = Path.home() / ".zshrc"
        if shell_config.exists():
            try:
                with open(shell_config, 'r') as f:
                    content = f.read()
                
                if "# Quark Performance Monitoring Aliases" not in content:
                    with open(shell_config, 'a') as f:
                        f.write(aliases)
                    
                    self.optimizations_applied.append("Added performance monitoring aliases")
                    return True
                    
            except Exception as e:
                self.logger.warning(f"Could not update .zshrc with aliases: {e}")
        
        return False
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        system_info = self.check_system_resources()
        
        # Performance recommendations
        recommendations = []
        
        if system_info["memory"]["usage_percent"] > 80:
            recommendations.append("⚠️ High memory usage - consider closing unused applications")
        
        if system_info["cpu"]["usage_percent"] > 70:
            recommendations.append("⚠️ High CPU usage - check for runaway processes")
        
        if system_info["disk"]["usage_percent"] > 85:
            recommendations.append("⚠️ Low disk space - consider cleaning up large files")
        
        if system_info["processes"]["cursor_count"] > 3:
            recommendations.append("💡 Multiple Cursor processes detected - consider restarting Cursor")
        
        if len(self.optimizations_applied) == 0:
            recommendations.append("✅ All optimizations already applied")
        
        return {
            "timestamp": "2025-08-28T05:15:00",
            "system_resources": system_info,
            "optimizations_applied": self.optimizations_applied,
            "recommendations": recommendations,
            "performance_score": self._calculate_performance_score(system_info)
        }
    
    def _calculate_performance_score(self, system_info: Dict[str, Any]) -> int:
        """Calculate overall performance score (0-100)"""
        
        score = 100
        
        # Deduct for high resource usage
        if system_info["memory"]["usage_percent"] > 80:
            score -= 20
        elif system_info["memory"]["usage_percent"] > 60:
            score -= 10
        
        if system_info["cpu"]["usage_percent"] > 70:
            score -= 15
        elif system_info["cpu"]["usage_percent"] > 50:
            score -= 8
        
        if system_info["disk"]["usage_percent"] > 90:
            score -= 25
        elif system_info["disk"]["usage_percent"] > 80:
            score -= 10
        
        # Bonus for optimizations applied
        score += min(len(self.optimizations_applied) * 2, 10)
        
        return max(0, min(100, score))
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """Run all optimization procedures"""
        
        print("🚀 Running System Optimization for Quark Development")
        print("=" * 55)
        
        optimizations = [
            ("File descriptor limits", self.optimize_file_limits),
            ("Python environment", self.optimize_python_environment),
            ("Cursor cache cleanup", self.optimize_cursor_cache),
            ("Performance aliases", self.create_performance_aliases),
            ("Spotlight indexing", self.optimize_spotlight_indexing)
        ]
        
        results = {}
        
        for name, optimization_func in optimizations:
            try:
                print(f"🔧 Optimizing {name}...")
                success = optimization_func()
                results[name] = "success" if success else "skipped"
                print(f"   {'✅' if success else '⏭️'} {name}")
            except Exception as e:
                results[name] = f"error: {e}"
                print(f"   ❌ {name}: {e}")
        
        # Generate final report
        report = self.generate_optimization_report()
        
        print(f"\n📊 Optimization Complete!")
        print(f"   Performance Score: {report['performance_score']}/100")
        print(f"   Memory Usage: {report['system_resources']['memory']['usage_percent']}%")
        print(f"   CPU Usage: {report['system_resources']['cpu']['usage_percent']}%")
        print(f"   Optimizations Applied: {len(self.optimizations_applied)}")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   {rec}")
        
        return report

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            optimizer = SystemOptimizer()
            report = optimizer.generate_optimization_report()
            
            print("📊 Quark System Status")
            print("=" * 25)
            print(f"Performance Score: {report['performance_score']}/100")
            print(f"Memory: {report['system_resources']['memory']['usage_percent']}%")
            print(f"CPU: {report['system_resources']['cpu']['usage_percent']}%") 
            print(f"Disk: {report['system_resources']['disk']['usage_percent']}%")
            
        elif command == "clean":
            optimizer = SystemOptimizer()
            optimizer.optimize_cursor_cache()
            print("🧹 Cache cleanup complete")
            
        else:
            print("Usage: python system_optimizer.py [status|clean]")
    else:
        # Run full optimization
        optimizer = SystemOptimizer()
        optimizer.run_full_optimization()

if __name__ == "__main__":
    main()
