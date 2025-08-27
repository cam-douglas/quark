#!/usr/bin/env python3
"""
Cursor Network Fix Script for Docker Environments
Purpose: Diagnose and resolve Cursor Chat/Agent network issues in Docker containers
Author: Quark Development Team
Dependencies: requests, json, subprocess
"""

import os
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class CursorNetworkFixer:
    """
    Comprehensive solution for Cursor Docker network issues.
    Handles streaming problems, DNS resolution, and container configuration.
    """
    
    def __init__(self):
        self.cursor_config_paths = [
            Path.home() / ".cursor" / "settings.json",
            Path.home() / ".config" / "cursor" / "settings.json",
            Path("/opt/cursor/settings.json"),  # Common Docker location
        ]
        self.backup_dir = Path.home() / ".cursor_network_fix_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def diagnose_environment(self) -> Dict[str, any]:
        """Diagnose current environment and network configuration."""
        diagnosis = {
            "docker_detected": self._is_running_in_docker(),
            "cursor_config_found": None,
            "current_settings": None,
            "network_connectivity": {},
            "dns_resolution": {},
            "environment_vars": {}
        }
        
        # Find Cursor config
        for config_path in self.cursor_config_paths:
            if config_path.exists():
                diagnosis["cursor_config_found"] = str(config_path)
                try:
                    with open(config_path, 'r') as f:
                        diagnosis["current_settings"] = json.load(f)
                except Exception as e:
                    diagnosis["config_read_error"] = str(e)
                break
        
        # Test network connectivity
        cursor_endpoints = [
            "api2.cursor.sh",
            "api3.cursor.sh",
            "cursor.sh"
        ]
        
        for endpoint in cursor_endpoints:
            diagnosis["network_connectivity"][endpoint] = self._test_connectivity(endpoint)
            diagnosis["dns_resolution"][endpoint] = self._test_dns_resolution(endpoint)
        
        # Check environment variables
        cursor_env_vars = [
            "CURSOR_DISABLE_STREAMING",
            "CURSOR_DOCKER_MODE", 
            "CURSOR_USE_POLLING",
            "HTTP_PROXY",
            "HTTPS_PROXY"
        ]
        
        for var in cursor_env_vars:
            diagnosis["environment_vars"][var] = os.environ.get(var)
            
        return diagnosis
    
    def _is_running_in_docker(self) -> bool:
        """Check if running inside Docker container."""
        try:
            # Check for .dockerenv file
            if Path("/.dockerenv").exists():
                return True
            
            # Check cgroup for docker
            with open("/proc/1/cgroup", "r") as f:
                return "docker" in f.read()
        except:
            return False
    
    def _test_connectivity(self, hostname: str, port: int = 443) -> Dict[str, any]:
        """Test network connectivity to hostname."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((hostname, port))
            sock.close()
            return {
                "success": result == 0,
                "port": port,
                "error": None if result == 0 else f"Connection failed: {result}"
            }
        except Exception as e:
            return {
                "success": False,
                "port": port,
                "error": str(e)
            }
    
    def _test_dns_resolution(self, hostname: str) -> Dict[str, any]:
        """Test DNS resolution for hostname."""
        try:
            import socket
            ip = socket.gethostbyname(hostname)
            return {
                "success": True,
                "ip": ip,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "ip": None,
                "error": str(e)
            }
    
    def backup_current_config(self) -> Optional[Path]:
        """Backup current Cursor configuration."""
        for config_path in self.cursor_config_paths:
            if config_path.exists():
                timestamp = int(time.time())
                backup_path = self.backup_dir / f"cursor_settings_backup_{timestamp}.json"
                
                try:
                    with open(config_path, 'r') as src:
                        with open(backup_path, 'w') as dst:
                            dst.write(src.read())
                    
                    print(f"✅ Backup created: {backup_path}")
                    return backup_path
                except Exception as e:
                    print(f"❌ Backup failed: {e}")
                    return None
        
        print("⚠️  No Cursor config found to backup")
        return None
    
    def apply_docker_network_fix(self) -> bool:
        """Apply network fixes for Docker environment."""
        # Find config file
        config_path = None
        for path in self.cursor_config_paths:
            if path.exists():
                config_path = path
                break
        
        if not config_path:
            # Create default config location
            config_path = self.cursor_config_paths[0]
            config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create new
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
        except:
            config = {}
        
        # Apply Docker-specific network settings
        docker_settings = {
            "cursor.network.dockerMode": True,
            "cursor.network.disableStreaming": True,
            "cursor.chat.usePolling": True,
            "cursor.agent.usePolling": True,
            "cursor.network.streamingFallback": "polling",
            "cursor.network.timeout": 30000,
            "cursor.network.retryAttempts": 3,
            "cursor.network.pollInterval": 1000,
            "http.proxySupport": "off"
        }
        
        # Update config
        config.update(docker_settings)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Applied Docker network fix to: {config_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to update config: {e}")
            return False
    
    def set_environment_variables(self) -> None:
        """Set required environment variables."""
        env_vars = {
            "CURSOR_DISABLE_STREAMING": "1",
            "CURSOR_DOCKER_MODE": "1",
            "CURSOR_USE_POLLING": "1",
            "CURSOR_NETWORK_TIMEOUT": "30000"
        }
        
        print("Setting environment variables...")
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"  {var}={value}")
        
        # Also output shell commands for manual execution
        print("\n📋 Run these commands in your shell:")
        for var, value in env_vars.items():
            print(f"export {var}={value}")
    
    def generate_docker_run_commands(self) -> List[str]:
        """Generate Docker run commands with network fixes."""
        commands = [
            # Host networking (best solution)
            "docker run --network host your-cursor-container",
            
            # DNS and host resolution fixes
            "docker run --add-host=api2.cursor.sh:44.196.185.7 \\\n"
            "           --add-host=api3.cursor.sh:104.18.19.125 \\\n"
            "           --dns=8.8.8.8 \\\n"
            "           --dns=8.8.4.4 \\\n"
            "           your-container",
            
            # Environment variable approach
            "docker run -e CURSOR_DISABLE_STREAMING=1 \\\n"
            "           -e CURSOR_DOCKER_MODE=1 \\\n"
            "           -e CURSOR_USE_POLLING=1 \\\n"
            "           your-container"
        ]
        
        return commands
    
    def run_comprehensive_fix(self) -> None:
        """Run complete diagnostic and fix process."""
        print("🔍 CURSOR NETWORK DIAGNOSTIC & FIX")
        print("=" * 50)
        
        # Diagnose
        print("\n1. 🔍 Diagnosing environment...")
        diagnosis = self.diagnose_environment()
        
        print(f"   Docker detected: {diagnosis['docker_detected']}")
        print(f"   Config found: {diagnosis['cursor_config_found']}")
        
        # Show connectivity issues
        connectivity_issues = []
        for endpoint, result in diagnosis["network_connectivity"].items():
            if not result["success"]:
                connectivity_issues.append(f"{endpoint}: {result['error']}")
        
        if connectivity_issues:
            print("   ❌ Connectivity issues:")
            for issue in connectivity_issues:
                print(f"      {issue}")
        else:
            print("   ✅ Network connectivity OK")
        
        # Backup config
        print("\n2. 💾 Creating backup...")
        self.backup_current_config()
        
        # Apply fixes
        print("\n3. 🔧 Applying network fixes...")
        self.apply_docker_network_fix()
        
        # Set environment variables
        print("\n4. 🌍 Setting environment variables...")
        self.set_environment_variables()
        
        # Generate Docker commands
        print("\n5. 🐳 Docker run command suggestions:")
        commands = self.generate_docker_run_commands()
        for i, cmd in enumerate(commands, 1):
            print(f"\n   Option {i}:")
            print(f"   {cmd}")
        
        print("\n" + "=" * 50)
        print("✅ FIXES APPLIED")
        print("📋 Next Steps:")
        print("   1. Restart Cursor completely")
        print("   2. If still having issues, try Docker commands above")
        print("   3. Check cursor_network_test.py for verification")
        print("   4. Restore from backup if needed")

def main():
    """Main execution function."""
    fixer = CursorNetworkFixer()
    fixer.run_comprehensive_fix()

if __name__ == "__main__":
    main()


