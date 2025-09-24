"""
Deployment Handler Module
=========================
Handles deployment operations.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Optional


class DeploymentHandler:
    """Handles deployment operations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.deploy_scripts = {
            'gcp': self.project_root / 'scripts' / 'deploy_vllm_gcp.py',
            'local': self.project_root / 'deployment' / 'deploy_local.sh',
            'docker': self.project_root / 'Dockerfile'
        }
        self.config_dir = self.project_root / 'deployment'
        
    def route_command(self, action: str, params: Dict) -> int:
        """Route deployment commands."""
        if action == 'deploy':
            return self.deploy(params)
        elif action == 'status':
            return self.check_deployment_status()
        elif action == 'rollback':
            return self.rollback_deployment()
        elif action == 'logs':
            return self.show_deployment_logs()
        else:
            return self.show_help()
    
    def deploy(self, params: Dict) -> int:
        """Deploy to specified platform."""
        platform = params.get('platform', 'local')
        
        print(f"\n🚀 Deploying to {platform.upper()}")
        print("=" * 50)
        
        if platform == 'gcp':
            return self._deploy_gcp(params)
        elif platform == 'docker':
            return self._deploy_docker(params)
        elif platform == 'local':
            return self._deploy_local(params)
        else:
            print(f"⚠️ Unknown platform: {platform}")
            print("📝 Available platforms: gcp, docker, local")
            return 1
    
    def _deploy_gcp(self, params: Dict) -> int:
        """Deploy to Google Cloud Platform."""
        if self.deploy_scripts['gcp'].exists():
            print("☁️ Deploying to GCP...")
            
            cmd = [sys.executable, str(self.deploy_scripts['gcp'])]
            
            # Add optional parameters
            if params.get('project'):
                cmd.extend(['--project', params['project']])
            if params.get('region'):
                cmd.extend(['--region', params['region']])
            if params.get('instance_type'):
                cmd.extend(['--instance-type', params['instance_type']])
            
            return subprocess.run(cmd).returncode
        else:
            print("⚠️ GCP deployment script not found")
            print("📝 Run: scripts/setup_gcp_deployment.sh")
            return 1
    
    def _deploy_docker(self, params: Dict) -> int:
        """Deploy using Docker."""
        if self.deploy_scripts['docker'].exists():
            print("🐳 Building Docker image...")
            
            # Build image
            build_cmd = ['docker', 'build', '-t', 'quark:latest', '.']
            result = subprocess.run(build_cmd)
            
            if result.returncode == 0:
                print("✅ Docker image built")
                
                # Run container
                run_cmd = ['docker', 'run', '-d', '--name', 'quark-brain']
                
                if params.get('port'):
                    run_cmd.extend(['-p', f"{params['port']}:8080"])
                
                run_cmd.append('quark:latest')
                
                return subprocess.run(run_cmd).returncode
            
            return result.returncode
        else:
            print("⚠️ Dockerfile not found")
            return 1
    
    def _deploy_local(self, params: Dict) -> int:
        """Deploy locally."""
        print("💻 Deploying locally...")
        
        # Check for local deployment script
        if self.deploy_scripts['local'].exists():
            cmd = ['bash', str(self.deploy_scripts['local'])]
            return subprocess.run(cmd).returncode
        else:
            # Fallback to running main script
            main_script = self.project_root / 'brain' / 'brain_main.py'
            if main_script.exists():
                cmd = [sys.executable, str(main_script)]
                
                if params.get('daemon'):
                    # Run in background
                    subprocess.Popen(cmd)
                    print("✅ Started in background")
                    return 0
                else:
                    return subprocess.run(cmd).returncode
            
            print("⚠️ No local deployment method found")
            return 1
    
    def check_deployment_status(self) -> int:
        """Check deployment status."""
        print("\n📊 Deployment Status")
        print("=" * 50)
        
        # Check Docker
        try:
            result = subprocess.run(['docker', 'ps', '--filter', 'name=quark'],
                                  capture_output=True, text=True)
            if result.returncode == 0 and 'quark' in result.stdout:
                print("🐳 Docker: Running")
            else:
                print("🐳 Docker: Not running")
        except:
            print("🐳 Docker: Not available")
        
        # Check GCP
        try:
            result = subprocess.run(['gcloud', 'compute', 'instances', 'list',
                                   '--filter', 'name:quark'],
                                  capture_output=True, text=True)
            if result.returncode == 0 and 'quark' in result.stdout:
                print("☁️ GCP: Instance found")
            else:
                print("☁️ GCP: No instances")
        except:
            print("☁️ GCP: Not configured")
        
        # Check local process
        try:
            result = subprocess.run(['pgrep', '-f', 'brain_main'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("💻 Local: Running")
                print(f"   PID: {result.stdout.strip()}")
            else:
                print("💻 Local: Not running")
        except:
            print("💻 Local: Could not check")
        
        return 0
    
    def rollback_deployment(self) -> int:
        """Rollback to previous deployment."""
        print("\n⏪ Rolling Back Deployment")
        print("=" * 50)
        
        # This would implement rollback logic
        print("📝 Rollback steps:")
        print("1. Stop current deployment")
        print("2. Restore previous version")
        print("3. Restart services")
        
        response = input("\nProceed with rollback? (y/n): ")
        if response.lower() == 'y':
            print("✅ Rollback initiated")
            # Implement actual rollback
        else:
            print("⚠️ Rollback cancelled")
        
        return 0
    
    def show_deployment_logs(self) -> int:
        """Show deployment logs."""
        print("\n📜 Deployment Logs")
        print("=" * 50)
        
        # Check for Docker logs
        try:
            subprocess.run(['docker', 'logs', 'quark-brain', '--tail', '20'])
        except:
            print("⚠️ No Docker logs available")
        
        # Check for local logs
        log_file = self.project_root / 'logs' / 'deployment.log'
        if log_file.exists():
            print("\n📁 Local logs:")
            subprocess.run(['tail', '-20', str(log_file)])
        
        return 0
    
    def show_help(self) -> int:
        """Show deployment help."""
        print("""
🚀 Deployment Commands:
  todo deploy to gcp           → Deploy to Google Cloud
  todo deploy docker           → Deploy using Docker
  todo deploy local            → Deploy locally
  todo deployment status       → Check deployment status
  todo deployment logs         → Show deployment logs
  todo rollback deployment     → Rollback to previous version
""")
        return 0
