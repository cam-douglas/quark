#!/usr/bin/env python3
"""
Google Cloud API Manager for Quark Brain Architecture.

This module handles programmatic enabling of Google Cloud APIs
and manages authentication state.

Created: 2025-01-20
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleCloudAPIManager:
    """Manages Google Cloud APIs and authentication state."""
    
    # Common APIs for Quark brain architecture
    QUARK_REQUIRED_APIS = {
        'vision': 'vision.googleapis.com',
        'language': 'language.googleapis.com',
        'translate': 'translate.googleapis.com',
        'storage': 'storage-api.googleapis.com',
        'compute': 'compute.googleapis.com',
        'aiplatform': 'aiplatform.googleapis.com',  # Vertex AI
        'bigquery': 'bigquery.googleapis.com',
        'run': 'run.googleapis.com',  # Cloud Run
        'cloudfunctions': 'cloudfunctions.googleapis.com',
        'pubsub': 'pubsub.googleapis.com',
        'documentai': 'documentai.googleapis.com',
        'speech': 'speech.googleapis.com',
        'texttospeech': 'texttospeech.googleapis.com',
        'healthcare': 'healthcare.googleapis.com',  # For biomedical data
        'lifesciences': 'lifesciences.googleapis.com',  # For genomics
    }
    
    def __init__(self):
        """Initialize the API manager."""
        self.authenticated = False
        self.project_id = None
        self.account_email = None
        self._check_authentication()
    
    def _check_authentication(self) -> None:
        """Check current authentication status."""
        try:
            # Check for authenticated accounts
            result = subprocess.run(
                ['gcloud', 'auth', 'list', '--format=json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                accounts = json.loads(result.stdout)
                active_account = next((acc for acc in accounts if acc.get('status') == 'ACTIVE'), None)
                
                if active_account:
                    self.authenticated = True
                    self.account_email = active_account.get('account')
                    logger.info(f"✅ Authenticated as: {self.account_email}")
                else:
                    logger.warning("⚠️ No active authentication found")
            
            # Check for project
            result = subprocess.run(
                ['gcloud', 'config', 'get-value', 'project'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip() != '(unset)':
                self.project_id = result.stdout.strip()
                logger.info(f"✅ Project configured: {self.project_id}")
            else:
                logger.warning("⚠️ No project configured")
                
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.error(f"❌ Error checking authentication: {e}")
    
    def authenticate(self, project_id: Optional[str] = None) -> bool:
        """
        Authenticate with Google Cloud.
        
        Args:
            project_id: Optional project ID to set
            
        Returns:
            bool: True if authentication successful
        """
        try:
            logger.info("🔐 Starting authentication process...")
            logger.info("📋 Please follow the instructions in your browser")
            
            # Authenticate
            result = subprocess.run(
                ['gcloud', 'auth', 'login', '--no-launch-browser'],
                capture_output=False,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("❌ Authentication failed")
                return False
            
            # Set project if provided
            if project_id:
                result = subprocess.run(
                    ['gcloud', 'config', 'set', 'project', project_id],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self.project_id = project_id
                    logger.info(f"✅ Project set to: {project_id}")
            
            # Refresh authentication status
            self._check_authentication()
            return self.authenticated
            
        except subprocess.SubprocessError as e:
            logger.error(f"❌ Authentication error: {e}")
            return False
    
    def setup_application_default_credentials(self) -> bool:
        """
        Set up Application Default Credentials for programmatic access.
        
        Returns:
            bool: True if setup successful
        """
        try:
            logger.info("🔧 Setting up Application Default Credentials...")
            logger.info("📋 This allows Python libraries to authenticate automatically")
            
            result = subprocess.run(
                ['gcloud', 'auth', 'application-default', 'login', '--no-launch-browser'],
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Application Default Credentials configured")
                return True
            else:
                logger.error("❌ ADC setup failed")
                return False
                
        except subprocess.SubprocessError as e:
            logger.error(f"❌ ADC setup error: {e}")
            return False
    
    def enable_api(self, api_name: str) -> Tuple[bool, str]:
        """
        Enable a specific Google Cloud API programmatically.
        
        Args:
            api_name: Either a friendly name from QUARK_REQUIRED_APIS or a full API name
            
        Returns:
            Tuple[bool, str]: (Success status, Message)
        """
        if not self.authenticated:
            return False, "❌ Not authenticated. Run authenticate() first"
        
        if not self.project_id:
            return False, "❌ No project configured. Set a project first"
        
        # Resolve API name
        api_id = self.QUARK_REQUIRED_APIS.get(api_name, api_name)
        
        try:
            logger.info(f"🔧 Enabling API: {api_id}")
            
            result = subprocess.run(
                ['gcloud', 'services', 'enable', api_id, '--quiet'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                message = f"✅ API enabled: {api_id}"
                logger.info(message)
                return True, message
            else:
                message = f"❌ Failed to enable {api_id}: {result.stderr}"
                logger.error(message)
                return False, message
                
        except subprocess.TimeoutExpired:
            message = f"⏱️ Timeout enabling {api_id} (may still complete)"
            logger.warning(message)
            return False, message
        except subprocess.SubprocessError as e:
            message = f"❌ Error enabling {api_id}: {e}"
            logger.error(message)
            return False, message
    
    def enable_multiple_apis(self, api_names: List[str]) -> Dict[str, Tuple[bool, str]]:
        """
        Enable multiple APIs at once.
        
        Args:
            api_names: List of API names to enable
            
        Returns:
            Dict mapping API name to (success, message) tuple
        """
        results = {}
        for api_name in api_names:
            results[api_name] = self.enable_api(api_name)
        return results
    
    def list_enabled_apis(self) -> List[str]:
        """
        List currently enabled APIs in the project.
        
        Returns:
            List of enabled API names
        """
        if not self.authenticated or not self.project_id:
            logger.error("❌ Must be authenticated with a project to list APIs")
            return []
        
        try:
            result = subprocess.run(
                ['gcloud', 'services', 'list', '--enabled', '--format=json'],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                apis = json.loads(result.stdout)
                return [api['config']['name'] for api in apis]
            else:
                logger.error(f"❌ Failed to list APIs: {result.stderr}")
                return []
                
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.error(f"❌ Error listing APIs: {e}")
            return []
    
    def check_api_status(self, api_name: str) -> bool:
        """
        Check if a specific API is enabled.
        
        Args:
            api_name: API to check (friendly name or full name)
            
        Returns:
            bool: True if API is enabled
        """
        api_id = self.QUARK_REQUIRED_APIS.get(api_name, api_name)
        enabled_apis = self.list_enabled_apis()
        return api_id in enabled_apis
    
    def enable_quark_essential_apis(self) -> Dict[str, Tuple[bool, str]]:
        """
        Enable essential APIs for Quark brain architecture.
        
        Returns:
            Results dictionary
        """
        essential_apis = [
            'vision',
            'language', 
            'translate',
            'storage',
            'aiplatform',
            'bigquery',
            'documentai'
        ]
        
        logger.info("🚀 Enabling essential Quark APIs...")
        return self.enable_multiple_apis(essential_apis)
    
    def get_authentication_status(self) -> Dict[str, any]:
        """
        Get detailed authentication status.
        
        Returns:
            Status dictionary
        """
        self._check_authentication()  # Refresh status
        
        status = {
            'authenticated': self.authenticated,
            'account': self.account_email,
            'project_id': self.project_id,
            'has_adc': False
        }
        
        # Check for Application Default Credentials
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'application-default', 'print-access-token'],
                capture_output=True,
                text=True,
                timeout=5
            )
            status['has_adc'] = result.returncode == 0
        except subprocess.SubprocessError:
            pass
        
        return status


def main():
    """Interactive CLI for Google Cloud API management."""
    manager = GoogleCloudAPIManager()
    
    print("🌐 Google Cloud API Manager for Quark")
    print("=" * 60)
    
    # Show status
    status = manager.get_authentication_status()
    print("\n📊 Current Status:")
    print(f"  • Authenticated: {'✅ Yes' if status['authenticated'] else '❌ No'}")
    print(f"  • Account: {status['account'] or 'None'}")
    print(f"  • Project: {status['project_id'] or 'None'}")
    print(f"  • Application Default Credentials: {'✅ Yes' if status['has_adc'] else '❌ No'}")
    
    if not status['authenticated']:
        print("\n⚠️ You need to authenticate first!")
        print("\nRun one of these commands:")
        print("  1. python google_cloud_api_manager.py --auth")
        print("  2. python google_cloud_api_manager.py --auth PROJECT_ID")
        return
    
    # List enabled APIs
    print("\n📋 Currently Enabled APIs:")
    enabled = manager.list_enabled_apis()
    if enabled:
        for api in enabled[:10]:  # Show first 10
            print(f"  • {api}")
        if len(enabled) > 10:
            print(f"  ... and {len(enabled) - 10} more")
    else:
        print("  None found or unable to list")
    
    print("\n💡 Available Commands:")
    print("  --auth [PROJECT_ID]      : Authenticate with Google Cloud")
    print("  --adc                    : Set up Application Default Credentials")
    print("  --enable API_NAME        : Enable a specific API")
    print("  --enable-essential       : Enable essential Quark APIs")
    print("  --list                   : List all enabled APIs")
    print("  --check API_NAME         : Check if an API is enabled")


if __name__ == "__main__":
    import sys
    
    manager = GoogleCloudAPIManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--auth":
            project_id = sys.argv[2] if len(sys.argv) > 2 else None
            if project_id:
                print(f"🔐 Authenticating with project: {project_id}")
            else:
                print("🔐 Authenticating (no project specified)")
            success = manager.authenticate(project_id)
            if success:
                print("✅ Authentication successful!")
            else:
                print("❌ Authentication failed")
        
        elif command == "--adc":
            success = manager.setup_application_default_credentials()
            if success:
                print("✅ Application Default Credentials configured!")
        
        elif command == "--enable" and len(sys.argv) > 2:
            api_name = sys.argv[2]
            success, message = manager.enable_api(api_name)
            print(message)
        
        elif command == "--enable-essential":
            print("🚀 Enabling essential Quark APIs...")
            results = manager.enable_quark_essential_apis()
            for api, (success, message) in results.items():
                print(f"  {api}: {message}")
        
        elif command == "--list":
            apis = manager.list_enabled_apis()
            print(f"\n📋 Enabled APIs ({len(apis)} total):")
            for api in apis:
                print(f"  • {api}")
        
        elif command == "--check" and len(sys.argv) > 2:
            api_name = sys.argv[2]
            enabled = manager.check_api_status(api_name)
            if enabled:
                print(f"✅ {api_name} is enabled")
            else:
                print(f"❌ {api_name} is not enabled")
        
        else:
            print("❌ Unknown command. Run without arguments for help.")
    else:
        main()
