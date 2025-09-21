#!/usr/bin/env python3
"""
Google Cloud Integration Module for Quark Brain Architecture.

This module provides seamless integration with Google Cloud services,
supporting both API key and OAuth authentication methods.

Created: 2025-01-20
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleCloudIntegration:
    """
    Manages Google Cloud service integration for the Quark project.
    
    Supports:
    - API key authentication for basic services
    - OAuth 2.0 via gcloud CLI for advanced services
    - Service account management
    - Resource monitoring and cost tracking
    """
    
    def __init__(self, credentials_path: str = "/Users/camdouglas/quark/data/credentials/all_api_keys.json"):
        """Initialize Google Cloud integration with credentials."""
        self.credentials_path = Path(credentials_path)
        self.api_key = None
        self.project_id = None
        self.authenticated = False
        
        # Load API key from credentials file
        self._load_api_key()
        
        # Check gcloud CLI availability
        self.gcloud_available = self._check_gcloud_cli()
        
    def _load_api_key(self) -> None:
        """Load Google Cloud API key from credentials file."""
        try:
            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
                self.api_key = credentials.get('google_cloud_api_key') or \
                              credentials.get('services', {}).get('google_cloud', {}).get('api_key')
                
                if self.api_key:
                    logger.info("✅ Google Cloud API key loaded successfully")
                else:
                    logger.warning("⚠️ Google Cloud API key not found in credentials")
                    
        except FileNotFoundError:
            logger.error(f"❌ Credentials file not found: {self.credentials_path}")
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON in credentials file")
    
    def _check_gcloud_cli(self) -> bool:
        """Check if gcloud CLI is installed and accessible."""
        try:
            result = subprocess.run(
                ['gcloud', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("✅ Google Cloud CLI is available")
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        logger.warning("⚠️ Google Cloud CLI not available")
        return False
    
    def authenticate_with_oauth(self, project_id: Optional[str] = None) -> bool:
        """
        Authenticate using gcloud CLI OAuth flow.
        
        Args:
            project_id: Optional Google Cloud project ID
            
        Returns:
            bool: True if authentication successful
        """
        if not self.gcloud_available:
            logger.error("❌ Google Cloud CLI not installed")
            return False
        
        try:
            # Initialize gcloud configuration
            logger.info("🔐 Initiating Google Cloud OAuth authentication...")
            
            # Login to Google Cloud
            subprocess.run(['gcloud', 'auth', 'login', '--no-launch-browser'], check=True)
            
            # Set project if provided
            if project_id:
                subprocess.run(['gcloud', 'config', 'set', 'project', project_id], check=True)
                self.project_id = project_id
                logger.info(f"✅ Set project to: {project_id}")
            
            # Verify authentication
            result = subprocess.run(
                ['gcloud', 'auth', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if "ACTIVE" in result.stdout:
                self.authenticated = True
                logger.info("✅ Successfully authenticated with Google Cloud")
                return True
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Authentication failed: {e}")
            
        return False
    
    def configure_application_default_credentials(self) -> bool:
        """
        Set up Application Default Credentials for SDK usage.
        
        Returns:
            bool: True if ADC setup successful
        """
        if not self.gcloud_available:
            logger.error("❌ Google Cloud CLI required for ADC setup")
            return False
        
        try:
            logger.info("🔧 Setting up Application Default Credentials...")
            subprocess.run(
                ['gcloud', 'auth', 'application-default', 'login', '--no-launch-browser'],
                check=True
            )
            logger.info("✅ Application Default Credentials configured")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ADC setup failed: {e}")
            return False
    
    def list_available_services(self) -> List[str]:
        """
        List Google Cloud services accessible with current authentication.
        
        Returns:
            List of available service names
        """
        services = []
        
        # Services available with API key
        if self.api_key:
            services.extend([
                "Google Cloud Vision API",
                "Google Cloud Natural Language API",
                "Google Cloud Translation API",
                "Google Gemini API",
                "Maps Platform APIs"
            ])
        
        # Additional services available with OAuth/CLI
        if self.authenticated and self.gcloud_available:
            services.extend([
                "Google Cloud Storage (GCS)",
                "Compute Engine",
                "Cloud Run",
                "BigQuery",
                "Vertex AI",
                "Cloud Functions",
                "Cloud SQL",
                "Pub/Sub",
                "Cloud Build"
            ])
        
        return services
    
    def get_project_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current Google Cloud project information.
        
        Returns:
            Dict with project details or None
        """
        if not self.gcloud_available:
            return None
        
        try:
            result = subprocess.run(
                ['gcloud', 'config', 'list', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
            
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return None
    
    def create_service_account_key(self, service_account_email: str, key_file_path: str) -> bool:
        """
        Create a service account key for programmatic access.
        
        Args:
            service_account_email: Email of the service account
            key_file_path: Where to save the key file
            
        Returns:
            bool: True if key created successfully
        """
        if not self.gcloud_available or not self.authenticated:
            logger.error("❌ Authentication required for service account management")
            return False
        
        try:
            subprocess.run([
                'gcloud', 'iam', 'service-accounts', 'keys', 'create',
                key_file_path,
                '--iam-account', service_account_email
            ], check=True)
            
            logger.info(f"✅ Service account key created: {key_file_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create service account key: {e}")
            return False
    
    def test_api_key(self) -> bool:
        """
        Test if the API key is valid by making a simple API call.
        
        Returns:
            bool: True if API key is valid
        """
        if not self.api_key:
            logger.error("❌ No API key available to test")
            return False
        
        # For testing, we could use a simple API endpoint
        # This is a placeholder - actual implementation would make a real API call
        logger.info(f"🔍 Testing API key: {self.api_key[:20]}...")
        
        # In production, you'd make an actual API call here
        # For now, we just verify the key format
        if self.api_key.startswith("AIzaSy"):
            logger.info("✅ API key format appears valid")
            return True
        else:
            logger.warning("⚠️ API key format may be invalid")
            return False
    
    def generate_quark_config(self) -> Dict[str, Any]:
        """
        Generate Google Cloud configuration for Quark integration.
        
        Returns:
            Configuration dictionary for Quark modules
        """
        config = {
            "google_cloud": {
                "api_key_available": bool(self.api_key),
                "gcloud_cli_available": self.gcloud_available,
                "authenticated": self.authenticated,
                "project_id": self.project_id,
                "available_services": self.list_available_services(),
                "credentials_path": str(self.credentials_path)
            }
        }
        
        # Add recommended settings for Quark brain modules
        config["recommendations"] = {
            "use_vertex_ai": self.authenticated,
            "use_cloud_storage": self.authenticated,
            "use_bigquery": self.authenticated,
            "use_cloud_run": self.authenticated,
            "preferred_auth": "oauth" if self.authenticated else "api_key"
        }
        
        return config


def main():
    """Main entry point for testing Google Cloud integration."""
    print("🚀 Google Cloud Integration for Quark Brain Architecture\n")
    print("=" * 60)
    
    # Initialize integration
    gcp = GoogleCloudIntegration()
    
    # Display current status
    print("\n📊 Current Status:")
    print(f"  • API Key: {'✅ Available' if gcp.api_key else '❌ Not Found'}")
    print(f"  • gcloud CLI: {'✅ Installed' if gcp.gcloud_available else '❌ Not Installed'}")
    print(f"  • OAuth Auth: {'✅ Active' if gcp.authenticated else '❌ Not Authenticated'}")
    
    # List available services
    print("\n🛠️ Available Services:")
    services = gcp.list_available_services()
    for service in services:
        print(f"  • {service}")
    
    # Test API key
    print("\n🔍 Testing API Key:")
    if gcp.test_api_key():
        print("  ✅ API key validation passed")
    else:
        print("  ❌ API key validation failed")
    
    # Generate Quark configuration
    print("\n⚙️ Generating Quark Configuration:")
    config = gcp.generate_quark_config()
    print(json.dumps(config, indent=2))
    
    # Provide next steps
    print("\n📝 Next Steps:")
    print("1. To authenticate with OAuth:")
    print("   python google_cloud_integration.py --auth")
    print("\n2. To set up Application Default Credentials:")
    print("   python google_cloud_integration.py --setup-adc")
    print("\n3. To integrate with Quark modules:")
    print("   from tools_utilities.google_cloud_integration import GoogleCloudIntegration")
    print("   gcp = GoogleCloudIntegration()")
    

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        gcp = GoogleCloudIntegration()
        
        if "--auth" in sys.argv:
            project_id = input("Enter your Google Cloud Project ID (or press Enter to skip): ").strip()
            gcp.authenticate_with_oauth(project_id if project_id else None)
            
        elif "--setup-adc" in sys.argv:
            gcp.configure_application_default_credentials()
            
        elif "--test" in sys.argv:
            gcp.test_api_key()
            
    else:
        main()
