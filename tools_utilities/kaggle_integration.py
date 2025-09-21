#!/usr/bin/env python3
"""
Kaggle API Integration Module for Quark Brain Architecture.

This module manages Kaggle API authentication and dataset access.
Created: 2025-01-20
"""

import json
import os
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleIntegration:
    """Manages Kaggle API integration for the Quark project."""
    
    def __init__(self, credentials_path: str = "/Users/camdouglas/quark/data/credentials/all_api_keys.json"):
        """Initialize Kaggle integration."""
        self.credentials_path = Path(credentials_path)
        self.kaggle_dir = Path.home() / ".kaggle"
        self.kaggle_json = self.kaggle_dir / "kaggle.json"
        self.api_key_from_file = None
        self.username = None
        
        # Load credentials
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load Kaggle credentials from various sources."""
        # Load from all_api_keys.json
        try:
            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
                self.api_key_from_file = credentials.get('kaggle_api_key') or \
                                        credentials.get('services', {}).get('kaggle', {}).get('api_key')
                logger.info(f"✅ Found Kaggle API key in credentials file: {self.api_key_from_file[:10]}...")
        except Exception as e:
            logger.error(f"❌ Error loading credentials: {e}")
        
        # Check existing kaggle.json
        if self.kaggle_json.exists():
            try:
                with open(self.kaggle_json, 'r') as f:
                    kaggle_creds = json.load(f)
                    self.username = kaggle_creds.get('username')
                    existing_key = kaggle_creds.get('key')
                    logger.info(f"✅ Found existing Kaggle config for user: {self.username}")
                    
                    if existing_key != self.api_key_from_file:
                        logger.warning("⚠️ API key mismatch between kaggle.json and all_api_keys.json")
                        logger.info(f"   kaggle.json key: {existing_key[:10]}...")
                        logger.info(f"   all_api_keys key: {self.api_key_from_file[:10] if self.api_key_from_file else 'None'}...")
            except Exception as e:
                logger.error(f"❌ Error reading kaggle.json: {e}")
    
    def test_api_access(self) -> bool:
        """Test Kaggle API access with current credentials."""
        try:
            logger.info("🔍 Testing Kaggle API access...")
            
            # Try to list datasets (lightweight API call)
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '--sort-by', 'hottest', '--max-size', '1'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("✅ Kaggle API access confirmed!")
                return True
            else:
                logger.error(f"❌ Kaggle API error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Kaggle API request timed out")
            return False
        except FileNotFoundError:
            logger.error("❌ Kaggle CLI not found. Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False
    
    def get_account_info(self) -> dict:
        """Get Kaggle account information."""
        try:
            # Use kaggle API to get account info
            result = subprocess.run(
                ['kaggle', 'config', 'view'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse the output
                info = {}
                for line in result.stdout.split('\n'):
                    if 'username' in line.lower():
                        info['username'] = line.split(':')[-1].strip()
                    elif 'competition' in line.lower():
                        info['competition'] = line
                        
                return info
            else:
                return {'error': result.stderr}
                
        except Exception as e:
            return {'error': str(e)}
    
    def update_kaggle_json(self, username: str = None, api_key: str = None) -> bool:
        """Update the kaggle.json file with provided credentials."""
        try:
            # Use provided values or fall back to loaded ones
            username = username or self.username or input("Enter Kaggle username: ")
            api_key = api_key or self.api_key_from_file
            
            if not username or not api_key:
                logger.error("❌ Username and API key are required")
                return False
            
            # Create .kaggle directory if it doesn't exist
            self.kaggle_dir.mkdir(exist_ok=True)
            
            # Write kaggle.json
            kaggle_config = {
                "username": username,
                "key": api_key
            }
            
            with open(self.kaggle_json, 'w') as f:
                json.dump(kaggle_config, f)
            
            # Set proper permissions (read/write for user only)
            os.chmod(self.kaggle_json, 0o600)
            
            logger.info(f"✅ Updated {self.kaggle_json}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update kaggle.json: {e}")
            return False
    
    def download_dataset(self, dataset_path: str, output_dir: str = "data/datasets") -> bool:
        """
        Download a Kaggle dataset.
        
        Args:
            dataset_path: Dataset path in format "owner/dataset-name"
            output_dir: Directory to save the dataset
            
        Returns:
            bool: Success status
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📦 Downloading dataset: {dataset_path}")
            
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset_path, '-p', str(output_path), '--unzip'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Dataset downloaded to {output_path}")
                return True
            else:
                logger.error(f"❌ Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            return False
    
    def search_datasets(self, search_term: str, max_results: int = 5) -> list:
        """
        Search for datasets on Kaggle.
        
        Args:
            search_term: Search query
            max_results: Maximum number of results
            
        Returns:
            List of dataset information
        """
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '-s', search_term, '--max-size', str(max_results)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    return lines[1:]
                return []
            else:
                logger.error(f"Search failed: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            return []
    
    def list_competitions(self) -> list:
        """List active Kaggle competitions."""
        try:
            result = subprocess.run(
                ['kaggle', 'competitions', 'list'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
            else:
                return [f"Error: {result.stderr}"]
                
        except Exception as e:
            return [f"Error: {str(e)}"]


def main():
    """Test Kaggle integration."""
    print("🎯 Kaggle API Integration for Quark")
    print("=" * 60)
    
    # Initialize
    kaggle = KaggleIntegration()
    
    # Show current status
    print("\n📊 Current Configuration:")
    print(f"  • Kaggle config exists: {kaggle.kaggle_json.exists()}")
    print(f"  • Username: {kaggle.username or 'Not set'}")
    print(f"  • API key in credentials: {'✅ Yes' if kaggle.api_key_from_file else '❌ No'}")
    
    # Test API access
    print("\n🔍 Testing API Access:")
    if kaggle.test_api_access():
        print("  ✅ API access successful!")
        
        # Get account info
        info = kaggle.get_account_info()
        if 'error' not in info:
            print("\n👤 Account Information:")
            for key, value in info.items():
                print(f"  • {key}: {value}")
        
        # Search for a dataset
        print("\n🔎 Sample Dataset Search (brain imaging):")
        results = kaggle.search_datasets("brain imaging", max_results=3)
        for result in results:
            print(f"  • {result}")
    else:
        print("  ❌ API access failed")
        print("\n💡 To fix:")
        print("  1. Ensure you have a Kaggle account")
        print("  2. Get your API key from https://www.kaggle.com/account")
        print("  3. Update credentials:")
        print("     python kaggle_integration.py --update")
    
    print("\n📝 Available Commands:")
    print("  --update          : Update Kaggle credentials")
    print("  --test            : Test API access")
    print("  --search TERM     : Search for datasets")
    print("  --download DATASET: Download a dataset")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        kaggle = KaggleIntegration()
        command = sys.argv[1]
        
        if command == "--update":
            username = input("Enter Kaggle username (or press Enter to use existing): ").strip()
            api_key = input("Enter Kaggle API key (or press Enter to use from credentials): ").strip()
            
            if kaggle.update_kaggle_json(username or None, api_key or None):
                print("✅ Credentials updated successfully!")
                kaggle.test_api_access()
        
        elif command == "--test":
            if kaggle.test_api_access():
                print("✅ Kaggle API is working!")
            else:
                print("❌ Kaggle API test failed")
        
        elif command == "--search" and len(sys.argv) > 2:
            search_term = " ".join(sys.argv[2:])
            print(f"🔎 Searching for: {search_term}")
            results = kaggle.search_datasets(search_term)
            for result in results:
                print(result)
        
        elif command == "--download" and len(sys.argv) > 2:
            dataset = sys.argv[2]
            kaggle.download_dataset(dataset)
        
        else:
            print("Unknown command. Run without arguments for help.")
    else:
        main()
