#!/usr/bin/env python3
"""
Google Drive Setup Script for Quark Project
Simplified setup process for cloud storage integration.
"""

import os
import subprocess
import json
from pathlib import Path

def check_rclone():
    """Check if rclone is installed and working."""
    try:
        result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ rclone is installed and working")
            return True
        else:
            print("❌ rclone is installed but not working")
            return False
    except FileNotFoundError:
        print("❌ rclone not found. Please install it first:")
        print("   brew install rclone")
        return False

def setup_google_drive():
    """Guide user through Google Drive setup."""
    print("\n🚀 Setting up Google Drive Integration")
    print("=" * 50)
    
    # Check rclone
    if not check_rclone():
        return False
    
    print("\n📋 Google Drive Setup Steps:")
    print("1. Open a new terminal window")
    print("2. Run: rclone config")
    print("3. Follow these steps:")
    print("   - Type 'n' for new remote")
    print("   - Name it 'gdrive'")
    print("   - Choose 'Google Drive' (option 18)")
    print("   - Choose 'Application Default Credentials' (option 1)")
    print("   - Leave client_id and client_secret blank (press Enter)")
    print("   - Choose 'Yes' to use auto config")
    print("   - Your browser will open for Google authentication")
    print("   - Sign in with your Google account")
    print("   - Grant permissions to rclone")
    print("   - Copy the verification code back to terminal")
    print("   - Choose 'Yes' to configure as team drive")
    print("   - Choose 'No' for advanced config")
    print("   - Type 'y' to save")
    print("   - Type 'q' to quit")
    
    print("\n⏳ After completing the setup above, press Enter to continue...")
    input()
    
    # Test the connection
    print("\n🧪 Testing Google Drive connection...")
    try:
        result = subprocess.run(["rclone", "lsd", "gdrive:"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Google Drive connection successful!")
            print("Available folders:")
            print(result.stdout)
            return True
        else:
            print("❌ Google Drive connection failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def create_cloud_folder():
    """Create a folder in Google Drive for the Quark project."""
    print("\n📁 Creating Quark project folder in Google Drive...")
    
    folder_name = "Quark_Project_Cloud"
    
    try:
        # Create the folder
        result = subprocess.run(["rclone", "mkdir", f"gdrive:{folder_name}"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Created folder: {folder_name}")
            
            # List contents to verify
            result = subprocess.run(["rclone", "lsd", "gdrive:"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Google Drive contents:")
                print(result.stdout)
            
            return True
        else:
            print(f"❌ Failed to create folder: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating folder: {e}")
        return False

def test_file_upload():
    """Test uploading a small file to Google Drive."""
    print("\n🧪 Testing file upload to Google Drive...")
    
    # Create a test file
    test_file = "test_upload.txt"
    test_content = "This is a test file for Google Drive integration."
    
    try:
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Upload to Google Drive
        result = subprocess.run(["rclone", "copy", test_file, "gdrive:Quark_Project_Cloud/"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Test file uploaded successfully!")
            
            # List files in the folder
            result = subprocess.run(["rclone", "ls", "gdrive:Quark_Project_Cloud/"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Files in Quark_Project_Cloud:")
                print(result.stdout)
            
            # Clean up test file
            os.remove(test_file)
            return True
        else:
            print(f"❌ Upload failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during upload test: {e}")
        return False

def update_cloud_integration():
    """Update the cloud integration script with Google Drive configuration."""
    print("\n🔧 Updating cloud integration configuration...")
    
    config_file = ".cloud_storage_config.json"
    
    config = {
        "cloud_services": {
            "google_drive": {
                "enabled": True,
                "folder_name": "Quark_Project_Cloud",
                "free_gb": 15,
                "sync_method": "rclone"
            }
        },
        "local_references": {
            "use_symbolic_links": True,
            "fallback_to_references": True,
            "reference_file_suffix": ".cloud_ref"
        },
        "sync_settings": {
            "auto_sync": True,
            "sync_interval_minutes": 30,
            "preserve_local_changes": True
        }
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def main():
    """Main setup process."""
    print("🚀 Google Drive Setup for Quark Project")
    print("=" * 50)
    
    # Step 1: Setup Google Drive
    if not setup_google_drive():
        print("\n❌ Google Drive setup failed. Please try again.")
        return
    
    # Step 2: Create project folder
    if not create_cloud_folder():
        print("\n❌ Failed to create project folder. Please try again.")
        return
    
    # Step 3: Test file upload
    if not test_file_upload():
        print("\n❌ File upload test failed. Please check your setup.")
        return
    
    # Step 4: Update configuration
    if not update_cloud_integration():
        print("\n❌ Failed to update configuration.")
        return
    
    print("\n🎉 Google Drive setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python3 cloud_storage_integration.py --setup google_drive")
    print("2. Run: python3 cloud_storage_integration.py --migrate 'results/**/*.png' '*.pth'")
    print("3. Check: python3 cloud_storage_integration.py --list")
    
    print("\n✅ Your Quark project is now ready for cloud storage!")

if __name__ == "__main__":
    main()
