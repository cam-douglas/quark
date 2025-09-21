#!/usr/bin/env python3
"""
Interactive Google Cloud Authentication Helper for Quark.
Project: quark-469604
Created: 2025-01-20
"""

import subprocess
import sys
import time

def authenticate_with_code():
    """Interactive authentication with verification code."""
    print("🔐 Google Cloud Authentication for Project: quark-469604")
    print("=" * 60)
    
    # Start the authentication process
    print("\n📋 Starting authentication process...")
    print("\nPaste your verification code when prompted.")
    print("Code format should be: 4/0A...")
    print("\nYour code: 4/0AVGzR1CWjZB6XLPpM4dsdED_7FDhR2eEsJfMTaj0VTwURSdF_zF9WUv6i86EuMuOawM3hw")
    print("\n" + "=" * 60)
    
    try:
        # Run gcloud auth login interactively
        result = subprocess.run(
            ['gcloud', 'auth', 'login', '--no-launch-browser'],
            input="4/0AVGzR1CWjZB6XLPpM4dsdED_7FDhR2eEsJfMTaj0VTwURSdF_zF9WUv6i86EuMuOawM3hw\n",
            text=True,
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n✅ Authentication successful!")
            return True
        else:
            print("\n❌ Authentication failed. Trying fresh authentication...")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    """Main authentication flow."""
    # First attempt with existing code
    if not authenticate_with_code():
        print("\n🔄 Starting fresh authentication...")
        print("\n1. Running: gcloud auth login --no-launch-browser")
        print("2. A new URL will be displayed")
        print("3. Open the URL in your browser")
        print("4. Get a new verification code")
        print("5. Paste it when prompted\n")
        
        # Start fresh authentication
        subprocess.run(['gcloud', 'auth', 'login', '--no-launch-browser'])
    
    # Verify authentication
    print("\n📊 Checking authentication status...")
    subprocess.run(['gcloud', 'auth', 'list'])
    
    print("\n🎯 Next steps:")
    print("1. Set up Application Default Credentials:")
    print("   gcloud auth application-default login --no-launch-browser")
    print("\n2. Verify setup:")
    print("   python tools_utilities/google_cloud_api_manager.py")
    print("\n3. Enable essential APIs:")
    print("   python tools_utilities/google_cloud_api_manager.py --enable-essential")

if __name__ == "__main__":
    main()
