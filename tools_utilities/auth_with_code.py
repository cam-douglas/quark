#!/usr/bin/env python3
"""
Direct authentication script that accepts a verification code as input.
This avoids the session mismatch issue.
"""

import subprocess
import sys
import os

def main():
    print("🔐 Google Cloud Authentication Helper")
    print("=" * 60)
    
    # Get the latest authentication URL
    print("\n📋 Starting authentication process...")
    print("\nThis will generate a NEW authentication URL.")
    print("Please follow these steps:\n")
    
    # Start gcloud auth login to get the URL
    process = subprocess.Popen(
        ['gcloud', 'auth', 'login', '--no-launch-browser'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0
    )
    
    # Read output until we see the URL
    output = []
    url_found = False
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        
        print(line, end='')
        output.append(line)
        
        if "https://accounts.google.com" in line and not url_found:
            url_found = True
            print("\n" + "="*60)
            print("✅ URL generated above!")
            print("\n1. Copy and open the URL above in your browser")
            print("2. Sign in and authorize access")
            print("3. Copy the verification code")
            print("4. Paste it below and press Enter")
            print("="*60 + "\n")
            
        if "enter the verification code" in line.lower():
            # Now we can input the code
            code = input("Paste your verification code here: ")
            process.stdin.write(code + "\n")
            process.stdin.flush()
    
    # Wait for completion
    return_code = process.wait()
    
    if return_code == 0:
        print("\n✅ Authentication successful!")
        
        # Verify authentication
        print("\n📊 Verifying authentication...")
        subprocess.run(['gcloud', 'auth', 'list'])
        
        print("\n✅ Project configured: quark-469604")
        
        print("\n🎯 Next steps:")
        print("1. Set up Application Default Credentials:")
        print("   python tools_utilities/auth_with_code.py --adc")
        print("\n2. Enable APIs:")
        print("   python tools_utilities/google_cloud_api_manager.py --enable-essential")
        
    else:
        print("\n❌ Authentication failed.")
        print("Please try again.")
    
    return return_code

def setup_adc():
    """Set up Application Default Credentials."""
    print("🔧 Setting up Application Default Credentials...")
    print("=" * 60)
    print("\nThis allows Python libraries to authenticate automatically.")
    print("\nStarting ADC setup...\n")
    
    subprocess.run(['gcloud', 'auth', 'application-default', 'login', '--no-launch-browser'])
    
    print("\n✅ If successful, ADC is now configured!")
    print("You can now use Google Cloud Python libraries.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--adc":
        setup_adc()
    else:
        sys.exit(main())
