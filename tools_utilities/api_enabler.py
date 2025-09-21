#!/usr/bin/env python3
"""
Quick API enabler for Quark essential services.
"""

import subprocess
import sys

# Essential APIs for Quark Brain Architecture
ESSENTIAL_APIS = {
    'vision.googleapis.com': 'Cloud Vision (image analysis)',
    'language.googleapis.com': 'Natural Language (text analysis)',
    'translate.googleapis.com': 'Translation API',
    'aiplatform.googleapis.com': 'Vertex AI (ML platform)',
    'documentai.googleapis.com': 'Document AI (document processing)',
    'healthcare.googleapis.com': 'Healthcare API (medical data)',
    'lifesciences.googleapis.com': 'Life Sciences (genomics)',
    'speech.googleapis.com': 'Speech-to-Text',
    'texttospeech.googleapis.com': 'Text-to-Speech',
}

def check_and_enable_api(api_name, description):
    """Check if API is enabled and offer to enable it."""
    print(f"\n📍 Checking {api_name} ({description})...")
    
    # Check if enabled
    result = subprocess.run(
        ['gcloud', 'services', 'list', '--enabled', '--filter', f'name:{api_name}', '--format=json'],
        capture_output=True,
        text=True
    )
    
    import json
    enabled_list = json.loads(result.stdout) if result.returncode == 0 else []
    
    if enabled_list:
        print(f"   ✅ Already enabled")
        return True
    else:
        print(f"   ❌ Not enabled")
        response = input(f"   Enable {description}? (y/n): ").strip().lower()
        
        if response == 'y':
            print(f"   🔧 Enabling {api_name}...")
            result = subprocess.run(
                ['gcloud', 'services', 'enable', api_name, '--quiet'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"   ✅ Successfully enabled!")
                return True
            else:
                print(f"   ❌ Failed to enable: {result.stderr}")
                return False
        return False

def main():
    print("🚀 Google Cloud API Enabler for Quark")
    print("=" * 60)
    print("\nProject: quark-469604")
    print("Account: shuffle.ops@gmail.com")
    print("\nChecking essential APIs for Quark brain architecture...")
    
    enabled_count = 0
    total_count = len(ESSENTIAL_APIS)
    
    for api, description in ESSENTIAL_APIS.items():
        if check_and_enable_api(api, description):
            enabled_count += 1
    
    print("\n" + "=" * 60)
    print(f"\n📊 Summary: {enabled_count}/{total_count} essential APIs enabled")
    
    if enabled_count == total_count:
        print("✅ All essential APIs are enabled!")
    else:
        print(f"⚠️ {total_count - enabled_count} APIs still need to be enabled")
    
    print("\n🎯 Next: Set up Application Default Credentials")
    print("Run: gcloud auth application-default login --no-launch-browser")

if __name__ == "__main__":
    main()
