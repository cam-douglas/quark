#!/bin/bash

echo "═══════════════════════════════════════════════════════════════"
echo "    🔐 Google Cloud Authentication for Project: quark-469604"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Starting authentication process..."
echo ""

# Run the authentication command
gcloud auth login --no-launch-browser

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Authentication successful!"
    echo ""
    echo "📊 Your authenticated accounts:"
    gcloud auth list
    echo ""
    echo "📋 Current project: quark-469604"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Set up Application Default Credentials:"
    echo "   gcloud auth application-default login --no-launch-browser"
    echo ""
    echo "2. Enable essential APIs:"
    echo "   python tools_utilities/google_cloud_api_manager.py --enable-essential"
else
    echo ""
    echo "❌ Authentication failed. Please try again."
fi
