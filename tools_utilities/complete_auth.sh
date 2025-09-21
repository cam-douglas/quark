#!/bin/bash
# Quick authentication completion script
# Just paste your verification code when prompted

echo "🔐 Google Cloud Authentication Helper"
echo "===================================="
echo ""
echo "When prompted, paste your verification code and press Enter."
echo "The code should look like: 4/0A..."
echo ""

# Run authentication
gcloud auth login --no-launch-browser

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Authentication successful!"
    echo ""
    echo "📊 Checking account status..."
    gcloud auth list
    echo ""
    echo "🎯 Next: Set up Application Default Credentials"
    echo "Run: gcloud auth application-default login --no-launch-browser"
else
    echo ""
    echo "❌ Authentication failed. Please try again with a fresh code."
fi
