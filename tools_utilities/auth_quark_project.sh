#!/bin/bash
# Google Cloud Authentication Script for Quark Project
# Project ID: quark-469604
# Created: 2025-01-20

echo "🔐 Authenticating Google Cloud Project: quark-469604"
echo "=" * 60

# Step 1: Set the project
echo "📋 Step 1: Setting project to quark-469604..."
gcloud config set project quark-469604

# Step 2: Show authentication URL
echo ""
echo "📋 Step 2: Please authenticate by:"
echo "1. Opening this URL in your browser:"
echo ""
echo "https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=32555940559.apps.googleusercontent.com&redirect_uri=https%3A%2F%2Fsdk.cloud.google.com%2Fauthcode.html&scope=openid+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fappengine.admin+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fsqlservice.login+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcompute+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faccounts.reauth&state=o7rTev28pnEfdwc8uBf2gU03ZRF5Lo&prompt=consent&token_usage=remote&access_type=offline&code_challenge=NqrMeNe2E3f9FwQp3Oyl4Rw1tk9Kptuq4hrmnLqiRRQ&code_challenge_method=S256"
echo ""
echo "2. Complete the sign-in process"
echo "3. Copy the verification code"
echo "4. Run: gcloud auth login --no-launch-browser"
echo "5. Paste the verification code when prompted"

# Step 3: Check status
echo ""
echo "📋 Step 3: After authentication, run:"
echo "python tools_utilities/google_cloud_api_manager.py --list"
