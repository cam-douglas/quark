set -euo pipefail

#!/bin/bash
# Quick SmallMind AWS Deployment

echo "🧠 SmallMind Quick AWS Deployment"
echo "=================================="

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &>/dev/null; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if boto3 is installed
if ! python3 -c "import boto3" &>/dev/null; then
    echo "📦 Installing boto3..."
    pip3 install boto3
fi

# Deploy
echo "🚀 Starting deployment..."
python3 aws_deploy.py --key-name "$1" --region "${2:-us-east-1}" --instance-type "${3:-g4dn.xlarge}"

echo ""
echo "🎉 Deployment initiated!"
echo "⏳ Wait 5-10 minutes for setup to complete."
echo "📱 Check the URLs above to access your SmallMind instance."
