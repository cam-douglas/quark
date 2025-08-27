#!/bin/bash
# Quark Brain Simulation Setup Script for Tokyo Instance
# Instance: quark-tokyo (c5.xlarge)
# Generated: 2025-08-28 05:00:34

echo "🧠⚡ Setting up Quark Brain Simulation System"
echo "=========================================="
echo "Instance: quark-tokyo in ap-northeast-1"
echo "Storage: 200GB | RAM: 8GB"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "🐍 Installing Python and dependencies..."
sudo apt install -y python3 python3-pip python3-venv git curl wget

# Create Quark directory and virtual environment
echo "🧠 Setting up Quark environment..."
mkdir -p ~/quark
cd ~/quark
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "📚 Installing Python packages..."
pip install boto3 amazon-braket-sdk anthropic torch numpy pandas scipy matplotlib tqdm requests

# Configure AWS CLI (assumes credentials are already set)
echo "⚙️ Configuring AWS for Tokyo region..."
aws configure set region ap-northeast-1

# Create cache directories
echo "💾 Setting up local cache..."
mkdir -p ~/.quark/model_cache/models
mkdir -p ~/.quark/model_cache/datasets

# Download Quark state system
echo "📡 Setting up Quark state system..."
# This would download the Quark repository or specific files

# Test S3 connectivity
echo "🔗 Testing S3 connectivity..."
aws s3 ls s3://quark-tokyo-bucket || echo "⚠️ S3 access issue - check credentials"

# Test other AWS services
echo "🧪 Testing AWS services..."
aws bedrock list-foundation-models --region us-east-1 --max-items 1 > /dev/null && echo "✅ Bedrock accessible" || echo "⚠️ Bedrock access issue"

echo ""
echo "🎉 Quark setup complete!"
echo "💡 Run 'python quark_state_system/s3_model_manager.py' to initialize models"
echo "🚀 Start with: source venv/bin/activate && python QUARK_STATE_SYSTEM.py status"
