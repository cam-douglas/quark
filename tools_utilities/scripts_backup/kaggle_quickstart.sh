#!/bin/bash

# 🚀 Kaggle Quick Start Script for SmallMind
# This script sets up Kaggle integration and uploads your models

set -e

echo "🚀 SmallMind Kaggle Integration Quick Start"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "src/models/models/configs/model_registry.json" ]; then
    echo "❌ Please run this script from the small-mind project root directory"
    exit 1
fi

# Install Kaggle CLI if not present
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle CLI..."
    pip install kaggle
else
    echo "✅ Kaggle CLI already installed"
fi

# Check Kaggle credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "🔐 Kaggle credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Download kaggle.json"
    echo "4. Place it in ~/.kaggle/kaggle.json"
    echo "5. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "After setting up credentials, run this script again."
    exit 1
fi

# Set proper permissions
chmod 600 "$HOME/.kaggle/kaggle.json"
chmod 700 "$HOME/.kaggle"

# Test Kaggle connection
echo "🔍 Testing Kaggle connection..."
if kaggle datasets list --limit 1 &> /dev/null; then
    echo "✅ Kaggle connection successful!"
else
    echo "❌ Kaggle connection failed. Please check your credentials."
    exit 1
fi

# Prepare models for upload
echo "📦 Preparing models for upload..."
python src/tools/kaggle_upload.py

# Ask user if they want to upload now
echo ""
read -p "🤔 Do you want to upload your models to Kaggle now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting upload process..."
    
    # Get username from kaggle.json
    USERNAME=$(grep -o '"username":"[^"]*"' "$HOME/.kaggle/kaggle.json" | cut -d'"' -f4)
    
    if [ -z "$USERNAME" ]; then
        echo "❌ Could not extract username from kaggle.json"
        exit 1
    fi
    
    echo "👤 Uploading as user: $USERNAME"
    
    # Upload Qwen1.5-MoE
    if [ -d "kaggle_export/qwen1.5-moe" ]; then
        echo "📤 Uploading Qwen1.5-MoE model..."
        cd kaggle_export/qwen1.5-moe
        
        # Create dataset-metadata.yaml
        cat > dataset-metadata.yaml << EOF
title: "SmallMind Qwen1.5-MoE - Brain Development Model"
id: "$USERNAME/smallmind-qwen15-moe"
licenses:
  - name: "mit"
description: "Trained Qwen1.5-MoE model for brain development simulation and neuroscience research"
keywords:
  - "brain-development"
  - "neuroscience"
  - "machine-learning"
  - "transformers"
  - "qwen"
  - "moe"
EOF
        
        kaggle datasets create -p .
        echo "✅ Qwen1.5-MoE uploaded successfully!"
        cd ../..
    fi
    
    # Upload DeepSeek-v2
    if [ -d "kaggle_export/deepseek-v2" ]; then
        echo "📤 Uploading DeepSeek-v2 model..."
        cd kaggle_export/deepseek-v2
        
        # Create dataset-metadata.yaml
        cat > dataset-metadata.yaml << EOF
title: "SmallMind DeepSeek-v2 - Brain Development Model"
id: "$USERNAME/smallmind-deepseek-v2"
licenses:
  - name: "mit"
description: "Trained DeepSeek-v2 model for brain development simulation and neuroscience research"
keywords:
  - "brain-development"
  - "neuroscience"
  - "machine-learning"
  - "transformers"
  - "deepseek"
EOF
        
        kaggle datasets create -p .
        echo "✅ DeepSeek-v2 uploaded successfully!"
        cd ../..
    fi
    
    echo ""
    echo "🎉 All models uploaded successfully!"
    echo ""
    echo "📊 Your datasets are now available at:"
    echo "   https://www.kaggle.com/$USERNAME"
    echo ""
    echo "🔗 You can now:"
    echo "   - Share your models with the community"
    echo "   - Use them in Kaggle competitions"
    echo "   - Collaborate with other researchers"
    echo "   - Track usage and get feedback"
    
else
    echo "📋 Models prepared but not uploaded."
    echo "Run the following commands when ready:"
    echo ""
    echo "  cd kaggle_export/qwen1.5-moe"
    echo "  kaggle datasets create -p ."
    echo ""
    echo "  cd ../deepseek-v2"
    echo "  kaggle datasets create -p ."
fi

echo ""
echo "✨ Kaggle integration setup complete!"
echo "📚 See docs/KAGGLE_INTEGRATION.md for detailed information"
