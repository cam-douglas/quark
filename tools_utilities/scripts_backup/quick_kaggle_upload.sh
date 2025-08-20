#!/bin/bash
# Quick Kaggle Upload Script for Quark Brain Simulation

echo "🚀 Starting Kaggle upload process..."

# Check Kaggle installation
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check authentication
if ! kaggle datasets list &> /dev/null; then
    echo "❌ Kaggle authentication failed. Please check your kaggle.json"
    exit 1
fi

echo "✅ Kaggle setup verified"

# Create datasets for competition files
echo "📊 Creating competition datasets..."

# Upload competition data
kaggle datasets create -p competitions/brain-simulation-benchmark-2025/data --title "brain-simulation-competition-data"

echo "✅ Competition data uploaded"

# Instructions for notebook upload
echo ""
echo "📓 Next steps for notebook upload:"
echo "1. Go to https://www.kaggle.com/code"
echo "2. Upload dna_consciousness_training.ipynb"
echo "3. Enable GPU and run training"
echo "4. Download results and integrate back"
echo ""
echo "🎉 Upload process complete!"
