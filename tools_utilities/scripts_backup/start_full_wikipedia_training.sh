#!/bin/bash

# Full Wikipedia Training Starter Script
# =====================================
# This script starts training on the COMPLETE English Wikipedia database

echo "🌍 Full Wikipedia Database Training Pipeline"
echo "============================================"
echo "This will train on EVERY English Wikipedia article!"
echo "Estimated time: 24-48 hours for complete training"
echo "Estimated storage: ~100GB for complete pipeline"
echo ""

# Check system resources
echo "🔍 Checking system resources..."

# Check available disk space
available_space=$(df . | awk 'NR==2 {print $4}')
available_gb=$((available_space / 1024 / 1024))
echo "   Available disk space: ${available_gb}GB"

if [ $available_gb -lt 100 ]; then
    echo "⚠️  Warning: Less than 100GB available. Training may fail."
    echo "   Consider freeing up space or using external storage."
fi

# Check available RAM
total_ram=$(free -g | awk 'NR==2 {print $2}')
echo "   Total RAM: ${total_ram}GB"

if [ $total_ram -lt 32 ]; then
    echo "⚠️  Warning: Less than 32GB RAM. Training may be slow or fail."
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        echo "   GPU: $line"
    done
else
    echo "⚠️  No NVIDIA GPU detected. Training will be much slower on CPU."
fi

echo ""

# Activate environment
echo "🔧 Activating Python environment..."
source wikipedia_env/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate environment. Please check setup."
    exit 1
fi

echo "✅ Environment activated"

# Check Python dependencies
echo "📦 Checking Python dependencies..."
python -c "import torch, transformers, datasets, tqdm, rich, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install torch transformers datasets tqdm rich psutil
fi

echo "✅ Dependencies verified"

# Create output directories
echo "📁 Creating output directories..."
mkdir -p wikipedia_dumps
mkdir -p full_wikipedia_training
mkdir -p logs

echo "✅ Directories created"

# Start training
echo ""
echo "🚀 Starting Full Wikipedia Training..."
echo "====================================="
echo "This will begin downloading and processing the complete Wikipedia database."
echo "The process will take 24-48 hours for complete training."
echo ""
echo "Training will start in 10 seconds..."
echo "Press Ctrl+C to cancel"

sleep 10

echo "🔥 Launching training pipeline..."
python scripts/full_wikipedia_training.py

echo ""
echo "🎉 Training pipeline completed!"
echo "Check the logs and output directories for results."
