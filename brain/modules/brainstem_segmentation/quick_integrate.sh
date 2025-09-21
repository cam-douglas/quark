#!/bin/bash
# Quick integration script for Colab model

echo "============================================================"
echo "🔄 Quick Model Integration Script"
echo "============================================================"

# Common download locations
DOWNLOAD_PATHS=(
    "$HOME/Downloads/best_model_colab.pth"
    "$HOME/Desktop/best_model_colab.pth"
    "$HOME/Documents/best_model_colab.pth"
    "./best_model_colab.pth"
)

MODEL_FOUND=""

# Search for the model
echo "🔍 Searching for model file..."
for path in "${DOWNLOAD_PATHS[@]}"; do
    if [ -f "$path" ]; then
        MODEL_FOUND="$path"
        echo "✅ Found model at: $path"
        break
    fi
done

if [ -z "$MODEL_FOUND" ]; then
    echo "❌ Model not found in common locations"
    echo ""
    echo "Please specify the path to best_model_colab.pth:"
    echo "Usage: ./quick_integrate.sh /path/to/best_model_colab.pth"
    exit 1
fi

# Use provided path if given
if [ $# -eq 1 ]; then
    MODEL_FOUND="$1"
    echo "📁 Using provided path: $MODEL_FOUND"
fi

# Copy to expected location
TARGET_DIR="/Users/camdouglas/quark/brain/modules/brainstem_segmentation"
echo "📦 Copying model to integration directory..."
cp "$MODEL_FOUND" "$TARGET_DIR/best_model_colab.pth"

# Also check for training results
RESULTS_PATH="${MODEL_FOUND%.*}.json"
if [ -f "$RESULTS_PATH" ]; then
    echo "📊 Found training results, copying..."
    cp "$RESULTS_PATH" "$TARGET_DIR/training_results.json"
fi

# Alternative results path
RESULTS_PATH="${MODEL_FOUND%best_model_colab.pth}training_results.json"
if [ -f "$RESULTS_PATH" ]; then
    echo "📊 Found training results, copying..."
    cp "$RESULTS_PATH" "$TARGET_DIR/training_results.json"
fi

# Run integration
echo ""
echo "🚀 Running integration script..."
cd /Users/camdouglas/quark
python brain/modules/brainstem_segmentation/integrate_colab_model.py

echo ""
echo "============================================================"
echo "✅ Integration complete!"
echo "============================================================"

