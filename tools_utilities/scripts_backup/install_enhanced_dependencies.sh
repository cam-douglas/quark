set -euo pipefail

#!/bin/bash

# Enhanced Small-Mind Dependencies Installation Script
# This script installs all the enhanced GitHub repository integrations

echo "🚀 Installing Enhanced Small-Mind Dependencies"
echo "================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core scientific computing
echo "🧮 Installing core scientific computing packages..."
pip install numpy scipy matplotlib

# Install neural simulation and machine learning
echo "🧠 Installing neural simulation and ML packages..."
pip install torch scikit-learn transformers huggingface-hub

# Install physics simulation
echo "⚡ Installing physics simulation packages..."
pip install pybullet pymunk

# Install data handling and visualization
echo "📊 Installing data handling and visualization packages..."
pip install pandas seaborn plotly

# Install PyVista and VTK (3D visualization)
echo "🎨 Installing PyVista 3D visualization..."
pip install pyvista vtk

# Install machine learning optimization
echo "🔍 Installing Optuna hyperparameter optimization..."
pip install optuna

# Install large-scale data processing
echo "📈 Installing large-scale data processing packages..."
pip install dask

# Install interactive visualization
echo "🖱️  Installing interactive visualization packages..."
pip install holoviews bokeh

# Install web framework
echo "🌐 Installing web framework packages..."
pip install streamlit fastapi uvicorn

# Install development tools
echo "🛠️  Installing development tools..."
pip install pytest black

# Install additional utilities
echo "🔧 Installing additional utilities..."
pip install tqdm requests aiohttp

echo ""
echo "✅ Enhanced dependencies installation completed!"
echo ""
echo "📋 Installed packages:"
echo "   - NEST Simulator (mock interface)"
echo "   - Optuna (hyperparameter optimization)"
echo "   - PyVista (3D visualization)"
echo "   - Dask (parallel computing)"
echo "   - Holoviews & Bokeh (interactive viz)"
echo "   - FastAPI & Streamlit (web frameworks)"
echo ""
echo "🧪 Test the integration:"
echo "   python enhanced_integration_demo.py"
echo ""
echo "📚 Read the documentation:"
echo "   ENHANCED_INTEGRATION_README.md"
echo ""
echo "⚠️  Note: NEST Simulator requires system-level installation:"
echo "   macOS: brew install nest"
echo "   Ubuntu: sudo apt-get install nest"
echo "   Or build from source: https://github.com/nest/nest-simulator"
