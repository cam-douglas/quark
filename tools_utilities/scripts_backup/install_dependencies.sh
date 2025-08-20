#!/bin/bash
# Budget Wikipedia Training - Dependency Installation
# ==================================================

echo "🔧 Installing dependencies for budget Wikipedia training..."

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
    USE_VENV=true
else
    echo "⚠️  No virtual environment detected"
    echo "📝 Creating virtual environment..."
    python3 -m venv wikipedia_training_env
    source wikipedia_training_env/bin/activate
    USE_VENV=true
fi

# Install Python packages
echo "📦 Installing Python packages..."
pip install --upgrade pip

# Core ML packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install wandb

# Cloud packages
pip install boto3 google-cloud-storage azure-storage-blob
pip install kubernetes docker

# Utility packages
pip install tqdm psutil beautifulsoup4 lxml regex

echo "✅ Python packages installed!"

# Check system tools
echo "🛠️  Checking system tools..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop:"
    echo "   https://www.docker.com/products/docker-desktop/"
fi

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Installing..."
    # macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install kubectl
    # Linux
    else
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        chmod +x kubectl
        sudo mv kubectl /usr/local/bin/
    fi
fi

if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Installing..."
    # macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install awscli
    # Linux
    else
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
    fi
fi

echo "🎉 Installation complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Configure AWS: aws configure"
echo "2. Test Docker: docker --version"
echo "3. Run training: python scripts/simple_budget_launch.py --launch"
