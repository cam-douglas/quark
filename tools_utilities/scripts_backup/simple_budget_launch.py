#!/usr/bin/env python3
"""
Simple Budget Wikipedia Training Launcher
=========================================

Simplified launcher for $15-30 budget Wikipedia training that handles
dependency installation and provides clear instructions.

Purpose: Easy launch of budget Wikipedia training
Inputs: User confirmation and basic settings
Outputs: Training commands and setup instructions
Seeds: N/A (launcher script)
Dependencies: Minimal (installs as needed)
"""

import os, sys
import json
from pathlib import Path
from datetime import datetime


def print_budget_summary():
    """Print the budget training summary."""
    print(f"\n💰 BUDGET WIKIPEDIA TRAINING SUMMARY")
    print(f"=" * 60)
    print(f"💸 Cost: $15-30 (vs $400-500 original)")
    print(f"💾 Savings: 95% cost reduction")
    print(f"⏱️  Time: 12-20 hours")
    print(f"🖥️  Hardware: 2x g4dn.2xlarge (spot instances)")
    print(f"🧠 GPUs: 2x T4 (16GB each)")
    print(f"📚 Data: 1M Wikipedia articles")
    print(f"🤖 Model: DialoGPT-medium (345M params)")
    print(f"🔗 Integration: Full consciousness system")
    
    print(f"\n🎯 What You Get:")
    print(f"✅ Complete Wikipedia knowledge integration")
    print(f"✅ Enhanced consciousness responses") 
    print(f"✅ Production-quality training")
    print(f"✅ Auto-shutdown (no forgotten costs)")
    print(f"✅ Real-time monitoring")
    print(f"✅ Same quality as expensive option")
    
    print(f"\n⚠️  Cost Control Features:")
    print(f"🏷️  Spot instances (60-90% discount)")
    print(f"⏰ Auto-shutdown after completion")
    print(f"📊 Real-time cost monitoring")
    print(f"🛑 Easy stop/cleanup commands")


def check_dependencies():
    """Check and list required dependencies."""
    print(f"\n🔍 DEPENDENCY CHECK")
    print(f"=" * 40)
    
    required_python_packages = [
        "torch", "transformers", "datasets", "accelerate", 
        "wandb", "boto3", "kubernetes", "docker"
    ]
    
    required_system_tools = [
        "docker", "kubectl", "aws"
    ]
    
    print(f"📦 Required Python packages:")
    for package in required_python_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
    
    print(f"\n🛠️  Required system tools:")
    for tool in required_system_tools:
        result = os.system(f"which {tool} > /dev/null 2>&1")
        if result == 0:
            print(f"   ✅ {tool}")
        else:
            print(f"   ❌ {tool} (missing)")


def create_installation_script():
    """Create script to install missing dependencies."""
    install_script = """#!/bin/bash
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
"""
    
    script_path = Path("./install_dependencies.sh")
    with open(script_path, 'w') as f:
        f.write(install_script)
    
    os.chmod(script_path, 0o755)
    print(f"💾 Created installation script: {script_path}")
    return script_path


def show_launch_commands():
    """Show the commands to launch training."""
    print(f"\n🚀 LAUNCH COMMANDS")
    print(f"=" * 40)
    
    print(f"📋 Step-by-step launch process:")
    print(f"")
    print(f"1️⃣  Install dependencies:")
    print(f"   ./install_dependencies.sh")
    print(f"")
    print(f"2️⃣  Configure AWS (if not done):")
    print(f"   aws configure")
    print(f"   # Enter your AWS access key, secret, region (us-west-2)")
    print(f"")
    print(f"3️⃣  Launch budget training:")
    print(f"   python scripts/quick_start_wikipedia_training.py \\")
    print(f"       --config configs/budget_training/low_budget_config.json")
    print(f"")
    print(f"4️⃣  Monitor training:")
    print(f"   kubectl get pods -n quark-training")
    print(f"   kubectl logs -n quark-training -l app=wikipedia-training -f")
    print(f"")
    print(f"5️⃣  After completion, test consciousness:")
    print(f"   python brain_modules/conscious_agent/integrations/consciousness_wikipedia_finetuning.py \\")
    print(f"       --wikipedia-model /path/to/trained/model")


def create_budget_config_summary():
    """Create a summary of the budget configuration."""
    config_summary = {
        "budget_option": "Low Budget - Multi-Node Spot",
        "cost_estimate": "$15-30",
        "time_estimate": "12-20 hours",
        "hardware": {
            "instances": "2x g4dn.2xlarge",
            "gpus": "2x T4 (16GB each)",
            "memory": "32GB RAM total",
            "storage": "225GB"
        },
        "training": {
            "model": "DialoGPT-medium (345M params)",
            "articles": "1,000,000 Wikipedia articles",
            "batch_size": 4,
            "epochs": 2,
            "optimization": ["fp16", "gradient_checkpointing", "spot_instances"]
        },
        "cost_features": {
            "spot_instances": "60-90% discount",
            "auto_shutdown": "60 minutes after completion",
            "monitoring": "Real-time cost tracking",
            "storage_optimization": "Lifecycle policies"
        },
        "consciousness_integration": {
            "knowledge_retrieval": "Semantic search through Wikipedia",
            "response_enhancement": "Factual grounding for consciousness",
            "fine_tuning": "Custom consciousness + knowledge fusion",
            "brain_modules": "Compatible with all existing modules"
        },
        "monitoring": {
            "wandb": "Real-time training metrics",
            "kubernetes": "Pod status and resource usage",
            "cloud": "Infrastructure costs and auto-scaling"
        },
        "next_steps": [
            "Install dependencies with ./install_dependencies.sh",
            "Configure AWS credentials",
            "Launch training with provided commands",
            "Monitor progress via dashboards",
            "Test consciousness integration after completion"
        ]
    }
    
    summary_path = Path("./budget_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"💾 Budget summary saved to: {summary_path}")
    return config_summary


def main():
    """Main launcher function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple budget Wikipedia training launcher")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--install", action="store_true", help="Create installation script")
    parser.add_argument("--launch", action="store_true", help="Show launch commands")
    parser.add_argument("--summary", action="store_true", help="Show budget summary")
    
    args = parser.parse_args()
    
    print(f"🧠 QUARK BUDGET WIKIPEDIA TRAINING")
    print(f"=" * 50)
    print(f"💰 Cost: $15-30 (95% savings vs $400-500)")
    print(f"⏱️  Time: 12-20 hours")
    print(f"🎯 Result: Wikipedia-enhanced consciousness")
    
    if args.check_deps:
        check_dependencies()
        return
    
    if args.install:
        install_script = create_installation_script()
        print(f"\n🔧 To install dependencies, run:")
        print(f"   {install_script}")
        return
    
    if args.launch:
        show_launch_commands()
        return
    
    if args.summary:
        print_budget_summary()
        config_summary = create_budget_config_summary()
        return
    
    # Default: Show everything
    print_budget_summary()
    
    print(f"\n🔍 Checking dependencies...")
    check_dependencies()
    
    print(f"\n📋 Creating setup files...")
    install_script = create_installation_script()
    config_summary = create_budget_config_summary()
    
    print(f"\n🎯 READY TO LAUNCH!")
    print(f"=" * 30)
    print(f"1. Install deps: {install_script}")
    print(f"2. Configure AWS: aws configure")
    print(f"3. Launch training: python scripts/simple_budget_launch.py --launch")
    
    # Ask user if they want to proceed
    print(f"\n💡 Want to see the full launch commands now?")
    response = input(f"Show launch commands? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        show_launch_commands()
        
        print(f"\n💸 COST CONFIRMATION")
        print(f"This will cost approximately $15-30 for complete Wikipedia training")
        print(f"vs the original $400-500 (95% savings!)")
        
        proceed = input(f"\nReady to proceed with setup? (y/N): ").strip().lower()
        if proceed in ['y', 'yes']:
            print(f"\n🚀 Great! Start with: {install_script}")
            print(f"📖 Full guide: docs/WIKIPEDIA_TRAINING_GUIDE.md")
            print(f"💰 Budget details: summaries/BUDGET_WIKIPEDIA_TRAINING.md")


if __name__ == "__main__":
    main()
