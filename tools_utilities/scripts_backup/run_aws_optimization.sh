set -euo pipefail

#!/bin/bash

# 🚀 AWS Performance Optimization Launcher
# This script sets up and runs the AWS optimization tool

echo "🚀 AWS Performance Optimization Tool"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements_aws_optimization.txt

# Check AWS credentials
echo "🔐 Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    echo "   Or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION environment variables."
    exit 1
fi

# Run the optimization tool
echo "🚀 Starting AWS Performance Optimization..."
python3 aws_performance_optimization.py

echo "✅ Optimization complete! Check aws_optimization_results.json for results."
