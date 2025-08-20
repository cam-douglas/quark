#!/usr/bin/env python3
"""
Quick Start Script for Quark Brain Simulation Notebooks

This script helps you quickly set up and launch Jupyter notebooks for brain simulation work.

Usage:
    python notebooks/quick_start.py [--type=training|simulation|analysis] [--port=8888]
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def check_virtual_environment():
    """Check if virtual environment is activated"""
    if sys.prefix == sys.base_prefix:
        print("❌ Virtual environment not activated!")
        print("Please run: source venv/bin/activate")
        return False
    print("✅ Virtual environment active")
    return True

def check_jupyter_installed():
    """Check if Jupyter is installed"""
    try:
        import jupyter
        import jupyterlab
        print("✅ Jupyter and JupyterLab are installed")
        return True
    except ImportError:
        print("❌ Jupyter not installed!")
        print("Installing Jupyter...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "notebook", "jupyterlab"], 
                         check=True, capture_output=True)
            print("✅ Jupyter installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Jupyter")
            return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['numpy', 'matplotlib', 'seaborn', 'pandas', 'scikit-learn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, 
                         check=True, capture_output=True)
            print("✅ All dependencies installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    else:
        print("✅ All dependencies are installed")
        return True

def get_notebook_recommendation(notebook_type=None):
    """Get notebook recommendation based on user needs"""
    
    notebooks = {
        'training': {
            'path': 'training/interactive_model_training.ipynb',
            'description': 'Train and experiment with brain components',
            'use_case': 'Developing and training new neural components'
        },
        'simulation': {
            'path': 'simulation/brain_component_simulation.ipynb', 
            'description': 'Real-time brain simulation and testing',
            'use_case': 'Testing component interactions and brain dynamics'
        },
        'analysis': {
            'path': 'analysis/data_exploration_analysis.ipynb',
            'description': 'Analyze simulation data and training results', 
            'use_case': 'Exploring data patterns and generating insights'
        }
    }
    
    if notebook_type and notebook_type in notebooks:
        return notebooks[notebook_type]
    
    print("\n🎯 What would you like to do?")
    print("1. Train brain components (training)")
    print("2. Run brain simulations (simulation)")  
    print("3. Analyze data (analysis)")
    
    while True:
        choice = input("\nEnter your choice (1-3) or notebook type: ").strip().lower()
        
        if choice in ['1', 'training']:
            return notebooks['training']
        elif choice in ['2', 'simulation']:
            return notebooks['simulation']
        elif choice in ['3', 'analysis']:
            return notebooks['analysis']
        else:
            print("Invalid choice. Please enter 1-3 or training/simulation/analysis")

def launch_jupyter(notebook_path=None, port=8888):
    """Launch Jupyter Lab with optional specific notebook"""
    
    # Change to notebooks directory
    notebooks_dir = Path(__file__).parent
    os.chdir(notebooks_dir)
    
    print(f"🚀 Launching Jupyter Lab on port {port}...")
    print(f"📁 Working directory: {notebooks_dir}")
    
    if notebook_path:
        print(f"📓 Opening: {notebook_path}")
        
    # Build Jupyter Lab command
    cmd = [sys.executable, "-m", "jupyterlab", f"--port={port}", "--no-browser"]
    
    try:
        print("\n" + "="*60)
        print("🧠 QUARK BRAIN SIMULATION NOTEBOOKS")
        print("="*60)
        print(f"🌐 Jupyter Lab URL: http://localhost:{port}")
        print("💡 Tips:")
        print("  - Press Ctrl+C to stop Jupyter")
        print("  - Use consciousness_bridge to connect to main brain simulation")
        print("  - Check README.md for detailed usage instructions")
        print("="*60)
        
        # Launch Jupyter Lab
        process = subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Jupyter Lab stopped")
    except Exception as e:
        print(f"❌ Error launching Jupyter Lab: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Quick start script for brain simulation notebooks")
    parser.add_argument('--type', choices=['training', 'simulation', 'analysis'],
                       help='Type of notebook to work with')
    parser.add_argument('--port', type=int, default=8888,
                       help='Port for Jupyter Lab (default: 8888)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check dependencies, don\'t launch Jupyter')
    
    args = parser.parse_args()
    
    print("🧠 Quark Brain Simulation Notebooks - Quick Start")
    print("="*50)
    
    # Check prerequisites
    if not check_virtual_environment():
        return 1
        
    if not check_jupyter_installed():
        return 1
        
    if not check_dependencies():
        return 1
    
    if args.check_only:
        print("\n✅ All checks passed! Ready to launch notebooks.")
        return 0
    
    # Get notebook recommendation
    recommended = get_notebook_recommendation(args.type)
    
    print(f"\n📓 Recommended notebook: {recommended['path']}")
    print(f"📋 Description: {recommended['description']}")
    print(f"🎯 Use case: {recommended['use_case']}")
    
    # Ask user if they want to proceed
    response = input(f"\nLaunch Jupyter Lab? (Y/n): ").strip().lower()
    if response in ['', 'y', 'yes']:
        launch_jupyter(recommended['path'], args.port)
    else:
        print("👋 Exiting without launching Jupyter")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
