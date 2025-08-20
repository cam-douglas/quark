#!/usr/bin/env python3
"""
Simple launcher for the Exponential Learning System
Handles Python path issues and provides easy startup
"""

import os, sys
import subprocess
from pathlib import Path

def main():
    """Launch the exponential learning system"""
    print("🚀 Launching Exponential Learning System...")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the script directory
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Add current directory to Python path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    # Check if required files exist
    required_files = [
        "exponential_learning_system.py",
        "research_agents.py", 
        "knowledge_synthesizer.py",
        "cloud_training_orchestrator.py",
        "knowledge_validation_system.py",
        "neuro_agent_enhancer.py",
        "main_orchestrator.py",
        "run_exponential_learning.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all components are present in the current directory.")
        return 1
    
    print("✅ All required files found")
    
    # Check if requirements are installed
    try:
        import aiohttp
        import yaml
        print("✅ Dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return 1
    
    # Launch the system
    print("🚀 Starting Exponential Learning System...")
    try:
        # Import and run the main function
        from run_exponential_learning import main as run_main
        import asyncio
        
        # Run the async main function
        asyncio.run(run_main())
        
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        print("\n🔧 Trying alternative startup method...")
        
        # Alternative: run as subprocess
        try:
            result = subprocess.run([sys.executable, "run_exponential_learning.py"], 
                                 cwd=script_dir, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"❌ Alternative startup failed: {e}")
            return 1
        except FileNotFoundError:
            print("❌ Could not find Python executable")
            return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
