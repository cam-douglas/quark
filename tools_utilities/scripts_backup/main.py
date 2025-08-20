#!/usr/bin/env python3
"""
main.py — Entry point for Quark Brain Simulation

This is the main entry point for running the brain simulation.
It imports the core modules and provides a clean interface.
"""

import sys
import os

# Add brain modules to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brain_modules'))

try:
    from brain_modules.conscious_agent.main.brain_launcher_v3 import main as run_brain_simulation
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def run_brain_simulation_wrapper():
    """Wrapper function for console script entry point"""
    return run_brain_simulation()

if __name__ == "__main__":
    run_brain_simulation()
