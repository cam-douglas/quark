#!/usr/bin/env python3
"""
Debug brain telemetry to understand what's being returned
"""

import sys
import os
import tempfile
# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'src'))

from core.brain_launcher_v3 import Brain

def debug_brain_telemetry():
    """Debug brain telemetry output"""
    print("🧠 Debugging Brain Telemetry...")
    
    brain_config = {
        "modules": {
            "architecture_agent": {"type": "architecture_agent"},
            "pfc": {"type": "pfc", "num_neurons": 50},
            "basal_ganglia": {"type": "basal_ganglia"},
            "thalamus": {"type": "thalamus"},
            "working_memory": {"type": "working_memory", "slots": 3, "num_neurons": 30},
            "dmn": {"type": "dmn"},
            "salience": {"type": "salience"},
            "attention": {"type": "attention"}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_metrics_file = f.name
    
    try:
        brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
        
        # Run a few steps and check telemetry
        for step in range(5):
            print(f"\n--- Step {step} ---")
            telemetry = brain.step(50)
            
            # Check what modules are in telemetry
            print(f"Telemetry keys: {list(telemetry.keys())}")
            
            # Check PFC telemetry specifically
            if "pfc" in telemetry:
                pfc_tel = telemetry["pfc"]
                print(f"PFC telemetry keys: {list(pfc_tel.keys())}")
                print(f"PFC telemetry: {pfc_tel}")
            else:
                print("No PFC in telemetry")
            
            # Check WM telemetry specifically
            if "working_memory" in telemetry:
                wm_tel = telemetry["working_memory"]
                print(f"WM telemetry keys: {list(wm_tel.keys())}")
                print(f"WM telemetry: {wm_tel}")
            else:
                print("No WM in telemetry")
        
    finally:
        if os.path.exists(temp_metrics_file):
            os.unlink(temp_metrics_file)

if __name__ == "__main__":
    debug_brain_telemetry()
