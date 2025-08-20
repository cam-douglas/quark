#!/usr/bin/env python3
"""
Debug longer simulation to see neural activity buildup
"""

import sys
import os
import tempfile
# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'src'))

from core.brain_launcher_v3 import Brain

def debug_long_simulation():
    """Debug longer simulation to see neural activity buildup"""
    print("🧠 Debugging Long Simulation...")
    
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
        
        # Run longer simulation
        for step in range(100):
            telemetry = brain.step(50)
            
            # Check every 10th step
            if step % 10 == 0:
                print(f"\n--- Step {step} ---")
                pfc_tel = telemetry.get("pfc", {})
                wm_tel = telemetry.get("working_memory", {})
                
                print(f"PFC firing rate: {pfc_tel.get('firing_rate', 0):.2f} Hz")
                print(f"PFC spike count: {pfc_tel.get('spike_count', 0)}")
                print(f"WM firing rate: {wm_tel.get('firing_rate', 0):.2f} Hz")
                print(f"WM spike count: {wm_tel.get('spike_count', 0)}")
                
                # Check if we have any spikes
                if pfc_tel.get('spike_count', 0) > 0 or wm_tel.get('spike_count', 0) > 0:
                    print("🎉 SPIKES DETECTED!")
                    break
        
        # Final summary
        print(f"\n--- Final Summary ---")
        pfc_tel = telemetry.get("pfc", {})
        wm_tel = telemetry.get("working_memory", {})
        
        print(f"Final PFC firing rate: {pfc_tel.get('firing_rate', 0):.2f} Hz")
        print(f"Final PFC spike count: {pfc_tel.get('spike_count', 0)}")
        print(f"Final WM firing rate: {wm_tel.get('firing_rate', 0):.2f} Hz")
        print(f"Final WM spike count: {wm_tel.get('spike_count', 0)}")
        
    finally:
        if os.path.exists(temp_metrics_file):
            os.unlink(temp_metrics_file)

if __name__ == "__main__":
    debug_long_simulation()
