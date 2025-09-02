#!/usr/bin/env python3
"""
Simple MuJoCo viewer that stays open until you close it.
Press Ctrl+C in the terminal to close the viewer.
"""

import os
import sys
import time

def main():
    """Launch MuJoCo viewer and keep it open."""
    
    print("🚀 Launching MuJoCo Viewer (staying open until Ctrl+C)...")
    
    try:
        import mujoco
        import mujoco.viewer
        
        # Find the model file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_path = os.path.join(project_root, "brain/architecture/embodiment/humanoid.xml")
        
        # Load model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print("✅ Model loaded successfully!")
        print("🖥️  MuJoCo viewer window should appear now...")
        print("📌 Press Ctrl+C in this terminal to close the viewer")
        
        # Launch viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        
        # Keep running until interrupted
        try:
            while True:
                # Step the simulation
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(1.0 / 60)  # 60 FPS
        except KeyboardInterrupt:
            print("\n👋 Closing viewer...")
            viewer.close()
            print("✅ Viewer closed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
