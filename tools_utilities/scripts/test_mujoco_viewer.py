#!/usr/bin/env python3
"""
Minimal test to see if MuJoCo viewer can open a window.
This bypasses all the brain simulation complexity.
"""

import os
import sys
import time

def test_viewer():
    """Test if MuJoCo viewer can open with the humanoid model."""

    print("🧪 Testing MuJoCo Viewer...")

    # Check if MuJoCo can be imported
    try:
        import mujoco
        import mujoco.viewer
        print("✅ MuJoCo imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MuJoCo: {e}")
        return 1

    # Find the model file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(project_root, "brain/architecture/embodiment/humanoid.xml")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return 1

    print(f"✅ Model file found: {model_path}")

    # Try to load the model
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("✅ MuJoCo model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Try to launch the viewer
    try:
        print("🚀 Attempting to launch MuJoCo viewer...")
        print("   (A viewer window should appear now)")

        viewer = mujoco.viewer.launch_passive(model, data)
        print("✅ MuJoCo viewer launched!")

        # Keep it open for a few seconds
        print("   Keeping viewer open for 10 seconds...")
        for i in range(10):
            print(f"   {10-i} seconds remaining...")
            time.sleep(1)
            # Step the simulation
            mujoco.mj_step(model, data)
            viewer.sync()

        viewer.close()
        print("✅ Test completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Failed to launch viewer: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_viewer())
