import mujoco
import mujoco.viewer
import time
import os

# Create a directory to store the videos
video_dir = "testing/simulations/videos"
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, "mujoco_test_simulation.mp4")

xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size=".1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Use the new mujoco.viewer context manager
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation running... Press Ctrl+C to exit and save video.")
        
        start_time = time.time()
        
        # Simulate for 5 seconds
        while time.time() - start_time < 5:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
        print(f"Simulation finished. Video will be saved to: {video_path}")
        # The viewer will handle saving the video on exit
        
except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except Exception as e:
    print(f"An error occurred during simulation: {e}")

print("Test complete. Check the video file to verify the simulation.")
