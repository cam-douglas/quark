#!/usr/bin/env python3
"""
MuJoCo Simulation Client for Quark's Embodiment

This client connects to the Brain-Body Interface and runs the simulation
with real-time motor control from Quark's cognitive core.
"""

import asyncio
import websockets
import json
import logging
import argparse
import time
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import mujoco
    from mujoco import MjData, MjModel
    import mujoco.viewer
except ImportError as e:
    logging.error(f"❌ Failed to import MuJoCo: {e}")
    logging.error("Please install MuJoCo: pip install mujoco")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationClient:
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.websocket = None
        self.simulation_running = False
        
        # Load the simpler balance model
        try:
            model_path = os.path.join(os.path.dirname(__file__), "simple_balance_locked_arms.xml")
            self.model = MjModel.from_xml_path(model_path)
            self.data = MjData(self.model)
            logger.info(f"✅ Loaded model: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            sys.exit(1)
        
        # Initialize viewer
        self.viewer = None
        self.step_count = 0
        
    async def connect_to_brain(self):
        """Connect to the Brain-Body Interface via WebSocket."""
        try:
            uri = f"ws://{self.host}:{self.port}/ws/simulation"
            self.websocket = await websockets.connect(uri)
            logger.info(f"✅ Successfully connected to Brain-Body Interface at {uri}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Brain-Body Interface: {e}")
            return False
    
    def setup_viewer(self):
        """Setup the MuJoCo viewer."""
        try:
            # Use the basic MuJoCo viewer that should work on macOS
            from mujoco import MjViewer
            self.viewer = MjViewer(self.model)
            logger.info("✅ MuJoCo viewer launched successfully with MjViewer")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to launch MuJoCo viewer: {e}")
            logger.error("This might be a display issue. Ensure you have a valid display environment.")
            return False
    
    async def send_sensory_data(self):
        """Send current simulation state as sensory data to the brain."""
        if self.websocket is None:
            return
        
        try:
            # Extract key state information for the simple balance model
            # qpos: [base_x, base_y, base_z, base_rot_x, base_rot_y, base_rot_z, pendulum_angle, counterweight_angle]
            # qvel: [base_vx, base_vy, base_vz, base_rot_vx, base_rot_vy, base_rot_vz, pendulum_vel, counterweight_vel]
            
            sensory_data = {
                "timestamp": time.time(),
                "state_vector": self.data.qpos.tolist() + self.data.qvel.tolist(),
                "model_info": {
                    "num_joints": self.model.njnt,
                    "num_actuators": self.model.nu,
                    "num_bodies": self.model.nbody
                }
            }
            
            await self.websocket.send(json.dumps(sensory_data))
            
        except Exception as e:
            logger.error(f"Error sending sensory data: {e}")
    
    async def receive_motor_command(self):
        """Receive motor commands from the brain."""
        if self.websocket is None:
            return None
        
        try:
            # Wait for motor command with timeout
            message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
            motor_data = json.loads(message)
            
            if "actuators" in motor_data and "controls" in motor_data["actuators"]:
                return motor_data["actuators"]["controls"]
            else:
                logger.warning(f"Invalid motor command format: {motor_data}")
                return None
                
        except asyncio.TimeoutError:
            # No command received, use zero command
            return None
        except Exception as e:
            logger.error(f"Error receiving motor command: {e}")
            return None
    
    def apply_motor_commands(self, controls):
        """Apply motor commands to the simulation."""
        if controls is None or len(controls) != self.model.nu:
            # Use zero commands if invalid
            self.data.ctrl[:] = 0.0
        else:
            # Apply the received controls
            for i, control in enumerate(controls):
                if i < self.model.nu:
                    self.data.ctrl[i] = control
    
    async def run_simulation(self):
        """Run the main simulation loop."""
        logger.info("🔄 Simulation reset. Starting Episode #1")
        episode = 1
        episode_steps = 0
        
        # Ensure viewer is properly set up
        if not self.viewer:
            logger.warning("No viewer available, running headless simulation")
        
        while True:
            try:
                # Send current sensory data to brain
                await self.send_sensory_data()
                
                # Receive motor commands from brain
                controls = await self.receive_motor_command()
                
                # Apply motor commands
                self.apply_motor_commands(controls)
                
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer if available
                if self.viewer:
                    try:
                        self.viewer.sync()
                    except Exception as e:
                        logger.warning(f"Viewer sync error: {e}")
                
                self.step_count += 1
                episode_steps += 1
                
                # Check for episode end (falling or timeout)
                base_height = self.data.qpos[2]  # base z-position
                pendulum_angle = abs(self.data.qpos[6])  # pendulum angle
                
                if base_height < 0.1 or pendulum_angle > 0.8:  # Fallen or too tilted
                    logger.info(f"💥 Episode {episode} finished after {episode_steps} steps.")
                    episode += 1
                    episode_steps = 0
                    
                    # Reset simulation for next episode
                    mujoco.mj_resetData(self.model, self.data)
                    logger.info(f"🔄 Simulation reset. Starting Episode #{episode}")
                    
                    # Limit episodes for demo
                    if episode > 20:
                        logger.info("Simulation finished.")
                        break
                
                # Small delay to make simulation visible and prevent overwhelming the viewer
                await asyncio.sleep(0.02)
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                break
    
    async def run(self):
        """Main run method."""
        # Connect to brain
        if not await self.connect_to_brain():
            return
        
        # Setup viewer
        if not self.setup_viewer():
            logger.warning("Continuing without viewer...")
        
        try:
            await self.run_simulation()
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            if self.websocket:
                await self.websocket.close()

async def main():
    parser = argparse.ArgumentParser(description="MuJoCo Simulation Client")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    args = parser.parse_args()
    
    client = SimulationClient(args.host, args.port)
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
