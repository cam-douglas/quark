#!/usr/bin/env python3
"""
MuJoCo Simulation Client for Quark's Embodiment

This script loads the nursery environment, connects to the Brain-Body Interface
server via WebSocket, and runs the main simulation loop.

- It loads the MuJoCo model.
- Connects to the WebSocket server.
- In a loop:
  - Gathers sensory data from the simulation.
  - Sends sensory data to Quark's brain.
  - Receives motor commands from Quark's brain.
  - Applies motor commands to the agent in the simulation.
  - Renders the simulation for live viewing.
"""

import mujoco
import mujoco.viewer
import asyncio
import websockets
import json
import time
import os
import logging
from pathlib import Path
import sys
import numpy as np

# Add the project root to the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from brain_architecture.learning.balancing_agent import BalancingAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Find the absolute path to the XML file
xml_path = Path(__file__).parent / "humanoid_v1.xml"

class SimulationClient:
    def __init__(self, model_path):
        try:
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
            self.data = mujoco.MjData(self.model)
            self.viewer = None
        except Exception as e:
            logger.error(f"❌ Failed to load MuJoCo model: {e}")
            exit()
            
        self.uri = "ws://localhost:8000/ws/simulation"
        self.websocket = None
        
        # Initialize the learning agent
        # state_dim = self.model.nq + self.model.nv  # qpos (positions) + qvel (velocities)
        # action_dim = self.model.nu # number of actuators/controls
        # self.agent = BalancingAgent(state_dim=state_dim, action_dim=action_dim)
        
        self.episode_count = 0
        self.step_count = 0

    async def connect_to_server(self, max_retries=5, retry_delay=2):
        """Establish WebSocket connection to the brain interface with retries."""
        for attempt in range(max_retries):
            try:
                self.websocket = await websockets.connect(self.uri)
                logger.info(f"✅ Successfully connected to Brain-Body Interface at {self.uri}")
                return
            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt + 1 == max_retries:
                    logger.error("❌ Max connection retries reached. Could not connect to the server.")
                    logger.error("   Please ensure the embodiment_interface server is running.")
                    exit()
                await asyncio.sleep(retry_delay)

    def reset_simulation(self):
        """Resets the simulation to its initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.episode_count += 1
        self.step_count = 0
        logger.info(f"🔄 Simulation reset. Starting Episode #{self.episode_count}")

    def get_reward_and_done(self, sensory_data: dict) -> (float, bool):
        """
        Calculates a simplified reward signal based on the current state.
        """
        torso_z_position = self.data.qpos[2]
        
        # Survival bonus for every step
        reward = 1.0 

        # Check if the episode is done (humanoid has fallen)
        done = torso_z_position < 0.8
        
        return reward, done

    def apply_action(self, action: np.ndarray):
        """Applies the given action to the model's actuators."""
        if action is not None:
            self.data.ctrl[:] = action

    async def send_sensory_data(self, sensory_data: dict):
        """Sends sensory data to the server asynchronously."""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(sensory_data))
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Could not send sensory data: connection closed.")

    def get_sensory_data(self) -> dict:
        """
        Gathers and packages a robust sensory representation from the simulation.
        """
        # A more stable state representation using MuJoCo's sensor data if available,
        # otherwise, a combination of qpos and qvel.
        # This representation is often more stable for learning.
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        # Use cinert, which is mass-normalized and provides a stable representation
        # of body orientations and positions.
        cinert = self.data.cinert.flatten()
        
        state_vector = np.concatenate([position, velocity, cinert])

        return {
            "timestamp": time.time(),
            "state_vector": state_vector.tolist(),
        }

    def apply_motor_commands(self, motor_command: dict):
        """Applies motor commands received from Quark's brain to the actuators."""
        if "actuators" in motor_command and "controls" in motor_command["actuators"]:
            controls = motor_command["actuators"]["controls"]
            if len(controls) == self.model.nu:
                self.data.ctrl[:] = controls
            else:
                logger.warning(
                    f"Received motor command with {len(controls)} controls, "
                    f"but model expects {self.model.nu}."
                )

    async def run_simulation(self):
        """The main simulation and communication loop."""
        await self.connect_to_server()
        
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as e:
            logger.error(f"❌ Failed to launch MuJoCo viewer: {e}")
            logger.error("This might be a display issue. Ensure you have a valid display environment (e.g., an X server on Linux).")
            return

        self.reset_simulation()
        current_state = self.get_sensory_data()
        
        # Correctly initialize the agent *after* we know the state size
        state_dim = len(current_state['state_vector'])
        action_dim = self.model.nu
        self.agent = BalancingAgent(state_dim=state_dim, action_dim=action_dim)


        while self.viewer.is_running():
            self.step_count += 1
            
            # 1. Agent decides on an action based on the current state
            action = self.agent.get_action(current_state)
            
            # 2. Apply the action to the simulation
            self.apply_action(action)
            mujoco.mj_step(self.model, self.data)
            
            # 3. Get the new state and the reward
            next_state = self.get_sensory_data()
            reward, done = self.get_reward_and_done(next_state)

            # 4. Store the experience in the agent's memory
            self.agent.store_experience(current_state, action, reward, next_state, done)
            
            # 5. Let the agent learn from its experiences
            self.agent.learn()
            
            # Send sensory data to the cognitive core for monitoring
            asyncio.create_task(self.send_sensory_data(next_state))

            # Update the current state
            current_state = next_state
            
            # Log reward for observation
            if self.step_count % 50 == 0:
                logger.info(f"Episode: {self.episode_count}, Step: {self.step_count}, Reward: {reward:.4f}")

            # 6. If the episode is over, reset the simulation
            if done:
                logger.info(f"💥 Episode {self.episode_count} finished after {self.step_count} steps.")
                self.reset_simulation()
                current_state = self.get_sensory_data()

            self.viewer.sync()

        self.viewer.close()
        logger.info("Simulation finished.")
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed.")

if __name__ == "__main__":
    if not xml_path.exists():
        logger.error(f"❌ Critical error: humanoid_v1.xml not found at {xml_path}")
        logger.error("Please ensure the XML file is in the same directory as this script.")
    else:
        client = SimulationClient(xml_path)
        try:
            asyncio.run(client.run_simulation())
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
