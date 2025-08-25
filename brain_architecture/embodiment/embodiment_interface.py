#!/usr/bin/env python3
"""
Brain-Body Interface for Quark's Embodied Cognition

This module serves as the central nervous system for Quark's physical embodiment.
It uses a FastAPI server with WebSockets to create a real-time, bidirectional
communication channel between Quark's core cognitive architecture (Python) and
the simulated physical environment (MuJoCo, visualized in a browser).

- It receives sensory data from the simulation (e.g., vision, touch).
- It sends motor commands from Quark's brain to the simulated body.
"""

import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import json
import time
from brain_architecture.embodiment.cognitive_core import EmbodiedCognitiveCore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New connection: {websocket.client}. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Connection closed: {websocket.client}. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Sends a message to all connected clients."""
        for connection in self.active_connections:
            await connection.send_text(message)

# --- FastAPI and WebSocket Setup ---
app = FastAPI()
manager = ConnectionManager()
# Create a single instance of the cognitive core to be shared by all connections
cognitive_core = EmbodiedCognitiveCore()


@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    """
    The main WebSocket endpoint for the brain-body interface.
    Handles communication with the simulation client.
    """
    await manager.connect(websocket)
    try:
        while True:
            sensory_data_str = await websocket.receive_text()
            # Pass sensory data to the cognitive core for processing and safety checks
            cognitive_core.process_sensory_data(sensory_data_str)
            
            # Generate the next motor command from the cognitive core
            motor_command_json = cognitive_core.generate_motor_command()
            
            await websocket.send_text(motor_command_json)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket endpoint: {e}")
        manager.disconnect(websocket)

@app.get("/")
def read_root():
    return {"message": "Quark Brain-Body Interface is running."}

# To run this server:
# source venv/bin/activate
# uvicorn brain_architecture.embodiment.embodiment_interface:app --reload
