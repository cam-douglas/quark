#!/usr/bin/env python3
"""
Test Client for the Brain-Body Interface

This script acts as a simple simulation client to test the WebSocket
connection with the embodiment_interface server. It sends mock sensory
data and prints the motor commands it receives back.
"""

import asyncio
import websockets
import json

async def test_brain_body_interface():
    """Connects to the server and performs a simple communication test."""
    uri = "ws://localhost:8000/ws/simulation"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Successfully connected to {uri}")

            # --- Test 1: Send mock sensory data ---
            print("\n--- Sending mock sensory data ---")
            mock_sensory_data = {
                "timestamp": 123456789.0,
                "vision": [0.1, 0.2, 0.3], # Simplified representation
                "touch": {"left_hand": True, "right_hand": False},
                "proprioception": {"arm_angle": 45.0}
            }
            await websocket.send(json.dumps(mock_sensory_data))
            print(f"> Sent: {json.dumps(mock_sensory_data)}")

            # --- Wait for motor command response ---
            response = await websocket.recv()
            print(f"< Received: {response}")
            print("----------------------------------")

            print("\n✅ Test complete. The interface is working.")

    except ConnectionRefusedError:
        print(f"❌ Connection failed. Is the server running?")
        print("   Run: uvicorn brain_architecture.embodiment.embodiment_interface:app --reload")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_brain_body_interface())
