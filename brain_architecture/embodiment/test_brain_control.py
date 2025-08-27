#!/usr/bin/env python3
"""
Test script for brain-to-body control system.

This script tests the cognitive core's ability to generate motor commands
from sensory data, simulating the brain's control of the body.
"""

import sys
import json
from pathlib import Path

# Add the project root to the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from brain_architecture.embodiment.cognitive_core import EmbodiedCognitiveCore

def test_brain_control():
    """Test the brain-to-body control system."""
    print("🧠 Testing Brain-to-Body Control System")
    print("=" * 50)
    
    # Initialize the cognitive core
    print("1. Initializing Embodied Cognitive Core...")
    cognitive_core = EmbodiedCognitiveCore()
    print("✅ Cognitive core initialized")
    
    # Test with sample sensory data
    print("\n2. Testing motor command generation...")
    
    # Sample sensory data (simulating humanoid state)
    sample_sensory_data = {
        "timestamp": 1234567890.123,
        "state_vector": [
            0.0, 0.0, 1.0,  # qpos: x, y, z (torso at height 1.0)
            0.0, 0.0, 0.0,  # qpos: roll, pitch, yaw
            0.0, 0.0, 0.0,  # qvel: linear velocities
            0.0, 0.0, 0.0,  # qvel: angular velocities
            1.0, 0.0, 0.0,  # cinert: body orientations
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
    }
    
    # Convert to JSON string (as it would be received from WebSocket)
    sensory_data_str = json.dumps(sample_sensory_data)
    
    # Process the sensory data
    print("   Processing sensory data...")
    cognitive_core.process_sensory_data(sensory_data_str)
    print("   ✅ Sensory data processed")
    
    # Generate motor command
    print("   Generating motor command...")
    motor_command = cognitive_core.generate_motor_command_from_sensory_data(sensory_data_str)
    print("   ✅ Motor command generated")
    
    # Parse and display the motor command
    motor_data = json.loads(motor_command)
    print(f"\n3. Generated Motor Command:")
    print(f"   Timestamp: {motor_data['timestamp']}")
    print(f"   Number of controls: {len(motor_data['actuators']['controls'])}")
    print(f"   First few controls: {motor_data['actuators']['controls'][:6]}")
    
    # Test with different sensory data (simulating falling)
    print("\n4. Testing with different sensory data (falling scenario)...")
    falling_sensory_data = {
        "timestamp": 1234567890.124,
        "state_vector": [
            0.0, 0.0, 0.5,  # qpos: x, y, z (torso at height 0.5 - falling!)
            0.0, 0.0, 0.0,  # qpos: roll, pitch, yaw
            0.0, 0.0, -0.1, # qvel: linear velocities (falling down)
            0.0, 0.0, 0.0,  # qvel: angular velocities
            1.0, 0.0, 0.0,  # cinert: body orientations
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
    }
    
    falling_sensory_str = json.dumps(falling_sensory_data)
    falling_motor_command = cognitive_core.generate_motor_command_from_sensory_data(falling_sensory_str)
    falling_motor_data = json.loads(falling_motor_command)
    
    print(f"   Falling scenario - First few controls: {falling_motor_data['actuators']['controls'][:6]}")
    
    print("\n✅ Brain-to-body control system test completed successfully!")
    print("\nNext steps:")
    print("1. Start the embodiment interface server: uvicorn brain_architecture.embodiment.embodiment_interface:app --reload")
    print("2. Run the simulation client: python brain_architecture/embodiment/simulation_client.py")
    print("3. Watch the brain take control of the humanoid!")

if __name__ == "__main__":
    test_brain_control()
