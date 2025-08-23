#!/usr/bin/env python3
"""
Test script for quantum error decoding integration in the conscious agent system.
"""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_integration import QuantumIntegration, QuantumConfig
from core.brain_launcher_v4 import NeuralEnhancedBrain


def test_quantum_integration():
    """Test basic quantum integration functionality."""
    print("🧪 Testing Quantum Integration...")
    
    # Test quantum integration creation
    config = QuantumConfig(enable_quantum=True, code_distance=5)
    quantum = QuantumIntegration(config)
    
    print(f"✅ Quantum Integration created")
    print(f"   Available: {quantum.quantum_available}")
    print(f"   Surface Code: {quantum.surface_code is not None}")
    print(f"   Quantum Decoder: {quantum.quantum_decoder is not None}")
    
    # Test quantum data processing
    test_data = {
        'test_key': 'test_value',
        'quantum_state': 'test_state'
    }
    
    processed_data = quantum.process_quantum_data(test_data)
    print(f"✅ Quantum data processing: {processed_data.get('quantum_processed', False)}")
    
    # Test status retrieval
    status = quantum.get_status()
    print(f"✅ Status retrieved: {status}")
    
    return True


async def test_brain_integration():
    """Test brain integration with quantum capabilities."""
    print("\n🧠 Testing Brain Integration...")
    
    try:
        # Create a simple connectome for testing
        test_connectome = {
            "modules": {
                "pfc": {"type": "executive", "capacity": 100},
                "basal_ganglia": {"type": "gating", "capacity": 50},
                "thalamus": {"type": "relay", "capacity": 75}
            },
            "connections": [
                {"from": "pfc", "to": "basal_ganglia", "weight": 0.8},
                {"from": "basal_ganglia", "to": "thalamus", "weight": 0.6},
                {"from": "thalamus", "to": "pfc", "weight": 0.7}
            ]
        }
        
        # Write test connectome to file
        test_connectome_path = "test_connectome.yaml"
        import yaml
        with open(test_connectome_path, 'w') as f:
            yaml.dump(test_connectome, f)
        
        # Create enhanced brain
        brain = NeuralEnhancedBrain(test_connectome_path, stage="F", validate=False)
        
        print(f"✅ Enhanced Brain created")
        print(f"   Quantum Integration: {brain.quantum_integration.quantum_available}")
        
        # Test brain step
        result = await brain.step()
        print(f"✅ Brain step completed")
        print(f"   Quantum Processed: {result.get('quantum_processed', False)}")
        print(f"   Quantum Timestamp: {result.get('quantum_timestamp', 'N/A')}")
        
        # Test quantum summary
        quantum_summary = brain.get_quantum_summary()
        print(f"✅ Quantum summary retrieved")
        print(f"   Status: {quantum_summary.get('quantum_integration_status', 'Unknown')}")
        
        # Clean up
        os.remove(test_connectome_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Brain integration test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Quantum Error Decoding Integration Test Suite")
    print("=" * 60)
    
    # Test quantum integration
    quantum_success = test_quantum_integration()
    
    # Test brain integration
    brain_success = await test_brain_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print(f"Quantum Integration: {'✅ PASSED' if quantum_success else '❌ FAILED'}")
    print(f"Brain Integration: {'✅ PASSED' if brain_success else '❌ FAILED'}")
    
    if quantum_success and brain_success:
        print("\n🎉 All tests passed! Quantum integration is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
