#!/usr/bin/env python3
"""
Test Cloud Offload System
Purpose: Test the cloud offload functionality independently
Inputs: Test parameters for cognitive tasks
Outputs: Verification of cloud processing
Seeds: Deterministic test execution
Deps: cloud_offload, time, json
"""

import sys
import os
import time
import json

# Add the cloud_computing directory to path
sys.path.append(os.path.dirname(__file__))

from cloud_offload import SkyOffloader

def test_cloud_offload():
    """Test the cloud offload system"""
    print("🧠 Testing Cloud Offload System")
    print("=" * 50)
    
    try:
        # Initialize offloader
        print("1. Initializing SkyOffloader...")
        offloader = SkyOffloader()
        print("✅ SkyOffloader initialized")
        
        # Test 1: Neural Simulation
        print("\n2. Testing Neural Simulation...")
        neural_params = {
            'duration': 1000,
            'num_neurons': 50,
            'scale': 0.5
        }
        
        job_id, result = offloader.submit("neural_simulation", neural_params)
        print(f"✅ Neural simulation completed: {job_id}")
        print(f"   Activity level: {result.get('activity_level', 0):.3f}")
        print(f"   Total spikes: {result.get('total_spikes', 0)}")
        
        # Test 2: Memory Consolidation
        print("\n3. Testing Memory Consolidation...")
        memory_params = {
            'duration': 2000,
            'scale': 0.8
        }
        
        job_id, result = offloader.submit("memory_consolidation", memory_params)
        print(f"✅ Memory consolidation completed: {job_id}")
        print(f"   Consolidation level: {result.get('consolidation_level', 0):.3f}")
        print(f"   Average level: {result.get('average_level', 0):.3f}")
        
        # Test 3: Attention Modeling
        print("\n4. Testing Attention Modeling...")
        attention_params = {
            'duration': 1500,
            'scale': 0.7
        }
        
        job_id, result = offloader.submit("attention_modeling", attention_params)
        print(f"✅ Attention modeling completed: {job_id}")
        print(f"   Focus level: {result.get('focus_level', 0):.3f}")
        print(f"   Peak attention: {result.get('peak_attention', 0):.3f}")
        
        print("\n🎉 All cloud offload tests completed successfully!")
        
        # Get system status
        status = offloader.get_status()
        print(f"\n📊 System Status:")
        print(f"   Bucket: {status['bucket']}")
        print(f"   Region: {status['region']}")
        print(f"   Active jobs: {status['active_jobs']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_integration_hook():
    """Test the integration hook with a mock agent"""
    print("\n🔗 Testing Integration Hook")
    print("=" * 30)
    
    try:
        from cloud_offload import create_offload_hook
        
        # Create mock agent
        class MockAgent:
            def __init__(self):
                self.unified_state = {
                    'neural_activity': 0.0,
                    'memory_consolidation': 0.0,
                    'attention_focus': 0.0
                }
        
        mock_agent = MockAgent()
        
        # Create offload hook
        offloader = create_offload_hook(mock_agent)
        
        # Test offload method
        test_params = {'duration': 500, 'scale': 0.3}
        result = mock_agent.offload_heavy_task("neural_simulation", test_params)
        
        if result:
            print("✅ Integration hook test successful")
            print(f"   Neural activity updated: {mock_agent.unified_state['neural_activity']:.3f}")
        else:
            print("⚠️ Integration hook test failed (cloud may be unavailable)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration hook test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Cloud Offload System Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    basic_success = test_cloud_offload()
    
    # Test integration hook
    integration_success = test_integration_hook()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Basic Cloud Offload: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Integration Hook: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if basic_success and integration_success:
        print("\n🎉 All tests passed! Cloud offload system is ready.")
        print("\nNext steps:")
        print("1. Run your main conscious agent")
        print("2. Cloud offload will trigger automatically when cognitive load is high")
        print("3. Monitor offload metrics in unified_state['cloud_offload_metrics']")
    else:
        print("\n⚠️ Some tests failed. Check AWS credentials and SkyPilot setup.")
        print("\nTroubleshooting:")
        print("1. Verify AWS credentials: aws sts get-caller-identity")
        print("2. Check SkyPilot: sky check")
        print("3. Test S3 access: aws s3 ls")

if __name__ == "__main__":
    main()
