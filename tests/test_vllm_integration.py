#!/usr/bin/env python3
"""Test script for vLLM Brain Wrapper integration.

This script tests the new vLLM integration with the Quark brain systems.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_vllm_brain_wrapper():
    """Test the VLLMBrainWrapper functionality."""
    print("🧠 Testing vLLM Brain Wrapper Integration")
    print("=" * 50)
    
    try:
        from brain.externals.vllm_brain_wrapper import VLLMBrainWrapper
        
        # Test model path
        model_path = project_root / "data/models/test_models/gpt2-small"
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("Please run the model download first")
            return False
            
        print(f"📁 Loading model from: {model_path}")
        
        # Initialize wrapper with brain-optimized settings
        wrapper = VLLMBrainWrapper(
            model_path,
            max_model_len=512,
            gpu_memory_utilization=0.3,
            enforce_eager=True
        )
        
        print("✅ vLLM Brain Wrapper initialized successfully!")
        
        # Test basic generation
        print("\n🧪 Testing basic generation...")
        response = wrapper.generate("The human brain processes information by")
        print(f"Response: {response[:100]}...")
        
        # Test brain consciousness generation
        print("\n🧠 Testing brain consciousness generation...")
        consciousness_response = wrapper.brain_consciousness_generate(
            neural_state="High activity in prefrontal cortex, moderate hippocampal activation",
            context="Problem-solving task with memory retrieval"
        )
        print(f"Consciousness: {consciousness_response[:100]}...")
        
        # Test brain-language integration
        print("\n🔗 Testing brain-language integration...")
        brain_data = {
            "activity_level": "high",
            "region_states": {"prefrontal_cortex": "active", "hippocampus": "moderate"},
            "attention_focus": "focused"
        }
        integration_response = wrapper.brain_language_integration(brain_data)
        print(f"Integration: {integration_response[:100]}...")
        
        # Test batch generation
        print("\n📦 Testing batch generation...")
        prompts = [
            "Neural networks learn by",
            "Consciousness emerges when",
            "The brain's attention system"
        ]
        batch_responses = wrapper.generate_batch(prompts)
        for i, response in enumerate(batch_responses):
            print(f"Batch {i+1}: {response[:50]}...")
        
        # Show model info
        print("\n📊 Model Information:")
        info = wrapper.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✅ All vLLM Brain Wrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility():
    """Test backward compatibility with existing code."""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        # Test that existing imports still work
        from brain.externals.vllm_brain_wrapper import LocalLLMWrapper
        print("✅ LocalLLMWrapper alias works")
        
        from brain.externals.vllm_brain_wrapper import integrate_vllm_brain_model
        print("✅ Integration function available")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 vLLM Integration Test Suite")
    print("=" * 60)
    
    # Run tests
    basic_test = test_vllm_brain_wrapper()
    compat_test = test_compatibility()
    
    print("\n" + "=" * 60)
    if basic_test and compat_test:
        print("🎉 All tests passed! vLLM integration ready for production.")
        print("\n📋 Next steps:")
        print("1. Update existing code to use VLLMBrainWrapper")
        print("2. Deploy to Google Cloud with vLLM")
        print("3. Run performance benchmarks")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
